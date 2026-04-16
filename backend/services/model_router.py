"""
Model Router — production-grade multi-provider LLM gateway.

Routes requests through available providers with:
 • Per-provider local quota guards aligned to real API limits
 • Smart error classification (rate-limit ≠ quota exhaustion)
 • Capped cooldown timers to prevent runaway stalls
 • Structured logging for every provider decision
 • Automatic fallback chain: Gemini → OpenRouter → Groq → HuggingFace
"""

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from services.config import Settings
from services.coordination import CoordinationStore

logger = logging.getLogger("model_router")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RequestBudget:
    max_input_chars: int
    max_output_tokens: int = 1024


@dataclass(slots=True)
class QuotaSnapshot:
    requests_used: int
    request_limit: int
    tokens_used: int
    token_limit: int
    retry_after_seconds: int


@dataclass(slots=True)
class RetryDecision:
    retry_same_provider: bool
    switch_provider: bool
    delay_seconds: float
    reason: str


@dataclass(slots=True)
class ProviderPolicy:
    name: str
    provider_type: str
    model: str
    task_types: tuple[str, ...]
    priority: int
    request_limit_per_minute: int
    token_limit_per_minute: int
    max_parallel_requests: int
    enabled: bool = True
    base_url: str | None = None


@dataclass(slots=True)
class ProviderState:
    policy: ProviderPolicy
    semaphore: asyncio.Semaphore


@dataclass(slots=True)
class ProviderBlock:
    policy: ProviderPolicy
    reason: str
    ttl_seconds: int
    details: dict[str, Any] | None = None


class ProviderUnavailableError(RuntimeError):
    def __init__(self, task_type: str, blocked: list[ProviderBlock], supported: list[str], message: str) -> None:
        super().__init__(message)
        self.task_type = task_type
        self.blocked = blocked
        self.supported = supported

    @property
    def retry_after_seconds(self) -> int:
        positive_ttls = [block.ttl_seconds for block in self.temporarily_blocked if block.ttl_seconds > 0]
        if not positive_ttls:
            positive_ttls = [block.ttl_seconds for block in self.blocked if block.ttl_seconds > 0]
        return max(1, min(positive_ttls)) if positive_ttls else 1

    @property
    def reasons(self) -> list[str]:
        return [block.reason for block in self.blocked]

    @property
    def temporarily_blocked(self) -> list[ProviderBlock]:
        return [block for block in self.blocked if block.reason in {"quota", "local_quota_guard", "rate_limit"}]

    @property
    def has_configuration_only_failure(self) -> bool:
        return bool(self.blocked) and not self.temporarily_blocked

    @property
    def should_wait_for_retryable_provider(self) -> bool:
        return bool(self.temporarily_blocked)


class ProviderClient(Protocol):
    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        ...

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        ...

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        ...


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def estimate_tokens_from_messages(messages: list[BaseMessage]) -> int:
    total_chars = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            total_chars += len(" ".join(str(item) for item in content))
        else:
            total_chars += len(str(content))
    return max(1, total_chars // 4)


def trim_messages_for_budget(messages: list[BaseMessage], budget: RequestBudget) -> list[BaseMessage]:
    if not messages:
        return []

    reversed_messages = list(reversed(messages))
    kept: list[BaseMessage] = []
    total_chars = 0

    for message in reversed_messages:
        content = getattr(message, "content", "")
        content_text = str(content)
        if total_chars + len(content_text) > budget.max_input_chars and kept:
            remaining = max(0, budget.max_input_chars - total_chars)
            if remaining <= 0:
                continue
            cloned = message.model_copy(deep=True)
            cloned.content = content_text[-remaining:]
            kept.append(cloned)
            total_chars += len(str(cloned.content))
            break

        kept.append(message)
        total_chars += len(content_text)
        if total_chars >= budget.max_input_chars:
            break

    return list(reversed(kept))


def classify_exception(error: Exception) -> str:
    """Classify a provider error into an actionable category.

    Categories:
      - configuration: wrong model, 404 — provider will never work until config is fixed.
      - quota: true billing/credit exhaustion — provider won't work until plan is upgraded.
      - rate_limit: per-minute/per-day throttle — provider recovers in seconds to minutes.
      - transient: server errors, timeouts — immediate retry is reasonable.
      - unknown: fallback.

    ORDER MATTERS: Rate-limit must be checked BEFORE configuration because provider
    error messages often contain model paths like 'models/llama-3.3-70b...' alongside
    rate-limit information.  If configuration were checked first, the 'models/' substring
    would cause a false match.
    """
    error_text = str(error).lower()

    # --- Rate limit FIRST (per-minute or per-day throttle, recovers automatically) ---
    #     Check this before configuration because rate-limit errors from providers
    #     often contain model names like 'models/xxx' which would false-match config.
    rate_limit_terms = (
        "rate limit", "rate_limit", "429", "too many requests", "throttle",
        "resource exhausted", "resource has been exhausted",
        "quota exceeded", "requests per minute", "tokens per minute",
        "rpm", "tpm", "retry after", "too many tokens",
    )
    if any(term in error_text for term in rate_limit_terms):
        return "rate_limit"

    # --- True billing / credit exhaustion (needs plan upgrade) ---
    billing_terms = (
        "credits depleted", "payment required", "402",
        "plan limit", "free tier exceeded", "subscription required",
        "billing not enabled", "billing account", "spending limit",
        "purchase more", "insufficient credits", "usage limit",
    )
    if any(term in error_text for term in billing_terms):
        return "quota"

    # --- Configuration errors (permanent until config change) ---
    #     More specific patterns to avoid false positives with model name substrings.
    if any(term in error_text for term in (
        "not_found", "model is not found", "model not found",
        "not supported for generatecontent", "unsupported model",
        "invalid model", "unknown model", "does not exist",
    )):
        return "configuration"
    # HTTP 404 specifically (but NOT if it's a 429 or other rate-limit context)
    if "404" in error_text and "rate" not in error_text:
        return "configuration"
    # 403 forbidden (but not if it's a rate-limit 403)
    if ("permission denied" in error_text or "403" in error_text) and "rate" not in error_text:
        return "configuration"

    # --- Transient server errors ---
    if any(term in error_text for term in (
        "timeout", "temporarily unavailable", "unavailable",
        "503", "500", "502", "504", "connection",
        "internal server error", "service unavailable",
    )):
        return "transient"

    return "unknown"


def parse_retry_after_seconds(error: Exception) -> int:
    """Extract Retry-After from HTTP response headers or error text."""
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and str(retry_after).isdigit():
            return max(1, int(retry_after))

    match = re.search(r"retry[- ]after[=: ]+(\d+)", str(error), flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 0


def _extract_error_payload(error: Exception) -> list[Any]:
    payloads: list[Any] = []
    body = getattr(error, "body", None)
    if body is not None:
        payloads.append(body)

    response = getattr(error, "response", None)
    if response is not None:
        payloads.append(getattr(response, "text", None))
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                payloads.append(json_fn())
            except Exception:
                pass

    payloads.append(str(error))
    return [payload for payload in payloads if payload not in (None, "")]


def _extract_failed_generation_texts(error: Exception) -> list[str]:
    texts: list[str] = []
    for payload in _extract_error_payload(error):
        if isinstance(payload, dict):
            nested = payload.get("error", payload)
            # Guard: nested might be a string like {"error": "some message"}
            if isinstance(nested, dict):
                failed_generation = nested.get("failed_generation")
                if isinstance(failed_generation, str):
                    texts.append(failed_generation)
            elif isinstance(nested, str):
                texts.append(nested)
        elif isinstance(payload, str):
            texts.append(payload)
    return texts


def _extract_text_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _coerce_tool_args(tool: type, args: Any) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    try:
        validated = tool(**args)
    except ValidationError:
        return None
    return validated.model_dump() if hasattr(validated, "model_dump") else dict(args)


def coerce_tool_calls_from_text(text: str, tools: list[type]) -> AIMessage | None:
    tool_map = {tool.__name__: tool for tool in tools}
    candidates: list[tuple[str, Any]] = []

    for match in re.finditer(r"<(?:function|tool)=(?P<name>[A-Za-z0-9_]+)>\s*(?P<args>.*?)\s*</(?:function|tool)>", text, flags=re.DOTALL):
        candidates.append((match.group("name"), match.group("args").strip()))

    cleaned_text = _strip_code_fences(text)
    if not candidates and cleaned_text:
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                tool_name = parsed.get("tool_name") or parsed.get("name")
                if isinstance(parsed.get("function"), dict):
                    tool_name = tool_name or parsed["function"].get("name")
                arguments = (
                    parsed.get("arguments")
                    or parsed.get("args")
                    or parsed.get("parameters")
                    or (parsed.get("function") or {}).get("arguments")
                )
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                if isinstance(tool_name, str):
                    candidates.append((tool_name, arguments))
        except json.JSONDecodeError:
            pass

    tool_calls = []
    for name, raw_args in candidates:
        tool = tool_map.get(name)
        if tool is None:
            continue
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
        normalized_args = _coerce_tool_args(tool, raw_args)
        if normalized_args is None:
            continue
        tool_calls.append(
            {
                "name": name,
                "args": normalized_args,
                "id": f"recovered_{uuid.uuid4().hex[:12]}",
            }
        )

    if not tool_calls:
        return None

    return AIMessage(content="", tool_calls=tool_calls)


async def repair_tool_call_with_plain_json(
    llm: Any,
    messages: list[BaseMessage],
    tools: list[type],
    budget: RequestBudget,
) -> AIMessage | None:
    tool_summaries = []
    for tool in tools:
        field_names = ", ".join(tool.model_fields.keys()) if hasattr(tool, "model_fields") else ""
        tool_summaries.append(f"- {tool.__name__}: fields [{field_names}]")

    repair_messages = trim_messages_for_budget(
        messages
        + [
            SystemMessage(
                content=(
                    "Your previous attempt produced invalid tool-call syntax. "
                    "Choose exactly one next action and respond with ONLY a JSON object of the form "
                    '{"tool_name":"<ToolName>","arguments":{...}}. '
                    "Do not use <function=...>, XML, markdown fences, or prose.\n"
                    "Allowed tools:\n"
                    + "\n".join(tool_summaries)
                )
            )
        ],
        RequestBudget(max_input_chars=max(1200, min(budget.max_input_chars, 4000)), max_output_tokens=min(400, budget.max_output_tokens)),
    )
    response = await llm.ainvoke(repair_messages)
    return coerce_tool_calls_from_text(_extract_text_content(response.content), tools)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class GeminiProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._policy = policy
        self._api_key = api_key
        # max_retries=0 disables the SDK's internal retry loop (which wastes 30-40s
        # retrying 429s).  Our ModelRouter handles retries at a higher level and can
        # switch providers instead of burning time on the same exhausted provider.
        self._llm = ChatGoogleGenerativeAI(
            model=policy.model,
            google_api_key=api_key,
            temperature=0,
            max_retries=0,
        )

    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        return await self._llm.ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        return await self._llm.bind_tools(tools).ainvoke(trim_messages_for_budget(messages, budget))


class GroqProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._policy = policy
        self._api_key = api_key
        self._llm = ChatGroq(model=policy.model, api_key=api_key, temperature=0)

    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        return await self._llm.ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        trimmed_messages = trim_messages_for_budget(messages, budget)
        try:
            return await self._llm.bind_tools(tools, tool_choice="any").ainvoke(trimmed_messages)
        except Exception as error:
            for text in _extract_failed_generation_texts(error):
                recovered = coerce_tool_calls_from_text(text, tools)
                if recovered is not None:
                    return recovered
            try:
                repaired = await repair_tool_call_with_plain_json(self._llm, trimmed_messages, tools, budget)
            except Exception:
                repaired = None
            if repaired is not None:
                return repaired
            raise


class OpenRouterProvider:
    """OpenRouter provider — uses the OpenAI-compatible API with a wide model catalog."""

    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._policy = policy
        # max_retries=0: let our ModelRouter handle retries instead of the SDK
        # wasting 7+ seconds retrying 429s internally.
        self._llm = ChatOpenAI(
            model=policy.model,
            api_key=api_key,
            base_url=policy.base_url,
            temperature=0,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://deep-research-agent.local",
                "X-Title": "Deep Research Agent",
            },
        )

    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        return await self._llm.ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        trimmed_messages = trim_messages_for_budget(messages, budget)
        try:
            return await self._llm.bind_tools(tools, tool_choice="any").ainvoke(trimmed_messages)
        except Exception as error:
            for text in _extract_failed_generation_texts(error):
                recovered = coerce_tool_calls_from_text(text, tools)
                if recovered is not None:
                    return recovered
            try:
                repaired = await repair_tool_call_with_plain_json(self._llm, trimmed_messages, tools, budget)
            except Exception:
                repaired = None
            if repaired is not None:
                return repaired
            raise


class HuggingFaceProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        # max_retries=0: let our ModelRouter handle retries.
        self._llm = ChatOpenAI(
            model=policy.model,
            api_key=api_key,
            base_url=policy.base_url,
            temperature=0,
            max_retries=0,
        )

    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        return await self._llm.ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        trimmed_messages = trim_messages_for_budget(messages, budget)
        try:
            # HuggingFace router doesn't support tool_choice="any", use plain bind_tools
            return await self._llm.bind_tools(tools).ainvoke(trimmed_messages)
        except Exception as error:
            for text in _extract_failed_generation_texts(error):
                recovered = coerce_tool_calls_from_text(text, tools)
                if recovered is not None:
                    return recovered
            try:
                repaired = await repair_tool_call_with_plain_json(self._llm, trimmed_messages, tools, budget)
            except Exception:
                repaired = None
            if repaired is not None:
                return repaired
            raise


class CerebrasProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._llm = ChatOpenAI(
            model=policy.model,
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=0,
            max_retries=0,
        )

    async def generate_text(self, messages: list[BaseMessage], budget: RequestBudget) -> AIMessage:
        return await self._llm.ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_structured(self, schema: type, messages: list[BaseMessage], budget: RequestBudget) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(trim_messages_for_budget(messages, budget))

    async def generate_tool_calling(
        self,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
    ) -> AIMessage:
        trimmed_messages = trim_messages_for_budget(messages, budget)
        try:
            return await self._llm.bind_tools(tools).ainvoke(trimmed_messages)
        except Exception as error:
            for text in _extract_failed_generation_texts(error):
                recovered = coerce_tool_calls_from_text(text, tools)
                if recovered is not None:
                    return recovered
            try:
                repaired = await repair_tool_call_with_plain_json(self._llm, trimmed_messages, tools, budget)
            except Exception:
                repaired = None
            if repaired is not None:
                return repaired
            raise


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    def __init__(
        self,
        coordination_store: CoordinationStore,
        settings: Settings,
        provider_clients: dict[str, ProviderClient] | None = None,
    ) -> None:
        self._coordination_store = coordination_store
        self._settings = settings
        self._states: dict[str, ProviderState] = {}
        self._clients: dict[str, ProviderClient] = provider_clients or {}
        self._rr_counters: dict[str, int] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        if self._clients:
            for name, client in self._clients.items():
                policy = ProviderPolicy(
                    name=name,
                    provider_type="test",
                    model=name,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=60,
                    token_limit_per_minute=1_000_000,
                    max_parallel_requests=4,
                )
                self._states[name] = ProviderState(policy=policy, semaphore=asyncio.Semaphore(policy.max_parallel_requests))
            return

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
        openrouter_api_key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")

        policies: list[ProviderPolicy] = []

        # Priority 0 — Groq Primary (llama-3.3-70b — best quality)
        if groq_api_key:
            policies.append(
                ProviderPolicy(
                    name="groq",
                    provider_type="groq",
                    model=self._settings.groq_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.groq_request_limit_per_minute,
                    token_limit_per_minute=self._settings.groq_token_limit_per_minute,
                    max_parallel_requests=self._settings.groq_max_parallel_requests,
                )
            )
            self._clients["groq"] = GroqProvider(policies[-1], groq_api_key)

        # Priority 0 — Groq Secondary (qwen3-32b — separate rate limits)
        if groq_api_key and self._settings.groq_secondary_model:
            policies.append(
                ProviderPolicy(
                    name="groq_secondary",
                    provider_type="groq",
                    model=self._settings.groq_secondary_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.groq_secondary_request_limit_per_minute,
                    token_limit_per_minute=self._settings.groq_secondary_token_limit_per_minute,
                    max_parallel_requests=1,
                )
            )
            self._clients["groq_secondary"] = GroqProvider(policies[-1], groq_api_key)

        # Priority 0 — Groq Tertiary (llama-4-scout — separate rate limits)
        if groq_api_key and self._settings.groq_tertiary_model:
            policies.append(
                ProviderPolicy(
                    name="groq_tertiary",
                    provider_type="groq",
                    model=self._settings.groq_tertiary_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.groq_tertiary_request_limit_per_minute,
                    token_limit_per_minute=self._settings.groq_tertiary_token_limit_per_minute,
                    max_parallel_requests=1,
                )
            )
            self._clients["groq_tertiary"] = GroqProvider(policies[-1], groq_api_key)

        # Priority 0 — Cerebras Primary (llama3.1-8b)
        if cerebras_api_key:
            policies.append(
                ProviderPolicy(
                    name="cerebras",
                    provider_type="cerebras",
                    model=self._settings.cerebras_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.cerebras_request_limit_per_minute,
                    token_limit_per_minute=self._settings.cerebras_token_limit_per_minute,
                    max_parallel_requests=self._settings.cerebras_max_parallel_requests,
                )
            )
            self._clients["cerebras"] = CerebrasProvider(policies[-1], cerebras_api_key)

        # Priority 0 — Cerebras Secondary (qwen-3-235b)
        if cerebras_api_key and self._settings.cerebras_secondary_model:
            policies.append(
                ProviderPolicy(
                    name="cerebras_secondary",
                    provider_type="cerebras",
                    model=self._settings.cerebras_secondary_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.cerebras_secondary_request_limit_per_minute,
                    token_limit_per_minute=self._settings.cerebras_secondary_token_limit_per_minute,
                    max_parallel_requests=self._settings.cerebras_max_parallel_requests,
                )
            )
            self._clients["cerebras_secondary"] = CerebrasProvider(policies[-1], cerebras_api_key)

        # Priority 1 — Gemini (high quality, limited daily quota on free tier)
        if gemini_api_key:
            policies.append(
                ProviderPolicy(
                    name="gemini",
                    provider_type="gemini",
                    model=self._settings.gemini_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=1,
                    request_limit_per_minute=self._settings.gemini_request_limit_per_minute,
                    token_limit_per_minute=self._settings.gemini_token_limit_per_minute,
                    max_parallel_requests=self._settings.gemini_max_parallel_requests,
                )
            )
            self._clients["gemini"] = GeminiProvider(policies[-1], gemini_api_key)

        # Priority 2 — OpenRouter (free shared models, may be rate-limited upstream)
        if openrouter_api_key:
            policies.append(
                ProviderPolicy(
                    name="openrouter",
                    provider_type="openrouter",
                    model=self._settings.openrouter_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=2,
                    request_limit_per_minute=self._settings.openrouter_request_limit_per_minute,
                    token_limit_per_minute=self._settings.openrouter_token_limit_per_minute,
                    max_parallel_requests=self._settings.openrouter_max_parallel_requests,
                    base_url=self._settings.openrouter_base_url,
                )
            )
            self._clients["openrouter"] = OpenRouterProvider(policies[-1], openrouter_api_key)

        # Priority 3 — HuggingFace (free API, additional fallback)
        if hf_api_key and self._settings.enable_huggingface_router:
            policies.append(
                ProviderPolicy(
                    name="huggingface",
                    provider_type="huggingface",
                    model=self._settings.huggingface_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=3,
                    request_limit_per_minute=self._settings.huggingface_request_limit_per_minute,
                    token_limit_per_minute=self._settings.huggingface_token_limit_per_minute,
                    max_parallel_requests=self._settings.huggingface_max_parallel_requests,
                    base_url=self._settings.huggingface_base_url,
                )
            )
            self._clients["huggingface"] = HuggingFaceProvider(policies[-1], hf_api_key)

        for policy in policies:
            self._states[policy.name] = ProviderState(
                policy=policy,
                semaphore=asyncio.Semaphore(policy.max_parallel_requests),
            )

        # Log registered providers at startup
        provider_summary = ", ".join(
            f"{p.name}({p.model}, pri={p.priority})" for p in policies
        ) or "NONE"
        logger.info("🚀 ModelRouter initialized — providers: %s", provider_summary)

    # --- Public API ---

    async def generate_text(
        self,
        task_type: str,
        messages: list[BaseMessage],
        budget: RequestBudget,
        trace_id: str,
    ) -> AIMessage:
        response = await self._execute(task_type, "text", messages, budget, trace_id, schema=None, tools=None)
        if hasattr(response, "content") and isinstance(response.content, str):
            # Clean out <think>...</think> blocks from models like Qwen/DeepSeek
            cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
            response.content = cleaned_content
        return response

    async def generate_structured(
        self,
        task_type: str,
        schema: type,
        messages: list[BaseMessage],
        budget: RequestBudget,
        trace_id: str,
    ) -> Any:
        return await self._execute(task_type, "structured", messages, budget, trace_id, schema=schema, tools=None)

    async def generate_tool_calling(
        self,
        task_type: str,
        messages: list[BaseMessage],
        tools: list[type],
        budget: RequestBudget,
        trace_id: str,
    ) -> AIMessage:
        return await self._execute(task_type, "tool_calling", messages, budget, trace_id, schema=None, tools=tools)

    # --- Core execution loop ---

    async def _execute(
        self,
        task_type: str,
        mode: str,
        messages: list[BaseMessage],
        budget: RequestBudget,
        trace_id: str,
        schema: type | None,
        tools: list[type] | None,
    ) -> Any:
        started_at = time.monotonic()
        last_error: Exception | None = None
        attempt_number = 0

        while time.monotonic() - started_at <= self._settings.provider_unavailable_wait_seconds:
            attempt_number += 1
            candidates, blocked = await self._select_candidates(task_type)

            if not candidates:
                supported = [
                    state.policy.name
                    for state in self._states.values()
                    if task_type in state.policy.task_types and state.policy.enabled
                ]
                last_error = ProviderUnavailableError(
                    task_type=task_type,
                    blocked=blocked,
                    supported=supported,
                    message=self._format_provider_unavailable_message(task_type, blocked),
                )
                if (
                    blocked
                    and last_error.should_wait_for_retryable_provider
                    and last_error.retry_after_seconds > self._settings.provider_unavailable_wait_seconds
                ):
                    logger.error(
                        "🛑 [%s] All providers blocked beyond wait limit (%ds > %ds) — failing immediately",
                        task_type, last_error.retry_after_seconds,
                        int(self._settings.provider_unavailable_wait_seconds),
                    )
                    raise last_error

                wait_time = self._next_provider_check_delay(blocked)
                remaining = self._settings.provider_unavailable_wait_seconds - (time.monotonic() - started_at)
                if remaining <= 0:
                    break

                blocked_summary = ", ".join(
                    f"{b.policy.name}:{b.reason}({b.ttl_seconds}s)" for b in blocked
                ) or "none"
                logger.warning(
                    "⏳ [%s] No available providers (attempt %d) — blocked: [%s] — rechecking in %.0fs",
                    task_type, attempt_number, blocked_summary, min(wait_time, remaining),
                )
                await asyncio.sleep(min(wait_time, remaining))
                continue

            for index, state in enumerate(candidates):
                client = self._clients[state.policy.name]
                result = await self._attempt_provider(
                    state=state,
                    client=client,
                    mode=mode,
                    schema=schema,
                    tools=tools,
                    messages=messages,
                    budget=budget,
                    trace_id=trace_id,
                )
                if isinstance(result, Exception):
                    last_error = result
                    if index < len(candidates) - 1:
                        next_provider = candidates[index + 1].policy.name
                        logger.info(
                            "🔀 [%s] Switching from %s → %s for %s",
                            trace_id[:8], state.policy.name, next_provider, task_type,
                        )
                        await self._coordination_store.increment_job_metric(trace_id, "provider_switch_count", 1)
                    continue
                return result

            await asyncio.sleep(1.0)

        if last_error is None:
            raise RuntimeError("No providers are currently available for this request.")
        raise last_error

    async def _select_candidates(self, task_type: str) -> tuple[list[ProviderState], list[ProviderBlock]]:
        eligible: list[tuple[int, int, int, ProviderState]] = []
        blocked: list[ProviderBlock] = []

        for state in self._states.values():
            policy = state.policy
            if not policy.enabled or task_type not in policy.task_types:
                continue

            cooldown_ttl = await self._coordination_store.get_ttl_seconds(f"provider:cooldown:{policy.name}")
            circuit_ttl = await self._coordination_store.get_ttl_seconds(f"provider:circuit:{policy.name}")
            if cooldown_ttl > 0 or circuit_ttl > 0:
                reason_prefix = "provider:cooldown" if cooldown_ttl > 0 else "provider:circuit"
                payload = await self._coordination_store.get_json(f"{reason_prefix}:{policy.name}")
                blocked.append(
                    ProviderBlock(
                        policy=policy,
                        reason=str((payload or {}).get("reason", "cooldown" if cooldown_ttl > 0 else "circuit")),
                        ttl_seconds=max(cooldown_ttl, circuit_ttl),
                        details=payload if isinstance(payload, dict) else None,
                    )
                )
                continue

            rr_value = self._rr_counters.get(task_type, 0)
            inflight = max(0, policy.max_parallel_requests - state.semaphore._value)
            eligible.append((policy.priority, inflight, rr_value, state))

        eligible.sort(key=lambda item: (item[0], item[1], item[2]))
        self._rr_counters[task_type] = self._rr_counters.get(task_type, 0) + 1
        return [item[-1] for item in eligible], blocked

    async def _attempt_provider(
        self,
        state: ProviderState,
        client: ProviderClient,
        mode: str,
        schema: type | None,
        tools: list[type] | None,
        messages: list[BaseMessage],
        budget: RequestBudget,
        trace_id: str,
    ) -> Any | Exception:
        policy = state.policy
        # Only count input tokens for the local quota guard estimate.
        # Including budget.max_output_tokens (which is just a ceiling, e.g. 6000)
        # would cause the guard to block requests that are well within the provider's
        # actual capacity.  The provider itself enforces output limits.
        input_tokens = estimate_tokens_from_messages(trim_messages_for_budget(messages, budget))
        token_cost = input_tokens + min(budget.max_output_tokens, 500)  # assume ~500 output for guard purposes

        # Local quota guard — prevent sending requests we know will 429
        reservation = await self._coordination_store.reserve_quota(
            key=f"provider:{policy.name}",
            request_cost=1,
            token_cost=token_cost,
            request_limit=policy.request_limit_per_minute,
            token_limit=policy.token_limit_per_minute,
            window_seconds=60,
        )
        if not reservation.allowed:
            # Cap local guard cooldown to the rate-limit max so it never exceeds 60s
            guard_ttl = min(reservation.retry_after_seconds, self._settings.rate_limit_max_cooldown_seconds)
            await self._coordination_store.set_json(
                f"provider:cooldown:{policy.name}",
                {"reason": "local_quota_guard"},
                ttl_seconds=guard_ttl,
            )
            logger.warning(
                "⚠️  [%s/%s] Local quota guard — %d/%d reqs, %d/%d tokens used — cooldown %ds",
                policy.name, policy.model,
                reservation.requests_used, policy.request_limit_per_minute,
                reservation.tokens_used, policy.token_limit_per_minute,
                guard_ttl,
            )
            return RuntimeError(f"{policy.name} local quota guard triggered.")

        request_start = time.monotonic()
        async with state.semaphore:
            logger.info(
                "📡 [%s/%s] %s request starting (mode=%s, budget=%d chars / %d tokens)",
                policy.name, policy.model, mode, mode,
                budget.max_input_chars, budget.max_output_tokens,
            )
            try:
                if mode == "text":
                    result = await client.generate_text(messages, budget)
                elif mode == "structured":
                    result = await client.generate_structured(schema, messages, budget)
                elif mode == "tool_calling":
                    result = await client.generate_tool_calling(messages, tools or [], budget)
                else:
                    raise ValueError(f"Unsupported model router mode: {mode}")

                elapsed_ms = int((time.monotonic() - request_start) * 1000)
                logger.info(
                    "✅ [%s/%s] %s completed in %dms",
                    policy.name, policy.model, mode, elapsed_ms,
                )
                return result

            except Exception as error:
                elapsed_ms = int((time.monotonic() - request_start) * 1000)
                classification = classify_exception(error)
                logger.error(
                    "❌ [%s/%s] %s failed after %dms — class=%s — %s",
                    policy.name, policy.model, mode, elapsed_ms,
                    classification, str(error)[:300],
                )

                decision = await self._build_retry_decision(policy.name, error)

                # One retry on same provider for transient errors
                if decision.retry_same_provider:
                    await self._coordination_store.increment_job_metric(trace_id, "retry_count", 1)
                    logger.info(
                        "🔄 [%s] Retrying same provider in %.1fs (reason=%s)",
                        policy.name, decision.delay_seconds, decision.reason,
                    )
                    await asyncio.sleep(decision.delay_seconds)
                    try:
                        if mode == "text":
                            result = await client.generate_text(messages, budget)
                        elif mode == "structured":
                            result = await client.generate_structured(schema, messages, budget)
                        else:
                            result = await client.generate_tool_calling(messages, tools or [], budget)

                        logger.info("✅ [%s/%s] Retry succeeded", policy.name, policy.model)
                        return result
                    except Exception as second_error:
                        second_classification = classify_exception(second_error)
                        logger.error(
                            "❌ [%s/%s] Retry also failed — class=%s — %s",
                            policy.name, policy.model,
                            second_classification, str(second_error)[:300],
                        )
                        error = second_error
                        decision = await self._build_retry_decision(policy.name, second_error)

                # Apply cooldown/circuit-breaker based on error class
                if decision.reason == "rate_limit":
                    # Use Retry-After header if available, else use configured cooldown, cap it
                    header_ttl = parse_retry_after_seconds(error)
                    ttl = header_ttl or max(1, int(decision.delay_seconds))
                    ttl = min(ttl, self._settings.rate_limit_max_cooldown_seconds)
                    await self._coordination_store.set_json(
                        f"provider:cooldown:{policy.name}",
                        {"reason": "rate_limit"},
                        ttl_seconds=ttl,
                    )
                    logger.warning(
                        "🚫 [%s] Rate-limited — cooldown %ds (header=%ds, configured=%ds, cap=%ds)",
                        policy.name, ttl, header_ttl, self._settings.provider_cooldown_seconds,
                        self._settings.rate_limit_max_cooldown_seconds,
                    )
                elif decision.reason == "quota":
                    ttl = self._settings.quota_cooldown_seconds
                    await self._coordination_store.set_json(
                        f"provider:circuit:{policy.name}",
                        {"reason": "quota"},
                        ttl_seconds=ttl,
                    )
                    logger.error(
                        "🚫 [%s] Billing/quota exhausted — circuit-breaker %ds",
                        policy.name, ttl,
                    )
                elif decision.reason == "configuration":
                    ttl = self._settings.quota_cooldown_seconds
                    await self._coordination_store.set_json(
                        f"provider:circuit:{policy.name}",
                        {"reason": "configuration", "error": str(error)[:500]},
                        ttl_seconds=ttl,
                    )
                    logger.error(
                        "🚫 [%s] Configuration error (bad model/API key) — circuit-breaker %ds — %s",
                        policy.name, ttl, str(error)[:200],
                    )

                return error

    async def _build_retry_decision(self, provider_name: str, error: Exception) -> RetryDecision:
        classification = classify_exception(error)
        if classification == "rate_limit":
            retry_after = parse_retry_after_seconds(error) or self._settings.provider_cooldown_seconds
            # Cap rate-limit delays so they never stall the system
            retry_after = min(retry_after, self._settings.rate_limit_max_cooldown_seconds)
            return RetryDecision(
                retry_same_provider=False,
                switch_provider=True,
                delay_seconds=retry_after,
                reason="rate_limit",
            )
        if classification == "quota":
            return RetryDecision(
                retry_same_provider=False,
                switch_provider=True,
                delay_seconds=self._settings.quota_cooldown_seconds,
                reason="quota",
            )
        if classification == "transient":
            delay = self._settings.transient_retry_base_seconds + random.random() * self._settings.transient_retry_jitter_seconds
            return RetryDecision(
                retry_same_provider=True,
                switch_provider=True,
                delay_seconds=delay,
                reason="transient",
            )
        if classification == "configuration":
            return RetryDecision(
                retry_same_provider=False,
                switch_provider=True,
                delay_seconds=self._settings.quota_cooldown_seconds,
                reason="configuration",
            )
        return RetryDecision(
            retry_same_provider=False,
            switch_provider=True,
            delay_seconds=0,
            reason="unknown",
        )

    def _next_provider_check_delay(self, blocked: list[ProviderBlock]) -> float:
        ttl_values = [block.ttl_seconds for block in blocked if block.ttl_seconds > 0]
        if not ttl_values:
            return max(1.0, self._settings.provider_poll_interval_seconds)
        return max(1.0, min(min(ttl_values), self._settings.provider_poll_interval_seconds))

    def _format_provider_unavailable_message(self, task_type: str, blocked: list[ProviderBlock]) -> str:
        if not self._states:
            return "No model providers are configured. Set at least one provider API key."

        supported = [state.policy.name for state in self._states.values() if task_type in state.policy.task_types and state.policy.enabled]
        if not supported:
            return f"No configured providers support task type '{task_type}'."

        if not blocked:
            return f"No providers are currently available for task type '{task_type}'."

        parts = []
        for block in blocked:
            detail = f"{block.policy.name}: {block.reason}"
            if block.ttl_seconds > 0:
                detail += f" ({block.ttl_seconds}s)"
            parts.append(detail)
        return f"All providers are temporarily unavailable for task type '{task_type}': " + ", ".join(parts)
