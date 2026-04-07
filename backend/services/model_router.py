import asyncio
import json
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
    error_text = str(error).lower()
    if any(term in error_text for term in ("not_found", "404", "model is not found", "models/", "not supported for generatecontent", "unsupported model")):
        return "configuration"
    if any(term in error_text for term in ("rate limit", "429", "too many requests", "throttle")):
        if any(term in error_text for term in ("quota", "credit", "billing", "exhausted")):
            return "quota"
        return "rate_limit"
    if any(term in error_text for term in ("quota", "credits depleted", "billing", "exhausted")):
        return "quota"
    if any(term in error_text for term in ("timeout", "temporarily unavailable", "unavailable", "503", "500", "connection")):
        return "transient"
    return "unknown"


def parse_retry_after_seconds(error: Exception) -> int:
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
            failed_generation = nested.get("failed_generation")
            if isinstance(failed_generation, str):
                texts.append(failed_generation)
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


class GeminiProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._policy = policy
        self._api_key = api_key
        self._llm = ChatGoogleGenerativeAI(model=policy.model, google_api_key=api_key, temperature=0)

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


class HuggingFaceProvider:
    def __init__(self, policy: ProviderPolicy, api_key: str) -> None:
        self._llm = ChatOpenAI(
            model=policy.model,
            api_key=api_key,
            base_url=policy.base_url,
            temperature=0,
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
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")

        policies: list[ProviderPolicy] = []
        if gemini_api_key:
            policies.append(
                ProviderPolicy(
                    name="gemini",
                    provider_type="gemini",
                    model=self._settings.gemini_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=0,
                    request_limit_per_minute=self._settings.gemini_request_limit_per_minute,
                    token_limit_per_minute=self._settings.gemini_token_limit_per_minute,
                    max_parallel_requests=self._settings.gemini_max_parallel_requests,
                )
            )
            self._clients["gemini"] = GeminiProvider(policies[-1], gemini_api_key)

        if groq_api_key:
            policies.append(
                ProviderPolicy(
                    name="groq",
                    provider_type="groq",
                    model=self._settings.groq_model,
                    task_types=("planner", "evaluator", "synthesis", "worker_tool_calling"),
                    priority=1,
                    request_limit_per_minute=self._settings.groq_request_limit_per_minute,
                    token_limit_per_minute=self._settings.groq_token_limit_per_minute,
                    max_parallel_requests=self._settings.groq_max_parallel_requests,
                )
            )
            self._clients["groq"] = GroqProvider(policies[-1], groq_api_key)

        if hf_api_key and self._settings.enable_huggingface_router:
            policies.append(
                ProviderPolicy(
                    name="huggingface",
                    provider_type="huggingface",
                    model=self._settings.huggingface_model,
                    task_types=("planner", "evaluator", "synthesis"),
                    priority=2,
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

    async def generate_text(
        self,
        task_type: str,
        messages: list[BaseMessage],
        budget: RequestBudget,
        trace_id: str,
    ) -> AIMessage:
        return await self._execute(task_type, "text", messages, budget, trace_id, schema=None, tools=None)

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

        while time.monotonic() - started_at <= self._settings.provider_unavailable_wait_seconds:
            candidates, blocked = await self._select_candidates(task_type)
            if not candidates:
                last_error = RuntimeError(self._format_provider_unavailable_message(task_type, blocked))
                remaining = self._settings.provider_unavailable_wait_seconds - (time.monotonic() - started_at)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self._next_provider_check_delay(blocked), remaining))
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
        token_cost = estimate_tokens_from_messages(trim_messages_for_budget(messages, budget)) + max(0, budget.max_output_tokens)
        reservation = await self._coordination_store.reserve_quota(
            key=f"provider:{policy.name}",
            request_cost=1,
            token_cost=token_cost,
            request_limit=policy.request_limit_per_minute,
            token_limit=policy.token_limit_per_minute,
            window_seconds=60,
        )
        if not reservation.allowed:
            await self._coordination_store.set_json(
                f"provider:cooldown:{policy.name}",
                {"reason": "local_quota_guard"},
                ttl_seconds=reservation.retry_after_seconds,
            )
            return RuntimeError(f"{policy.name} local quota guard triggered.")

        async with state.semaphore:
            try:
                if mode == "text":
                    return await client.generate_text(messages, budget)
                if mode == "structured":
                    return await client.generate_structured(schema, messages, budget)
                if mode == "tool_calling":
                    return await client.generate_tool_calling(messages, tools or [], budget)
                raise ValueError(f"Unsupported model router mode: {mode}")
            except Exception as error:
                decision = await self._build_retry_decision(policy.name, error)
                if decision.retry_same_provider:
                    await self._coordination_store.increment_job_metric(trace_id, "retry_count", 1)
                    await asyncio.sleep(decision.delay_seconds)
                    try:
                        if mode == "text":
                            return await client.generate_text(messages, budget)
                        if mode == "structured":
                            return await client.generate_structured(schema, messages, budget)
                        return await client.generate_tool_calling(messages, tools or [], budget)
                    except Exception as second_error:
                        error = second_error
                        decision = await self._build_retry_decision(policy.name, second_error)

                if decision.reason == "rate_limit":
                    ttl = max(1, int(decision.delay_seconds))
                    await self._coordination_store.set_json(
                        f"provider:cooldown:{policy.name}",
                        {"reason": "rate_limit"},
                        ttl_seconds=ttl,
                    )
                elif decision.reason == "quota":
                    await self._coordination_store.set_json(
                        f"provider:circuit:{policy.name}",
                        {"reason": "quota"},
                        ttl_seconds=self._settings.quota_cooldown_seconds,
                    )
                elif decision.reason == "configuration":
                    await self._coordination_store.set_json(
                        f"provider:circuit:{policy.name}",
                        {"reason": "configuration", "error": str(error)},
                        ttl_seconds=self._settings.quota_cooldown_seconds,
                    )

                return error

    async def _build_retry_decision(self, provider_name: str, error: Exception) -> RetryDecision:
        classification = classify_exception(error)
        if classification == "rate_limit":
            retry_after = parse_retry_after_seconds(error) or self._settings.provider_cooldown_seconds
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
