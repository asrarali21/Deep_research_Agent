import unittest
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from agents.sub_agent import SearchTool
from services.config import get_settings
from services.coordination import InMemoryCoordinationStore
from services.model_router import ModelRouter, ProviderBlock, ProviderPolicy, ProviderUnavailableError, RequestBudget, coerce_tool_calls_from_text


class FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    async def generate_text(self, messages, budget):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate_structured(self, schema, messages, budget):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate_tool_calling(self, messages, tools, budget):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ModelRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = InMemoryCoordinationStore()
        await self.store.start()
        self.settings = replace(
            get_settings(),
            provider_unavailable_wait_seconds=1.0,
            transient_retry_base_seconds=0.0,
            transient_retry_jitter_seconds=0.0,
        )

    async def test_rate_limit_switches_provider_and_records_cooldown(self):
        await self.store.set_job_record("trace-1", {"retry_count": 0, "provider_switch_count": 0})
        router = ModelRouter(
            self.store,
            self.settings,
            provider_clients={
                "primary": FakeProvider([RuntimeError("429 rate limit exceeded retry-after 4")]),
                "secondary": FakeProvider([AIMessage(content="fallback worked")]),
            },
        )

        response = await router.generate_text(
            task_type="planner",
            messages=[HumanMessage(content="hello")],
            budget=RequestBudget(max_input_chars=500),
            trace_id="trace-1",
        )

        record = await self.store.get_job_record("trace-1")
        self.assertEqual(response.content, "fallback worked")
        self.assertEqual(record["provider_switch_count"], 1)
        self.assertGreaterEqual(await self.store.get_ttl_seconds("provider:cooldown:primary"), 1)

    async def test_transient_error_retries_same_provider_once(self):
        await self.store.set_job_record("trace-2", {"retry_count": 0, "provider_switch_count": 0})
        router = ModelRouter(
            self.store,
            self.settings,
            provider_clients={
                "solo": FakeProvider([RuntimeError("timeout contacting provider"), AIMessage(content="recovered")]),
            },
        )

        response = await router.generate_text(
            task_type="synthesis",
            messages=[HumanMessage(content="hello")],
            budget=RequestBudget(max_input_chars=500),
            trace_id="trace-2",
        )

        record = await self.store.get_job_record("trace-2")
        self.assertEqual(response.content, "recovered")
        self.assertEqual(record["retry_count"], 1)

    async def test_configuration_error_circuits_bad_provider_and_falls_back(self):
        await self.store.set_job_record("trace-3", {"retry_count": 0, "provider_switch_count": 0})
        router = ModelRouter(
            self.store,
            self.settings,
            provider_clients={
                "bad-model": FakeProvider(
                    [
                        RuntimeError(
                            "Error calling model 'gemini-1.5-flash-latest' (NOT_FOUND): 404 NOT_FOUND. "
                            "models/gemini-1.5-flash-latest is not found for API version v1beta, "
                            "or is not supported for generateContent."
                        )
                    ]
                ),
                "fallback": FakeProvider([AIMessage(content="groq fallback worked")]),
            },
        )

        response = await router.generate_text(
            task_type="planner",
            messages=[HumanMessage(content="hello")],
            budget=RequestBudget(max_input_chars=500),
            trace_id="trace-3",
        )

        record = await self.store.get_job_record("trace-3")
        self.assertEqual(response.content, "groq fallback worked")
        self.assertEqual(record["provider_switch_count"], 1)
        self.assertGreaterEqual(await self.store.get_ttl_seconds("provider:circuit:bad-model"), 1)

    async def test_waits_for_temporary_cooldown_instead_of_failing_with_generic_no_provider(self):
        await self.store.set_job_record("trace-4", {"retry_count": 0, "provider_switch_count": 0})
        settings = replace(
            self.settings,
            provider_unavailable_wait_seconds=2.0,
            provider_poll_interval_seconds=0.1,
        )
        router = ModelRouter(
            self.store,
            settings,
            provider_clients={
                "primary": FakeProvider([AIMessage(content="recovered after cooldown")]),
            },
        )
        await self.store.set_json(
            "provider:cooldown:primary",
            {"reason": "rate_limit"},
            ttl_seconds=1,
        )

        response = await router.generate_text(
            task_type="planner",
            messages=[HumanMessage(content="hello")],
            budget=RequestBudget(max_input_chars=500),
            trace_id="trace-4",
        )

        self.assertEqual(response.content, "recovered after cooldown")

    def test_recovers_xml_style_failed_generation_into_tool_call(self):
        message = coerce_tool_calls_from_text(
            '<function=SearchTool>{"query":"whole unprocessed foods portion sizes and meal frequencies"}</function>',
            [SearchTool],
        )

        self.assertIsNotNone(message)
        self.assertEqual(message.tool_calls[0]["name"], "SearchTool")
        self.assertEqual(
            message.tool_calls[0]["args"]["query"],
            "whole unprocessed foods portion sizes and meal frequencies",
        )

    def test_recovers_json_style_tool_selection_into_tool_call(self):
        message = coerce_tool_calls_from_text(
            '{"tool_name":"SearchTool","arguments":{"query":"cardiac rehab meal plan"}}',
            [SearchTool],
        )

        self.assertIsNotNone(message)
        self.assertEqual(message.tool_calls[0]["name"], "SearchTool")
        self.assertEqual(message.tool_calls[0]["args"]["query"], "cardiac rehab meal plan")

    def test_provider_unavailable_error_waits_for_retryable_provider_even_with_configuration_block(self):
        quota_policy = ProviderPolicy(
            name="groq",
            provider_type="groq",
            model="llama-3.3-70b-versatile",
            task_types=("worker_tool_calling",),
            priority=1,
            request_limit_per_minute=10,
            token_limit_per_minute=1000,
            max_parallel_requests=1,
        )
        config_policy = ProviderPolicy(
            name="gemini",
            provider_type="gemini",
            model="gemini-2.5-flash",
            task_types=("worker_tool_calling",),
            priority=0,
            request_limit_per_minute=10,
            token_limit_per_minute=1000,
            max_parallel_requests=1,
        )
        error = ProviderUnavailableError(
            task_type="worker_tool_calling",
            blocked=[
                ProviderBlock(policy=config_policy, reason="configuration", ttl_seconds=784),
                ProviderBlock(policy=quota_policy, reason="quota", ttl_seconds=813),
            ],
            supported=["gemini", "groq"],
            message="All providers are temporarily unavailable for task type 'worker_tool_calling'",
        )

        self.assertTrue(error.should_wait_for_retryable_provider)
        self.assertEqual(error.retry_after_seconds, 813)
