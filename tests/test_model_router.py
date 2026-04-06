import unittest
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from services.config import get_settings
from services.coordination import InMemoryCoordinationStore
from services.model_router import ModelRouter, RequestBudget


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
