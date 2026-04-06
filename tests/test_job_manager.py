import unittest
from dataclasses import replace
from types import SimpleNamespace

from agents.lead_orchestrator_agent import route_batch_dispatch
from services.config import get_settings
from services.coordination import InMemoryCoordinationStore
from services.job_manager import ResearchJobManager


class DummyGraph:
    def __init__(self):
        self._states = {}

    async def astream(self, initial_state, config):
        thread_id = config["configurable"]["thread_id"]
        if initial_state is not None:
            plan = ["Task A", "Task B"]
            self._states[thread_id] = {
                "next": ["plan_review_node"],
                "values": {**initial_state, "research_plan": plan, "findings": []},
            }
            yield {"decompose_node": {"research_plan": plan}}
            return

        current = self._states[thread_id]["values"]
        current["final_report"] = "Final report"
        current["findings"] = [{"fact": "Fact", "source_url": "https://example.com", "confidence": 0.9}]
        self._states[thread_id] = {"next": [], "values": current}
        yield {"synthesize_node": {}}

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        state = self._states.get(
            thread_id,
            {"next": [], "values": {"research_plan": [], "findings": [], "final_report": ""}},
        )
        return SimpleNamespace(next=state["next"], values=state["values"])

    async def aupdate_state(self, config, updates):
        thread_id = config["configurable"]["thread_id"]
        self._states[thread_id]["values"].update(updates)


class JobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = InMemoryCoordinationStore()
        await self.store.start()
        self.settings = replace(
            get_settings(),
            max_active_jobs=1,
            queue_wait_timeout_seconds=0.05,
            heartbeat_interval_seconds=0.05,
            client_rate_limit_per_minute=100,
        )
        self.manager = ResearchJobManager(DummyGraph(), self.store, self.settings)
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.close()

    async def test_start_pause_resume_and_complete(self):
        thread_id = await self.manager.submit_start("research something", client_id="client-1")
        first_run_events = [event async for event in self.manager.event_stream(thread_id)]
        first_run_names = [event.event for event in first_run_events]

        self.assertIn("queued", first_run_names)
        self.assertIn("started", first_run_names)
        self.assertIn("plan", first_run_names)
        self.assertIn("paused", first_run_names)

        paused_status = await self.manager.get_status(thread_id)
        self.assertEqual(paused_status["status"], "paused")

        await self.manager.submit_resume(thread_id, "")
        second_run_events = [event async for event in self.manager.event_stream(thread_id)]
        second_run_names = [event.event for event in second_run_events]

        self.assertIn("queued", second_run_names)
        self.assertIn("started", second_run_names)
        self.assertIn("synthesize", second_run_names)
        self.assertIn("report", second_run_names)
        self.assertIn("done", second_run_names)

        done_status = await self.manager.get_status(thread_id)
        self.assertEqual(done_status["status"], "done")
        self.assertEqual(done_status["extracted_facts_count"], 1)

    async def test_sub_agent_dispatch_uses_trace_id_not_parent_thread_id_field(self):
        sends = route_batch_dispatch(
            {
                "thread_id": "thread-123",
                "original_query": "query",
                "research_plan": ["Task A"],
                "pending_tasks": [],
                "current_batch": ["Task A"],
                "human_feedback": "",
                "findings": [],
                "sources": [],
                "gaps": [],
                "evaluation_rounds": 0,
                "final_report": "",
            }
        )

        self.assertEqual(len(sends), 1)
        payload = sends[0].arg
        self.assertEqual(payload["trace_id"], "thread-123")
        self.assertNotIn("thread_id", payload)
