import unittest
import asyncio
from dataclasses import replace
from types import SimpleNamespace

from agents.lead_orchestrator_agent import _find_missing_sections, route_batch_dispatch
from agents.sub_agent import Finding, _finding_to_dict
from services.config import get_settings
from services.coordination import InMemoryCoordinationStore
from services.job_manager import ResearchJobManager
from services.model_router import ProviderBlock, ProviderPolicy, ProviderUnavailableError


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


class QuotaWaitingGraph:
    async def astream(self, initial_state, config):
        if False:
            yield {}
        policy = ProviderPolicy(
            name="gemini",
            provider_type="gemini",
            model="gemini-2.5-flash",
            task_types=("planner",),
            priority=0,
            request_limit_per_minute=10,
            token_limit_per_minute=1000,
            max_parallel_requests=1,
        )
        raise ProviderUnavailableError(
            task_type="planner",
            blocked=[ProviderBlock(policy=policy, reason="quota", ttl_seconds=1)],
            supported=["gemini"],
            message="All providers are temporarily unavailable for task type 'planner': gemini: quota (1s)",
        )

    async def aget_state(self, config):
        return SimpleNamespace(next=[], values={})

    async def aupdate_state(self, config, updates):
        return None


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

    def test_finding_objects_are_serialized_to_plain_dicts_before_graph_state_merge(self):
        finding = _finding_to_dict(Finding(fact="Fact", source_url="https://example.com", confidence=0.8))
        self.assertEqual(
            finding,
            {"fact": "Fact", "source_url": "https://example.com", "confidence": 0.8},
        )

    async def test_sub_agent_chunks_emit_live_source_and_evidence_events(self):
        thread_id = "thread-events"
        await self.store.set_job_record(thread_id, {"status": "running"})

        await self.manager._emit_graph_chunk(
            thread_id,
            {
                "sub_agent": {
                    "completed_tasks": ["Task A"],
                    "discovered_sources": ["https://example.com/a", "https://example.com/b"],
                    "sources": ["https://example.com/a"],
                    "evidence_cards": [
                        {
                            "claim": "Claim",
                            "source_url": "https://example.com/a",
                            "source_title": "Example",
                            "excerpt": "Excerpt",
                            "section_tag": "diet_pattern",
                            "source_type": "guideline",
                            "authority_score": 9,
                            "confidence": 0.9,
                        }
                    ],
                    "coverage_tags": ["diet_pattern"],
                    "findings": [{"fact": "Fact", "source_url": "https://example.com/a", "confidence": 0.9}],
                }
            },
        )

        replay, queue = await self.manager.subscribe(thread_id)
        try:
            self.assertEqual([event.event for event in replay], ["source_batch", "evidence_batch", "agent"])
            self.assertEqual(replay[0].data["discovered_count"], 2)
            self.assertEqual(replay[1].data["coverage_tags"], ["diet_pattern"])
            self.assertEqual(replay[2].data["task"], "Task A")
        finally:
            await self.manager.unsubscribe(thread_id, queue)

    def test_missing_sections_require_multiple_supporting_sources(self):
        missing = _find_missing_sections(
            ["Sodium", "Exercise"],
            [
                {
                    "claim": "Lower sodium intake helps.",
                    "source_url": "https://source-one.example/sodium",
                    "section_tag": "sodium",
                },
                {
                    "claim": "Exercise helps recovery.",
                    "source_url": "https://source-one.example/exercise",
                    "section_tag": "exercise",
                },
                {
                    "claim": "Structured activity supports rehab.",
                    "source_url": "https://source-two.example/exercise",
                    "section_tag": "exercise",
                },
            ],
        )

        self.assertEqual(missing, ["Sodium"])

    async def test_quota_unavailable_jobs_wait_and_requeue_instead_of_failing(self):
        local_store = InMemoryCoordinationStore()
        await local_store.start()
        quota_manager = ResearchJobManager(
            QuotaWaitingGraph(),
            local_store,
            replace(
                self.settings,
                max_active_jobs=0,
            ),
        )

        thread_id = "thread-quota"
        await local_store.set_job_record(
            thread_id,
            {
                "thread_id": thread_id,
                "status": "queued",
                "queued_at": 0,
                "started_at": None,
                "finished_at": None,
                "last_error": "",
                "retry_count": 0,
                "provider_switch_count": 0,
                "client_id": "client-1",
                "current_plan": [],
                "required_sections": [],
                "quota_wait_until": None,
                "quota_retry_after_seconds": 0,
                "waiting_task_type": "",
            },
        )

        await quota_manager._run_job(
            0,
            {
                "kind": "start",
                "thread_id": thread_id,
                "initial_state": {
                    "thread_id": thread_id,
                    "original_query": "query",
                    "human_feedback": "",
                    "research_plan": [],
                    "required_sections": [],
                    "pending_tasks": [],
                    "current_batch": [],
                    "findings": [],
                    "evidence_cards": [],
                    "sources": [],
                    "discovered_sources": [],
                    "coverage_tags": [],
                    "completed_tasks": [],
                    "gaps": [],
                    "quality_summary": "",
                    "evaluation_rounds": 0,
                    "outline_sections": [],
                    "section_drafts": {},
                    "final_report": "",
                },
            },
        )

        waiting_status = await quota_manager.get_status(thread_id)
        self.assertEqual(waiting_status["status"], "waiting_for_quota")
        self.assertEqual(waiting_status["waiting_task_type"], "planner")

        replay, queue = await quota_manager.subscribe(thread_id)
        try:
            self.assertIn("waiting_for_quota", [event.event for event in replay])
        finally:
            await quota_manager.unsubscribe(thread_id, queue)

        await asyncio.sleep(1.1)
        queued_status = await quota_manager.get_status(thread_id)
        self.assertEqual(queued_status["status"], "queued")
        self.assertIsNotNone(await local_store.queue_position(thread_id))
        await quota_manager.close()
        await local_store.close()
