import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import HTTPException

from services.config import Settings
from services.coordination import CoordinationStore
from services.model_router import ProviderUnavailableError


@dataclass(slots=True)
class JobEvent:
    event: str
    data: dict[str, Any]


class ResearchJobManager:
    def __init__(self, graph, coordination_store: CoordinationStore, settings: Settings) -> None:
        self._graph = graph
        self._coordination_store = coordination_store
        self._settings = settings
        self._worker_tasks: list[asyncio.Task] = []
        self._scheduled_requeues: dict[str, asyncio.Task] = {}
        self._subscriber_queues: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._recent_events: dict[str, deque[JobEvent]] = defaultdict(lambda: deque(maxlen=100))
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        if self._worker_tasks:
            return
        for index in range(self._settings.max_active_jobs):
            task = asyncio.create_task(self._worker_loop(index), name=f"research-worker-{index}")
            self._worker_tasks.append(task)

    async def close(self) -> None:
        self._shutdown.set()
        for task in self._worker_tasks:
            task.cancel()
        for task in self._scheduled_requeues.values():
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        await asyncio.gather(*self._scheduled_requeues.values(), return_exceptions=True)
        self._worker_tasks.clear()
        self._scheduled_requeues.clear()

    async def submit_start(self, query: str, client_id: str) -> str:
        decision = await self._coordination_store.check_rate_limit(
            key=f"client:{client_id}",
            limit=self._settings.client_rate_limit_per_minute,
            window_seconds=60,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this client.",
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )

        queue_depth = await self._coordination_store.queue_depth()
        if queue_depth >= self._settings.max_queue_depth:
            raise HTTPException(
                status_code=503,
                detail="Research queue is full. Please retry shortly.",
                headers={"Retry-After": str(self._settings.queue_full_retry_after_seconds)},
            )

        thread_id = str(uuid.uuid4())
        initial_state = {
            "thread_id": thread_id,
            "original_query": query,
            "depth_profile": "deep",
            "research_contract": {},
            "depth_budget": {},
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
            "targeted_gap_rounds": 0,
            "outline_sections": [],
            "section_packets": [],
            "priority_sections": [],
            "section_drafts": {},
            "section_verifications": {},
            "report_reference_urls": [],
            "structured_references": [],
            "final_report": "",
        }
        record = {
            "thread_id": thread_id,
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "last_error": "",
            "retry_count": 0,
            "provider_switch_count": 0,
            "client_id": client_id,
            "current_plan": [],
            "required_sections": [],
            "quota_wait_until": None,
            "quota_retry_after_seconds": 0,
            "waiting_task_type": "",
        }
        await self._coordination_store.set_job_record(thread_id, record)
        await self._coordination_store.enqueue_job(
            thread_id,
            {
                "kind": "start",
                "thread_id": thread_id,
                "initial_state": initial_state,
            },
        )
        await self.publish_event(thread_id, "queued", {"thread_id": thread_id, "status": "queued"})
        return thread_id

    async def submit_resume(self, thread_id: str, feedback: str) -> None:
        record = await self._coordination_store.get_job_record(thread_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if record.get("status") != "paused":
            raise HTTPException(status_code=409, detail="Thread is not paused.")

        config = {"configurable": {"thread_id": thread_id}}
        await self._graph.aupdate_state(config, {"human_feedback": feedback})
        await self._coordination_store.update_job_record(
            thread_id,
            status="queued",
            last_error="",
            quota_wait_until=None,
            quota_retry_after_seconds=0,
            waiting_task_type="",
        )
        await self._coordination_store.enqueue_job(
            thread_id,
            {
                "kind": "resume",
                "thread_id": thread_id,
                "initial_state": None,
            },
        )
        await self.publish_event(thread_id, "queued", {"thread_id": thread_id, "status": "queued", "resume": True})

    async def get_status(self, thread_id: str) -> dict[str, Any]:
        record = await self._coordination_store.get_job_record(thread_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        queue_position = await self._coordination_store.queue_position(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        current_plan: list[str] = record.get("current_plan", [])
        extracted_facts_count = 0
        evidence_card_count = 0
        required_sections: list[str] = record.get("required_sections", [])

        try:
            state = await self._graph.aget_state(config)
            if state and state.values:
                current_plan = state.values.get("research_plan", current_plan)
                extracted_facts_count = len(state.values.get("findings", []))
                evidence_card_count = len(state.values.get("evidence_cards", []))
                required_sections = state.values.get("required_sections", required_sections)
        except Exception:
            pass

        return {
            "thread_id": thread_id,
            "status": record.get("status", "queued"),
            "queue_position": queue_position,
            "retry_count": int(record.get("retry_count", 0)),
            "provider_switch_count": int(record.get("provider_switch_count", 0)),
            "current_plan": current_plan,
            "extracted_facts_count": extracted_facts_count,
            "evidence_card_count": evidence_card_count,
            "required_sections": required_sections,
            "quota_wait_until": record.get("quota_wait_until"),
            "quota_retry_after_seconds": int(record.get("quota_retry_after_seconds", 0)),
            "waiting_task_type": record.get("waiting_task_type", ""),
            "last_error": record.get("last_error", ""),
        }

    async def subscribe(self, thread_id: str) -> tuple[list[JobEvent], asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscriber_queues[thread_id].add(queue)
        replay = list(self._recent_events.get(thread_id, []))
        return replay, queue

    async def unsubscribe(self, thread_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscriber_queues.get(thread_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscriber_queues.pop(thread_id, None)

    async def event_stream(self, thread_id: str) -> AsyncIterator[JobEvent]:
        replay, queue = await self.subscribe(thread_id)
        try:
            terminal_events = {"paused", "done", "failed"}
            for item in replay:
                yield item
            if replay and replay[-1].event in terminal_events:
                return
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._settings.heartbeat_interval_seconds)
                    yield item
                    if item.event in terminal_events:
                        break
                except asyncio.TimeoutError:
                    status = await self.get_status(thread_id)
                    yield JobEvent(event="heartbeat", data=status)
                    if status["status"] in {"paused", "done", "failed"}:
                        break
        finally:
            await self.unsubscribe(thread_id, queue)

    async def publish_event(self, thread_id: str, event: str, data: dict[str, Any]) -> None:
        job_event = JobEvent(event=event, data=data)
        self._recent_events[thread_id].append(job_event)
        for subscriber in list(self._subscriber_queues.get(thread_id, set())):
            subscriber.put_nowait(job_event)

    async def _worker_loop(self, worker_index: int) -> None:
        while not self._shutdown.is_set():
            item = await self._coordination_store.dequeue_job(self._settings.queue_wait_timeout_seconds)
            if item is None:
                continue
            _, payload = item
            await self._run_job(worker_index, payload)

    async def _run_job(self, worker_index: int, payload: dict[str, Any]) -> None:
        thread_id = payload["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        await self._coordination_store.update_job_record(
            thread_id,
            status="running",
            started_at=time.time(),
            quota_wait_until=None,
            quota_retry_after_seconds=0,
            waiting_task_type="",
        )
        await self.publish_event(
            thread_id,
            "started",
            {"thread_id": thread_id, "status": "running", "worker": worker_index},
        )

        try:
            async for chunk in self._graph.astream(payload.get("initial_state"), config):
                await self._emit_graph_chunk(thread_id, chunk)

            state = await self._graph.aget_state(config)
            if state.next:
                await self._coordination_store.update_job_record(thread_id, status="paused")
                await self.publish_event(
                    thread_id,
                    "paused",
                    {"thread_id": thread_id, "status": "paused"},
                )
                return

            final_report = state.values.get("final_report", "") if state and state.values else ""
            await self._coordination_store.update_job_record(
                thread_id,
                status="done",
                finished_at=time.time(),
                quota_wait_until=None,
                quota_retry_after_seconds=0,
                waiting_task_type="",
            )
            if final_report:
                structured_references = state.values.get("structured_references", []) if state and state.values else []
                await self.publish_event(
                    thread_id,
                    "report",
                    {
                        "thread_id": thread_id,
                        "report": final_report,
                        "structured_references": structured_references,
                    },
                )
            await self.publish_event(thread_id, "done", {"thread_id": thread_id, "status": "done"})
        except ProviderUnavailableError as error:
            if error.should_wait_for_retryable_provider:
                await self._transition_to_quota_wait(thread_id, payload, error)
                return
            await self._coordination_store.update_job_record(
                thread_id,
                status="failed",
                finished_at=time.time(),
                last_error=str(error),
                quota_wait_until=None,
                quota_retry_after_seconds=0,
                waiting_task_type="",
            )
            await self.publish_event(
                thread_id,
                "failed",
                {"thread_id": thread_id, "status": "failed", "error": str(error)},
            )
        except Exception as error:
            await self._coordination_store.update_job_record(
                thread_id,
                status="failed",
                finished_at=time.time(),
                last_error=str(error),
            )
            await self.publish_event(
                thread_id,
                "failed",
                {"thread_id": thread_id, "status": "failed", "error": str(error)},
            )

    async def _emit_graph_chunk(self, thread_id: str, chunk: dict[str, Any]) -> None:
        for node_name, updates in chunk.items():
            if node_name == "decompose_node":
                plan = updates.get("research_plan", [])
                required_sections = updates.get("required_sections", [])
                await self._coordination_store.update_job_record(
                    thread_id,
                    current_plan=plan,
                    required_sections=required_sections,
                )
                await self.publish_event(
                    thread_id,
                    "plan",
                    {
                        "thread_id": thread_id,
                        "task_count": len(plan),
                        "plan": plan,
                        "required_sections": required_sections,
                    },
                )
            elif node_name == "sub_agent":
                completed_tasks = updates.get("completed_tasks", [])
                task = completed_tasks[-1] if completed_tasks else ""
                discovered_sources = updates.get("discovered_sources", [])
                scraped_sources = updates.get("sources", [])
                evidence_cards = updates.get("evidence_cards", [])
                coverage_tags = updates.get("coverage_tags", [])
                findings = updates.get("findings", [])

                if discovered_sources or scraped_sources:
                    await self.publish_event(
                        thread_id,
                        "source_batch",
                        {
                            "thread_id": thread_id,
                            "task": task,
                            "discovered_sources": discovered_sources,
                            "scraped_sources": scraped_sources,
                            "discovered_count": len(discovered_sources),
                            "scraped_count": len(scraped_sources),
                        },
                    )
                if evidence_cards or findings:
                    await self.publish_event(
                        thread_id,
                        "evidence_batch",
                        {
                            "thread_id": thread_id,
                            "task": task,
                            "evidence_count": len(evidence_cards),
                            "finding_count": len(findings),
                            "coverage_tags": coverage_tags,
                        },
                    )
                await self.publish_event(
                    thread_id,
                    "agent",
                    {
                        "thread_id": thread_id,
                        "status": "sub_agent_complete",
                        "task": task,
                        "evidence_count": len(evidence_cards),
                        "source_count": len(scraped_sources),
                    },
                )
            elif node_name == "evaluate_node":
                gaps = updates.get("gaps", [])
                await self.publish_event(
                    thread_id,
                    "evaluate",
                    {
                        "thread_id": thread_id,
                        "gaps": gaps,
                        "gap_count": len(gaps),
                        "quality_summary": updates.get("quality_summary", ""),
                    },
                )
            elif node_name == "build_outline_node":
                await self.publish_event(
                    thread_id,
                    "outline",
                    {
                        "thread_id": thread_id,
                        "sections": updates.get("outline_sections", []),
                    },
                )
            elif node_name == "build_evidence_briefs_node":
                await self.publish_event(
                    thread_id,
                    "evidence_brief",
                    {
                        "thread_id": thread_id,
                        "section_count": len(updates.get("section_packets", [])),
                        "priority_sections": updates.get("priority_sections", []),
                    },
                )
            elif node_name == "draft_sections_node":
                for section, content in updates.get("section_drafts", {}).items():
                    await self.publish_event(
                        thread_id,
                        "section_draft",
                        {
                            "thread_id": thread_id,
                            "section": section,
                            "status": "section_drafted",
                            "char_count": len(content),
                        },
                    )
            elif node_name == "verify_sections_node":
                await self.publish_event(
                    thread_id,
                    "section_verification",
                    {
                        "thread_id": thread_id,
                        "verified_sections": list(updates.get("section_verifications", {}).keys()),
                        "verification_count": len(updates.get("section_verifications", {})),
                    },
                )
            elif node_name in {"final_edit_node", "synthesize_node"}:
                await self.publish_event(
                    thread_id,
                    "synthesize",
                    {"thread_id": thread_id, "status": "final_editing_report"},
                )

    async def _transition_to_quota_wait(
        self,
        thread_id: str,
        payload: dict[str, Any],
        error: ProviderUnavailableError,
    ) -> None:
        retry_after_seconds = error.retry_after_seconds
        wait_until = time.time() + retry_after_seconds
        await self._coordination_store.update_job_record(
            thread_id,
            status="waiting_for_quota",
            last_error=str(error),
            quota_wait_until=wait_until,
            quota_retry_after_seconds=retry_after_seconds,
            waiting_task_type=error.task_type,
        )
        await self.publish_event(
            thread_id,
            "waiting_for_quota",
            {
                "thread_id": thread_id,
                "status": "waiting_for_quota",
                "task_type": error.task_type,
                "retry_after_seconds": retry_after_seconds,
                "available_at": wait_until,
                "error": str(error),
            },
        )
        self._schedule_requeue(thread_id, payload, retry_after_seconds)

    def _schedule_requeue(self, thread_id: str, payload: dict[str, Any], delay_seconds: int) -> None:
        existing = self._scheduled_requeues.pop(thread_id, None)
        if existing is not None:
            existing.cancel()

        async def delayed_requeue() -> None:
            try:
                await asyncio.sleep(max(1, delay_seconds))
                record = await self._coordination_store.get_job_record(thread_id)
                if record is None:
                    return
                if record.get("status") != "waiting_for_quota":
                    return
                await self._coordination_store.update_job_record(
                    thread_id,
                    status="queued",
                    last_error="",
                    quota_wait_until=None,
                    quota_retry_after_seconds=0,
                    waiting_task_type="",
                )
                await self._coordination_store.enqueue_job(
                    thread_id,
                    {
                        "kind": "resume_after_quota",
                        "thread_id": thread_id,
                        "initial_state": None,
                    },
                )
                await self.publish_event(
                    thread_id,
                    "queued",
                    {
                        "thread_id": thread_id,
                        "status": "queued",
                        "resume": True,
                        "reason": "quota_wait_complete",
                    },
                )
            except asyncio.CancelledError:
                raise
            finally:
                self._scheduled_requeues.pop(thread_id, None)

        task = asyncio.create_task(delayed_requeue(), name=f"quota-requeue-{thread_id}")
        self._scheduled_requeues[thread_id] = task
