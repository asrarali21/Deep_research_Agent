import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import HTTPException

from services.config import Settings
from services.coordination import CoordinationStore


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
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

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
            "human_feedback": "",
            "research_plan": [],
            "pending_tasks": [],
            "current_batch": [],
            "findings": [],
            "sources": [],
            "gaps": [],
            "evaluation_rounds": 0,
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

        try:
            state = await self._graph.aget_state(config)
            if state and state.values:
                current_plan = state.values.get("research_plan", current_plan)
                extracted_facts_count = len(state.values.get("findings", []))
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
            )
            if final_report:
                await self.publish_event(thread_id, "report", {"thread_id": thread_id, "report": final_report})
            await self.publish_event(thread_id, "done", {"thread_id": thread_id, "status": "done"})
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
                await self._coordination_store.update_job_record(thread_id, current_plan=plan)
                await self.publish_event(
                    thread_id,
                    "plan",
                    {"thread_id": thread_id, "task_count": len(plan), "plan": plan},
                )
            elif node_name == "sub_agent":
                await self.publish_event(thread_id, "agent", {"thread_id": thread_id, "status": "sub_agent_complete"})
            elif node_name == "evaluate_node":
                gaps = updates.get("gaps", [])
                await self.publish_event(
                    thread_id,
                    "evaluate",
                    {"thread_id": thread_id, "gaps": gaps, "gap_count": len(gaps)},
                )
            elif node_name == "synthesize_node":
                await self.publish_event(
                    thread_id,
                    "synthesize",
                    {"thread_id": thread_id, "status": "finalizing_report"},
                )
