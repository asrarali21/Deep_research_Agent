import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.runtime import get_job_manager

router = APIRouter()


class StartRequest(BaseModel):
    query: str


class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str = ""


def format_sse(data, event: str = "message") -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_job_events(thread_id: str):
    job_manager = get_job_manager()
    async for job_event in job_manager.event_stream(thread_id):
        yield format_sse(job_event.data, event=job_event.event)


@router.post("/research/start")
async def start_research(request: Request, req: StartRequest):
    client_id = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "unknown")
    job_manager = request.app.state.job_manager
    thread_id = await job_manager.submit_start(req.query, client_id=client_id)
    headers = {"X-Thread-ID": thread_id}

    return StreamingResponse(
        stream_job_events(thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/research/resume")
async def resume_research(request: Request, req: ResumeRequest):
    job_manager = request.app.state.job_manager
    await job_manager.submit_resume(req.thread_id, req.feedback)
    return StreamingResponse(
        stream_job_events(req.thread_id),
        media_type="text/event-stream",
    )


@router.get("/research/status/{thread_id}")
async def get_status(request: Request, thread_id: str):
    job_manager = request.app.state.job_manager
    return await job_manager.get_status(thread_id)
