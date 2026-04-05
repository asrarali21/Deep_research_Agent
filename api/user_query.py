import uuid
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# --- Pydantic Data Models ---
class StartRequest(BaseModel):
    query: str

class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str = ""  # Empty string means "I approve, proceed"

# --- SSE Helper ---
def format_sse(data: str, event: str = "message") -> str:
    """
    Format strings into Server-Sent Events standard shape.
    Wrapping data in a JSON payload makes frontend parsing incredibly robust.
    """
    payload = json.dumps({"text": data})
    return f"event: {event}\ndata: {payload}\n\n"

# --- Streaming Generator ---
def stream_graph_execution(graph, config, initial_state=None):
    """
    Translates LangGraph chunk updates into SSE text strings.
    If initial_state is provided, it starts a new execution.
    If initial_state is None, it resumes a paused execution.
    """
    try:
        # LangGraph Stream execution kicks off
        stream_iterator = graph.stream(initial_state, config)
        
        for chunk in stream_iterator:
            for node_name, updates in chunk.items():
                if node_name == "decompose_node":
                    plan = updates.get("research_plan", [])
                    yield format_sse(f"🧠 [Decompose] Created research plan mapping to {len(plan)} sub-tasks:")
                    for i, t in enumerate(plan, 1):
                        yield format_sse(f"   {i}. {t}")
                        
                elif node_name == "sub_agent":
                    yield format_sse(f"🤖 [Agent] A parallel sub-agent finished its research branch.")
                    
                elif node_name == "evaluate_node":
                    gaps = updates.get("gaps", [])
                    if gaps:
                         yield format_sse(f"🔍 [Evaluate] Evaluator found {len(gaps)} missing research gaps! Respawning agents...")
                    else:
                         yield format_sse(f"✅ [Evaluate] Knowledge base is complete. Proceeding to synthesis...")
                         
                elif node_name == "synthesize_node":
                    yield format_sse(f"✍️ [Synthesizer] Final report generated!")
                    
        # Once the stream loop finishes, we check if the Engine is Paused or Done
        state = graph.get_state(config)
        
        if state.next:
            yield format_sse(f"⏸️ [SYSTEM] Graph paused. Waiting for human approval on the research plan.", event="paused")
        else:
            # If there's a final report, yield it as a special event
            final_report = state.values.get("final_report")
            if final_report:
                yield format_sse(final_report, event="report")
            yield format_sse(f"🏁 [SYSTEM] Deep Research completely finished.", event="done")

    except Exception as e:
         yield format_sse(f"❌ [CRITICAL ERROR] {str(e)}", event="error")


# --- API Routes ---

@router.post("/research/start")
async def start_research(request: Request, req: StartRequest):
    """
    Starts a completely new isolated deep research process.
    """
    # 1. Generate a completely unique memory sandbox id
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 2. Build the initial empty state
    initial_state = {
        "original_query": req.query,
        "human_feedback": "",
        "research_plan": [],
        "findings": [],
        "sources": [],
        "gaps": [],
        "evaluation_rounds": 0,
        "final_report": ""
    }
    
    # 3. We pass the Thread ID back to the client via headers so they can capture it instantly 
    # to use on the /resume endpoint, while they watch the stream body!
    headers = {"X-Thread-ID": thread_id}
    
    graph = request.app.state.graph
    
    return StreamingResponse(
        stream_graph_execution(graph, config, initial_state), 
        media_type="text/event-stream",
        headers=headers
    )

@router.post("/research/resume")
async def resume_research(request: Request, req: ResumeRequest):
    """
    Resumes a paused research process, capturing human feedback.
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # 1. Verify the thread actually exists
    graph = request.app.state.graph
    state = graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=400, detail="This thread does not exist.")
        
    # 2. Inject the human feedback into the state checkpointer
    # If req.feedback is "", it means they approved. If it has text, it means edit.
    graph.update_state(config, {"human_feedback": req.feedback})
    
    # 3. Resume the stream! (initial_state=None tells LangGraph to continue from checkpoint)
    return StreamingResponse(
        stream_graph_execution(graph, config, None), 
        media_type="text/event-stream"
    )

@router.get("/research/status/{thread_id}")
async def get_status(request: Request, thread_id: str):
    """
    Simple polling endpoint to check where a thread is sitting.
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph
    state = graph.get_state(config)
    
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    return {
        "status": "paused" if state.next else "running_or_done",
        "current_plan": state.values.get("research_plan", []),
        "extracted_facts_count": len(state.values.get("findings", [])),
    }