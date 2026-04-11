import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging — production-grade setup for all service modules
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATE_FORMAT,
    stream=sys.stdout,
    force=True,
)
# Silence noisy third-party loggers
for _noisy in ("httpcore", "httpx", "urllib3", "asyncio", "langchain", "langchain_core"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg_pool import AsyncConnectionPool

from api.user_query import router as user_query_router
from agents.lead_orchestrator_agent import create_lead_orchestrator
from services.job_manager import ResearchJobManager
from services.runtime import initialize_runtime, set_job_manager

# Global variable to hold the DB pool so it can be cleanly closed
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events replace the old @app.on_event("startup")
    It safely establishes the DB Connection pool before accepting any requests!
    """
    global db_pool
    db_url = os.getenv("POSTGRES_DB_URL")
    if not db_url:
        raise ValueError("POSTGRES_DB_URL environment variable is missing!")

    print(f"🔌 Connecting to PostgreSQL at {db_url}...")

    settings, coordination_store, _ = await initialize_runtime()

    db_pool = AsyncConnectionPool(
        conninfo=db_url,
        max_size=20,
        kwargs={"autocommit": True},
        open=False,
    )
    await db_pool.open()

    checkpointer = AsyncPostgresSaver(
        db_pool,
        serde=JsonPlusSerializer(allowed_msgpack_modules=[("agents.sub_agent", "Finding")]),
    )
    print("🛠️  Initializing LangGraph Checkpoint SQL Tables (if they don't exist)...")
    await checkpointer.setup()

    print("🧠 Compiling Master Graph...")
    app.state.graph = create_lead_orchestrator(checkpointer=checkpointer)

    app.state.settings = settings
    app.state.coordination_store = coordination_store
    app.state.job_manager = ResearchJobManager(
        graph=app.state.graph,
        coordination_store=coordination_store,
        settings=settings,
    )
    set_job_manager(app.state.job_manager)
    await app.state.job_manager.start()

    print("✅ Server Online and fully connected to Database!")
    yield

    print("🛑 Shutting down server, cleanly closing Database Pool...")
    await app.state.job_manager.close()
    await coordination_store.close()
    await db_pool.close()

app = FastAPI(
    title="Deep Research Agent API",
    description="API Gateway for the Multi-Agent LangGraph Research System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS so frontend apps (React/Next.js) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register our API routes
app.include_router(user_query_router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "Deep Research Orchestrator Online"}

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
