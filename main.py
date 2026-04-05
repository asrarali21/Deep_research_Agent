import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from api.user_query import router as user_query_router
from agents.lead_orchestrator_agent import create_lead_orchestrator

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
    
    # kwargs={"autocommit": True} is strictly required for the LangGraph saver .setup()
    db_pool = AsyncConnectionPool(
        conninfo=db_url,
        max_size=20,
        kwargs={"autocommit": True},
        open=False
    )
    await db_pool.open()
    
    # 1. Boot up the Async Postgres Checkpointer using the active pool
    checkpointer = AsyncPostgresSaver(db_pool)
    
    # 2. Automatically create/migrate the SQL tables (checkpoints, checkpoint_blobs, etc)
    print("🛠️  Initializing LangGraph Checkpoint SQL Tables (if they don't exist)...")
    await checkpointer.setup()
    
    # 3. Create the graph EXACTLY ONCE, injecting the postgres checkpointer into it!
    print("🧠 Compiling Master Graph...")
    app.state.graph = create_lead_orchestrator(checkpointer=checkpointer)
    
    print("✅ Server Online and fully connected to Database!")
    yield # --- Server handles requests here ---
    
    print("🛑 Shutting down server, cleanly closing Database Pool...")
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
