# deep_research

`deep_research` is a local deep-research app with a FastAPI backend and a Next.js frontend. A user submits a research question, reviews the generated plan, then resumes the run to collect sources, evidence cards, section drafts, and a final Markdown report.

The backend uses LangGraph to coordinate research tasks and streams progress to the frontend over server-sent events.

## Repository Layout

- `backend/` - FastAPI app, LangGraph agents, provider routing, search/scrape tools, job queue, and backend tests.
- `frontend/` - Next.js chat UI, streamed research state, source panel, local session persistence, and frontend tests.
- `docker-compose.yml` - PostgreSQL and Redis services for local development.
- `services/` - currently contains an empty placeholder file.

## Architecture / How It Works

1. The frontend posts a query to `POST /api/research/start`.
2. The backend creates a job record, queues the job, and starts an SSE stream.
3. The lead orchestrator decomposes the query into research tasks and pauses before plan execution.
4. The frontend shows the plan. The user can continue as-is or send feedback through `POST /api/research/resume`.
5. Sub-agents run bounded search/scrape loops, submit findings, and produce structured evidence cards.
6. The orchestrator evaluates coverage, fills gaps when needed, builds section packets, drafts sections, verifies selected sections, and assembles the final report.
7. The frontend renders timeline updates, sources, status, and the final Markdown report.

Key backend pieces:

- `backend/main.py` - FastAPI app setup, PostgreSQL checkpoint setup, runtime initialization.
- `backend/api/user_query.py` - research start/resume/status endpoints and SSE formatting.
- `backend/services/job_manager.py` - job queue, replayable events, status tracking, quota-wait requeue handling.
- `backend/agents/lead_orchestrator_agent.py` - LangGraph research workflow.
- `backend/agents/sub_agent.py` - worker graph for search, scrape, and evidence submission.
- `backend/services/model_router.py` - model-provider selection, local quota guards, retries, cooldowns, and fallback.
- `backend/tools/firecrawl_tool.py` - search and scrape helpers with Firecrawl, DuckDuckGo, and trafilatura paths.

## Requirements

- Python 3.11+ recommended for the backend.
- Node.js and npm for the frontend.
- Docker for PostgreSQL and Redis.
- At least one model provider API key:
  - `GROQ_API_KEY`
  - `CEREBRAS_API_KEY`
  - `GEMINI_API_KEY`
  - `OPENROUTER_API_KEY` or `OPEN_ROUTER_API_KEY`
  - `HUGGINGFACE_API_KEY`
- Optional search/scrape key:
  - `FIRECRAWL_API_KEY`

## Configuration

Backend environment variables:

```bash
POSTGRES_DB_URL=postgresql://researcher:research_pass@localhost:5433/deep_research
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=your_groq_key
FIRECRAWL_API_KEY=your_firecrawl_key
```

The backend also supports tuning variables for queue limits, evidence thresholds, model names, provider rate limits, token budgets, and cooldowns. See `backend/services/config.py` for the full list and defaults.

Frontend environment variables:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

If omitted, the frontend defaults to `http://localhost:8000`.

## Installation

Start local infrastructure:

```bash
docker compose up -d
```

Set up the backend:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install \
  fastapi uvicorn pydantic python-dotenv \
  langgraph langchain-core langchain-google-genai langchain-groq langchain-openai langchain-community \
  psycopg[binary,pool] redis firecrawl-py trafilatura duckduckgo-search
```

No `requirements.txt` or `pyproject.toml` is currently committed for the backend. The command above is based on the imports used in `backend/`.

Set up the frontend:

```bash
cd frontend
npm install
```

## Running Locally

Run the backend from the `backend/` directory so its local imports resolve:

```bash
cd backend
source ../.venv/bin/activate
uvicorn main:app --host localhost --port 8000 --reload
```

Run the frontend:

```bash
cd frontend
npm run dev
```

Open the app at:

```text
http://localhost:3000
```

Health check:

```bash
curl http://localhost:8000/health
```

## API

### `GET /health`

Returns a basic backend health response.

### `POST /api/research/start`

Starts a research run and returns an SSE stream.

Request body:

```json
{
  "query": "Research question"
}
```

The response includes an `X-Thread-ID` header. Events may include `queued`, `started`, `plan`, `source_batch`, `evidence_batch`, `evaluate`, `outline`, `evidence_brief`, `section_draft`, `section_verification`, `synthesize`, `report`, `paused`, `done`, `failed`, `heartbeat`, and `waiting_for_quota`.

### `POST /api/research/resume`

Resumes a paused run after plan review.

Request body:

```json
{
  "thread_id": "existing-thread-id",
  "feedback": "Optional plan feedback"
}
```

### `GET /api/research/status/{thread_id}`

Returns queue position, status, plan, evidence counts, retry counts, provider switch counts, quota wait details, and the last error.

## Testing

Backend tests:

```bash
cd backend
python -m unittest discover tests
```

Frontend tests:

```bash
cd frontend
npm test
```

## Output Examples

_Add real outputs here_

## Limitations / Trade-offs

- Backend dependencies are not pinned in a committed manifest yet.
- The backend requires PostgreSQL for LangGraph checkpoints.
- Redis is optional at runtime; if Redis cannot be reached, coordination falls back to an in-memory store.
- Runs pause after plan generation because the graph is compiled with an interrupt before `plan_review_node`.
- Search and scrape quality depends on configured tools and upstream availability.
- Provider quotas can pause jobs; the job manager emits `waiting_for_quota` and requeues after the cooldown.
- CORS is currently configured with `allow_origins=["*"]`; restrict this before deployment.

## License

No license file is currently included.
