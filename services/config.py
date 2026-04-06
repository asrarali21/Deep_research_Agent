import os
from dataclasses import dataclass
from functools import lru_cache


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    redis_url: str | None
    max_initial_tasks: int
    max_active_sub_agents_per_job: int
    max_gap_tasks_per_round: int
    max_gap_rounds: int
    max_sub_agent_iterations: int
    max_active_jobs: int
    max_queue_depth: int
    queue_wait_timeout_seconds: float
    heartbeat_interval_seconds: float
    client_rate_limit_per_minute: int
    queue_full_retry_after_seconds: int
    search_cache_ttl_seconds: int
    scrape_cache_ttl_seconds: int
    working_summary_char_limit: int
    recent_message_count: int
    tool_result_char_limit: int
    planner_input_char_budget: int
    worker_input_char_budget: int
    synthesis_input_char_budget: int
    provider_cooldown_seconds: int
    quota_cooldown_seconds: int
    transient_retry_base_seconds: float
    transient_retry_jitter_seconds: float
    provider_unavailable_wait_seconds: float
    gemini_model: str
    groq_model: str
    huggingface_base_url: str
    huggingface_model: str
    enable_huggingface_router: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        max_initial_tasks=_int("MAX_INITIAL_TASKS", 4),
        max_active_sub_agents_per_job=_int("MAX_ACTIVE_SUB_AGENTS_PER_JOB", 2),
        max_gap_tasks_per_round=_int("MAX_GAP_TASKS_PER_ROUND", 2),
        max_gap_rounds=_int("MAX_GAP_ROUNDS", 2),
        max_sub_agent_iterations=_int("MAX_SUB_AGENT_ITERATIONS", 6),
        max_active_jobs=_int("MAX_ACTIVE_JOBS", 2),
        max_queue_depth=_int("MAX_QUEUE_DEPTH", 25),
        queue_wait_timeout_seconds=_float("QUEUE_WAIT_TIMEOUT_SECONDS", 1.0),
        heartbeat_interval_seconds=_float("SSE_HEARTBEAT_INTERVAL_SECONDS", 10.0),
        client_rate_limit_per_minute=_int("CLIENT_RATE_LIMIT_PER_MINUTE", 10),
        queue_full_retry_after_seconds=_int("QUEUE_FULL_RETRY_AFTER_SECONDS", 5),
        search_cache_ttl_seconds=_int("SEARCH_CACHE_TTL_SECONDS", 900),
        scrape_cache_ttl_seconds=_int("SCRAPE_CACHE_TTL_SECONDS", 86400),
        working_summary_char_limit=_int("WORKING_SUMMARY_CHAR_LIMIT", 4000),
        recent_message_count=_int("RECENT_MESSAGE_COUNT", 4),
        tool_result_char_limit=_int("TOOL_RESULT_CHAR_LIMIT", 4000),
        planner_input_char_budget=_int("PLANNER_INPUT_CHAR_BUDGET", 12000),
        worker_input_char_budget=_int("WORKER_INPUT_CHAR_BUDGET", 14000),
        synthesis_input_char_budget=_int("SYNTHESIS_INPUT_CHAR_BUDGET", 18000),
        provider_cooldown_seconds=_int("PROVIDER_COOLDOWN_SECONDS", 30),
        quota_cooldown_seconds=_int("QUOTA_COOLDOWN_SECONDS", 900),
        transient_retry_base_seconds=_float("TRANSIENT_RETRY_BASE_SECONDS", 1.0),
        transient_retry_jitter_seconds=_float("TRANSIENT_RETRY_JITTER_SECONDS", 0.5),
        provider_unavailable_wait_seconds=_float("PROVIDER_UNAVAILABLE_WAIT_SECONDS", 20.0),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        huggingface_base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        huggingface_model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        enable_huggingface_router=_bool("ENABLE_HUGGINGFACE_ROUTER", False),
    )
