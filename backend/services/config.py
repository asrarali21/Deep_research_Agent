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
    search_result_limit: int
    planner_input_char_budget: int
    worker_input_char_budget: int
    synthesis_input_char_budget: int
    section_draft_output_tokens: int
    final_report_output_tokens: int
    min_scraped_sources_per_task: int
    min_evidence_cards_per_task: int
    min_authoritative_sources_per_task: int
    min_distinct_sources_for_report: int
    min_authoritative_sources_for_report: int
    min_evidence_cards_for_report: int
    min_sources_per_section: int
    provider_cooldown_seconds: int
    quota_cooldown_seconds: int
    transient_retry_base_seconds: float
    transient_retry_jitter_seconds: float
    provider_unavailable_wait_seconds: float
    provider_poll_interval_seconds: float
    gemini_model: str
    gemini_request_limit_per_minute: int
    gemini_token_limit_per_minute: int
    gemini_max_parallel_requests: int
    groq_model: str
    groq_request_limit_per_minute: int
    groq_token_limit_per_minute: int
    groq_max_parallel_requests: int
    huggingface_base_url: str
    huggingface_model: str
    huggingface_request_limit_per_minute: int
    huggingface_token_limit_per_minute: int
    huggingface_max_parallel_requests: int
    enable_huggingface_router: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        max_initial_tasks=_int("MAX_INITIAL_TASKS", 4),
        max_active_sub_agents_per_job=_int("MAX_ACTIVE_SUB_AGENTS_PER_JOB", 1),
        max_gap_tasks_per_round=_int("MAX_GAP_TASKS_PER_ROUND", 3),
        max_gap_rounds=_int("MAX_GAP_ROUNDS", 3),
        max_sub_agent_iterations=_int("MAX_SUB_AGENT_ITERATIONS", 8),
        max_active_jobs=_int("MAX_ACTIVE_JOBS", 2),
        max_queue_depth=_int("MAX_QUEUE_DEPTH", 25),
        queue_wait_timeout_seconds=_float("QUEUE_WAIT_TIMEOUT_SECONDS", 1.0),
        heartbeat_interval_seconds=_float("SSE_HEARTBEAT_INTERVAL_SECONDS", 10.0),
        client_rate_limit_per_minute=_int("CLIENT_RATE_LIMIT_PER_MINUTE", 10),
        queue_full_retry_after_seconds=_int("QUEUE_FULL_RETRY_AFTER_SECONDS", 5),
        search_cache_ttl_seconds=_int("SEARCH_CACHE_TTL_SECONDS", 900),
        scrape_cache_ttl_seconds=_int("SCRAPE_CACHE_TTL_SECONDS", 86400),
        working_summary_char_limit=_int("WORKING_SUMMARY_CHAR_LIMIT", 8000),
        recent_message_count=_int("RECENT_MESSAGE_COUNT", 6),
        tool_result_char_limit=_int("TOOL_RESULT_CHAR_LIMIT", 8000),
        search_result_limit=_int("SEARCH_RESULT_LIMIT", 12),
        planner_input_char_budget=_int("PLANNER_INPUT_CHAR_BUDGET", 16000),
        worker_input_char_budget=_int("WORKER_INPUT_CHAR_BUDGET", 24000),
        synthesis_input_char_budget=_int("SYNTHESIS_INPUT_CHAR_BUDGET", 36000),
        section_draft_output_tokens=_int("SECTION_DRAFT_OUTPUT_TOKENS", 1200),
        final_report_output_tokens=_int("FINAL_REPORT_OUTPUT_TOKENS", 4200),
        min_scraped_sources_per_task=_int("MIN_SCRAPED_SOURCES_PER_TASK", 3),
        min_evidence_cards_per_task=_int("MIN_EVIDENCE_CARDS_PER_TASK", 5),
        min_authoritative_sources_per_task=_int("MIN_AUTHORITATIVE_SOURCES_PER_TASK", 1),
        min_distinct_sources_for_report=_int("MIN_DISTINCT_SOURCES_FOR_REPORT", 10),
        min_authoritative_sources_for_report=_int("MIN_AUTHORITATIVE_SOURCES_FOR_REPORT", 5),
        min_evidence_cards_for_report=_int("MIN_EVIDENCE_CARDS_FOR_REPORT", 14),
        min_sources_per_section=_int("MIN_SOURCES_PER_SECTION", 2),
        provider_cooldown_seconds=_int("PROVIDER_COOLDOWN_SECONDS", 30),
        quota_cooldown_seconds=_int("QUOTA_COOLDOWN_SECONDS", 900),
        transient_retry_base_seconds=_float("TRANSIENT_RETRY_BASE_SECONDS", 1.0),
        transient_retry_jitter_seconds=_float("TRANSIENT_RETRY_JITTER_SECONDS", 0.5),
        provider_unavailable_wait_seconds=_float("PROVIDER_UNAVAILABLE_WAIT_SECONDS", 90.0),
        provider_poll_interval_seconds=_float("PROVIDER_POLL_INTERVAL_SECONDS", 5.0),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_request_limit_per_minute=_int("GEMINI_REQUEST_LIMIT_PER_MINUTE", 10),
        gemini_token_limit_per_minute=_int("GEMINI_TOKEN_LIMIT_PER_MINUTE", 80000),
        gemini_max_parallel_requests=_int("GEMINI_MAX_PARALLEL_REQUESTS", 1),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_request_limit_per_minute=_int("GROQ_REQUEST_LIMIT_PER_MINUTE", 15),
        groq_token_limit_per_minute=_int("GROQ_TOKEN_LIMIT_PER_MINUTE", 40000),
        groq_max_parallel_requests=_int("GROQ_MAX_PARALLEL_REQUESTS", 1),
        huggingface_base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        huggingface_model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        huggingface_request_limit_per_minute=_int("HUGGINGFACE_REQUEST_LIMIT_PER_MINUTE", 10),
        huggingface_token_limit_per_minute=_int("HUGGINGFACE_TOKEN_LIMIT_PER_MINUTE", 30000),
        huggingface_max_parallel_requests=_int("HUGGINGFACE_MAX_PARALLEL_REQUESTS", 1),
        enable_huggingface_router=_bool("ENABLE_HUGGINGFACE_ROUTER", False),
    )
