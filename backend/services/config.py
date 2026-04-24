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
    min_body_sections_default: int
    max_body_sections_deep: int
    base_section_word_target: int
    priority_section_word_target: int
    priority_section_count: int
    min_evidence_cards_per_draftable_section: int
    min_distinct_sources_per_draftable_section: int
    min_quant_signals_for_numeric_sections: int
    max_targeted_gap_rounds_deep: int
    max_verifier_sections: int
    max_repair_passes: int
    max_priority_expansions: int
    search_scrape_cache_max_entries: int
    relevant_scrape_chunk_count: int
    provider_cooldown_seconds: int
    quota_cooldown_seconds: int
    rate_limit_max_cooldown_seconds: int
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
    groq_secondary_model: str
    groq_secondary_request_limit_per_minute: int
    groq_secondary_token_limit_per_minute: int
    groq_tertiary_model: str
    groq_tertiary_request_limit_per_minute: int
    groq_tertiary_token_limit_per_minute: int

    # Cerebras Fallback Models (Fast Inference)
    cerebras_model: str
    cerebras_request_limit_per_minute: int
    cerebras_token_limit_per_minute: int
    cerebras_max_parallel_requests: int
    cerebras_secondary_model: str
    cerebras_secondary_request_limit_per_minute: int
    cerebras_secondary_token_limit_per_minute: int

    openrouter_base_url: str
    openrouter_model: str
    openrouter_request_limit_per_minute: int
    openrouter_token_limit_per_minute: int
    openrouter_max_parallel_requests: int
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
        # --- Task limits (ceilings — the planner prompt decides actual count dynamically) ---
        max_initial_tasks=_int("MAX_INITIAL_TASKS", 4),
        max_active_sub_agents_per_job=_int("MAX_ACTIVE_SUB_AGENTS_PER_JOB", 1),
        max_gap_tasks_per_round=_int("MAX_GAP_TASKS_PER_ROUND", 3),
        max_gap_rounds=_int("MAX_GAP_ROUNDS", 2),
        max_sub_agent_iterations=_int("MAX_SUB_AGENT_ITERATIONS", 10),
        max_active_jobs=_int("MAX_ACTIVE_JOBS", 2),
        max_queue_depth=_int("MAX_QUEUE_DEPTH", 25),
        queue_wait_timeout_seconds=_float("QUEUE_WAIT_TIMEOUT_SECONDS", 1.0),
        heartbeat_interval_seconds=_float("SSE_HEARTBEAT_INTERVAL_SECONDS", 10.0),
        client_rate_limit_per_minute=_int("CLIENT_RATE_LIMIT_PER_MINUTE", 10),
        queue_full_retry_after_seconds=_int("QUEUE_FULL_RETRY_AFTER_SECONDS", 5),
        search_cache_ttl_seconds=_int("SEARCH_CACHE_TTL_SECONDS", 900),
        scrape_cache_ttl_seconds=_int("SCRAPE_CACHE_TTL_SECONDS", 86400),
        working_summary_char_limit=_int("WORKING_SUMMARY_CHAR_LIMIT", 6000),
        recent_message_count=_int("RECENT_MESSAGE_COUNT", 4),
        tool_result_char_limit=_int("TOOL_RESULT_CHAR_LIMIT", 6000),
        search_result_limit=_int("SEARCH_RESULT_LIMIT", 8),
        # --- Token budgets sized for Groq free tier (Qwen3 = 6K TPM) ---
        planner_input_char_budget=_int("PLANNER_INPUT_CHAR_BUDGET", 12000),
        worker_input_char_budget=_int("WORKER_INPUT_CHAR_BUDGET", 12000),
        # Synthesis gets larger budgets for Gemini-level report quality
        synthesis_input_char_budget=_int("SYNTHESIS_INPUT_CHAR_BUDGET", 32000),
        section_draft_output_tokens=_int("SECTION_DRAFT_OUTPUT_TOKENS", 2000),
        final_report_output_tokens=_int("FINAL_REPORT_OUTPUT_TOKENS", 6000),
        # --- Evidence thresholds lowered for free-tier throughput ---
        min_scraped_sources_per_task=_int("MIN_SCRAPED_SOURCES_PER_TASK", 2),
        min_evidence_cards_per_task=_int("MIN_EVIDENCE_CARDS_PER_TASK", 2),
        min_authoritative_sources_per_task=_int("MIN_AUTHORITATIVE_SOURCES_PER_TASK", 1),
        min_distinct_sources_for_report=_int("MIN_DISTINCT_SOURCES_FOR_REPORT", 6),
        min_authoritative_sources_for_report=_int("MIN_AUTHORITATIVE_SOURCES_FOR_REPORT", 3),
        min_evidence_cards_for_report=_int("MIN_EVIDENCE_CARDS_FOR_REPORT", 4),
        min_sources_per_section=_int("MIN_SOURCES_PER_SECTION", 2),
        min_body_sections_default=_int("MIN_BODY_SECTIONS_DEFAULT", 6),
        max_body_sections_deep=_int("MAX_BODY_SECTIONS_DEEP", 8),
        base_section_word_target=_int("BASE_SECTION_WORD_TARGET", 850),
        priority_section_word_target=_int("PRIORITY_SECTION_WORD_TARGET", 1200),
        priority_section_count=_int("PRIORITY_SECTION_COUNT", 2),
        min_evidence_cards_per_draftable_section=_int("MIN_EVIDENCE_CARDS_PER_DRAFTABLE_SECTION", 2),
        min_distinct_sources_per_draftable_section=_int("MIN_DISTINCT_SOURCES_PER_DRAFTABLE_SECTION", 2),
        min_quant_signals_for_numeric_sections=_int("MIN_QUANT_SIGNALS_FOR_NUMERIC_SECTIONS", 1),
        max_targeted_gap_rounds_deep=_int("MAX_TARGETED_GAP_ROUNDS_DEEP", 1),
        max_verifier_sections=_int("MAX_VERIFIER_SECTIONS", 2),
        max_repair_passes=_int("MAX_REPAIR_PASSES", 1),
        max_priority_expansions=_int("MAX_PRIORITY_EXPANSIONS", 2),
        search_scrape_cache_max_entries=_int("SEARCH_SCRAPE_CACHE_MAX_ENTRIES", 256),
        relevant_scrape_chunk_count=_int("RELEVANT_SCRAPE_CHUNK_COUNT", 4),
        # --- Provider resilience timers ---
        # Cooldown after a confirmed rate-limit 429 from the provider API
        provider_cooldown_seconds=_int("PROVIDER_COOLDOWN_SECONDS", 20),
        # Circuit-breaker for true billing/credit exhaustion (payment required, plan limit)
        quota_cooldown_seconds=_int("QUOTA_COOLDOWN_SECONDS", 120),
        # Hard cap on any rate-limit cooldown — prevents runaway stalls
        rate_limit_max_cooldown_seconds=_int("RATE_LIMIT_MAX_COOLDOWN_SECONDS", 60),
        transient_retry_base_seconds=_float("TRANSIENT_RETRY_BASE_SECONDS", 1.0),
        transient_retry_jitter_seconds=_float("TRANSIENT_RETRY_JITTER_SECONDS", 0.5),
        # Total time _execute() will keep polling for an available provider before raising
        provider_unavailable_wait_seconds=_float("PROVIDER_UNAVAILABLE_WAIT_SECONDS", 180.0),
        provider_poll_interval_seconds=_float("PROVIDER_POLL_INTERVAL_SECONDS", 5.0),
        # --- Groq (PRIMARY — fast, reliable free tier: 30 RPM, 15K TPM) ---
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_request_limit_per_minute=_int("GROQ_REQUEST_LIMIT_PER_MINUTE", 14),
        groq_token_limit_per_minute=_int("GROQ_TOKEN_LIMIT_PER_MINUTE", 15000),
        groq_max_parallel_requests=_int("GROQ_MAX_PARALLEL_REQUESTS", 1),
        # --- Groq Secondary (Qwen3 32B — ACTUAL limit is 6K TPM, we leave 1K headroom) ---
        groq_secondary_model=os.getenv("GROQ_SECONDARY_MODEL", "qwen/qwen3-32b"),
        groq_secondary_request_limit_per_minute=_int("GROQ_SECONDARY_REQUEST_LIMIT_PER_MINUTE", 14),
        groq_secondary_token_limit_per_minute=_int("GROQ_SECONDARY_TOKEN_LIMIT_PER_MINUTE", 5000),
        # --- Groq Tertiary (llama-3.1-8b — lightweight, fast, reliable tool calling) ---
        groq_tertiary_model=os.getenv("GROQ_TERTIARY_MODEL", "llama-3.1-8b-instant"),
        groq_tertiary_request_limit_per_minute=_int("GROQ_TERTIARY_REQUEST_LIMIT_PER_MINUTE", 14),
        groq_tertiary_token_limit_per_minute=_int("GROQ_TERTIARY_TOKEN_LIMIT_PER_MINUTE", 15000),
        # --- Cerebras fallback parameters ---
        cerebras_model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
        cerebras_request_limit_per_minute=_int("CEREBRAS_REQUEST_LIMIT_PER_MINUTE", 30),
        cerebras_token_limit_per_minute=_int("CEREBRAS_TOKEN_LIMIT_PER_MINUTE", 60000),
        cerebras_max_parallel_requests=_int("CEREBRAS_MAX_PARALLEL_REQUESTS", 2),
        cerebras_secondary_model=os.getenv("CEREBRAS_SECONDARY_MODEL", "qwen-3-235b-a22b-instruct-2507"),
        cerebras_secondary_request_limit_per_minute=_int("CEREBRAS_SECONDARY_REQUEST_LIMIT_PER_MINUTE", 30),
        cerebras_secondary_token_limit_per_minute=_int("CEREBRAS_SECONDARY_TOKEN_LIMIT_PER_MINUTE", 30000),
        # --- Gemini (SECONDARY — flash-lite: 20 RPD free tier, resets midnight PT) ---
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        gemini_request_limit_per_minute=_int("GEMINI_REQUEST_LIMIT_PER_MINUTE", 15),
        gemini_token_limit_per_minute=_int("GEMINI_TOKEN_LIMIT_PER_MINUTE", 250000),
        gemini_max_parallel_requests=_int("GEMINI_MAX_PARALLEL_REQUESTS", 1),
        # --- OpenRouter (free model auto-router picks best available free model) ---
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        openrouter_request_limit_per_minute=_int("OPENROUTER_REQUEST_LIMIT_PER_MINUTE", 10),
        openrouter_token_limit_per_minute=_int("OPENROUTER_TOKEN_LIMIT_PER_MINUTE", 30000),
        openrouter_max_parallel_requests=_int("OPENROUTER_MAX_PARALLEL_REQUESTS", 2),
        # --- HuggingFace (ENABLED — free API, serves as additional fallback) ---
        huggingface_base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        huggingface_model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        huggingface_request_limit_per_minute=_int("HUGGINGFACE_REQUEST_LIMIT_PER_MINUTE", 10),
        huggingface_token_limit_per_minute=_int("HUGGINGFACE_TOKEN_LIMIT_PER_MINUTE", 30000),
        huggingface_max_parallel_requests=_int("HUGGINGFACE_MAX_PARALLEL_REQUESTS", 1),
        enable_huggingface_router=_bool("ENABLE_HUGGINGFACE_ROUTER", True),
    )
