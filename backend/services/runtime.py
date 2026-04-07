from services.config import Settings, get_settings
from services.coordination import CoordinationStore, InMemoryCoordinationStore, create_coordination_store
from services.model_router import ModelRouter


_settings: Settings | None = None
_coordination_store: CoordinationStore | None = None
_model_router: ModelRouter | None = None
_job_manager = None


def _bootstrap_local_runtime() -> tuple[Settings, CoordinationStore, ModelRouter]:
    global _settings, _coordination_store, _model_router

    if _settings is None:
        _settings = get_settings()
    if _coordination_store is None:
        _coordination_store = InMemoryCoordinationStore()
    if _model_router is None:
        _model_router = ModelRouter(_coordination_store, _settings)
    return _settings, _coordination_store, _model_router


async def initialize_runtime() -> tuple[Settings, CoordinationStore, ModelRouter]:
    global _settings, _coordination_store, _model_router

    if _settings is None:
        _settings = get_settings()
    if _coordination_store is None:
        _coordination_store = await create_coordination_store(_settings)
    if _model_router is None:
        _model_router = ModelRouter(_coordination_store, _settings)
    return _settings, _coordination_store, _model_router


def get_settings_instance() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_coordination_store() -> CoordinationStore:
    return _bootstrap_local_runtime()[1]


def get_model_router() -> ModelRouter:
    return _bootstrap_local_runtime()[2]


def set_job_manager(job_manager) -> None:
    global _job_manager
    _job_manager = job_manager


def get_job_manager():
    if _job_manager is None:
        raise RuntimeError("Job manager has not been initialized.")
    return _job_manager
