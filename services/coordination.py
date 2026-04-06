import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

from services.config import Settings


@dataclass(slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


@dataclass(slots=True)
class QuotaReservation:
    allowed: bool
    retry_after_seconds: int
    requests_used: int
    tokens_used: int


class CoordinationStore(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        raise NotImplementedError

    @abstractmethod
    async def reserve_quota(
        self,
        key: str,
        request_cost: int,
        token_cost: int,
        request_limit: int,
        token_limit: int,
        window_seconds: int,
    ) -> QuotaReservation:
        raise NotImplementedError

    @abstractmethod
    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def get_ttl_seconds(self, key: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def increment_counter(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    async def enqueue_job(self, job_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def dequeue_job(self, timeout_seconds: float) -> tuple[str, dict[str, Any]] | None:
        raise NotImplementedError

    @abstractmethod
    async def queue_depth(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def queue_position(self, job_id: str) -> int | None:
        raise NotImplementedError

    @abstractmethod
    async def set_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_job_record(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def update_job_record(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def increment_job_metric(self, job_id: str, field: str, amount: int = 1) -> int:
        raise NotImplementedError


class InMemoryCoordinationStore(CoordinationStore):
    def __init__(self) -> None:
        self._data_lock = asyncio.Lock()
        self._queue_condition = asyncio.Condition()
        self._queue: deque[str] = deque()
        self._queue_payloads: dict[str, dict[str, Any]] = {}
        self._kv: dict[str, tuple[Any, float | None]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._rate_limits: dict[str, tuple[float, int]] = {}
        self._quota_windows: dict[str, tuple[float, int, int]] = {}

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _purge_if_expired(self, key: str) -> None:
        if key not in self._kv:
            return
        _, expires_at = self._kv[key]
        if expires_at is not None and expires_at <= time.time():
            self._kv.pop(key, None)

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        now = time.time()
        async with self._data_lock:
            window_end, count = self._rate_limits.get(key, (now + window_seconds, 0))
            if now >= window_end:
                window_end = now + window_seconds
                count = 0
            if count >= limit:
                return RateLimitDecision(allowed=False, retry_after_seconds=max(1, int(window_end - now)))
            self._rate_limits[key] = (window_end, count + 1)
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

    async def reserve_quota(
        self,
        key: str,
        request_cost: int,
        token_cost: int,
        request_limit: int,
        token_limit: int,
        window_seconds: int,
    ) -> QuotaReservation:
        now = time.time()
        async with self._data_lock:
            window_end, requests_used, tokens_used = self._quota_windows.get(
                key,
                (now + window_seconds, 0, 0),
            )
            if now >= window_end:
                window_end = now + window_seconds
                requests_used = 0
                tokens_used = 0

            next_requests = requests_used + request_cost
            next_tokens = tokens_used + token_cost
            if next_requests > request_limit or next_tokens > token_limit:
                return QuotaReservation(
                    allowed=False,
                    retry_after_seconds=max(1, int(window_end - now)),
                    requests_used=requests_used,
                    tokens_used=tokens_used,
                )

            self._quota_windows[key] = (window_end, next_requests, next_tokens)
            return QuotaReservation(
                allowed=True,
                retry_after_seconds=0,
                requests_used=next_requests,
                tokens_used=next_tokens,
            )

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        async with self._data_lock:
            self._kv[key] = (value, expires_at)

    async def get_json(self, key: str) -> Any | None:
        async with self._data_lock:
            self._purge_if_expired(key)
            value = self._kv.get(key)
            return None if value is None else value[0]

    async def get_ttl_seconds(self, key: str) -> int:
        async with self._data_lock:
            self._purge_if_expired(key)
            value = self._kv.get(key)
            if value is None:
                return 0
            _, expires_at = value
            if expires_at is None:
                return 0
            return max(0, int(expires_at - time.time()))

    async def increment_counter(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        async with self._data_lock:
            self._purge_if_expired(key)
            current = self._kv.get(key, (0, None))[0]
            next_value = int(current) + amount
            expires_at = time.time() + ttl_seconds if ttl_seconds else self._kv.get(key, (None, None))[1]
            self._kv[key] = (next_value, expires_at)
            return next_value

    async def enqueue_job(self, job_id: str, payload: dict[str, Any]) -> None:
        async with self._queue_condition:
            if job_id in self._queue_payloads:
                return
            self._queue.append(job_id)
            self._queue_payloads[job_id] = payload
            self._queue_condition.notify()

    async def dequeue_job(self, timeout_seconds: float) -> tuple[str, dict[str, Any]] | None:
        deadline = time.time() + timeout_seconds
        async with self._queue_condition:
            while not self._queue:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._queue_condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None

            job_id = self._queue.popleft()
            payload = self._queue_payloads.pop(job_id)
            return job_id, payload

    async def queue_depth(self) -> int:
        async with self._queue_condition:
            return len(self._queue)

    async def queue_position(self, job_id: str) -> int | None:
        async with self._queue_condition:
            for index, queued_id in enumerate(self._queue, start=1):
                if queued_id == job_id:
                    return index
            return None

    async def set_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        async with self._data_lock:
            self._jobs[job_id] = dict(record)

    async def get_job_record(self, job_id: str) -> dict[str, Any] | None:
        async with self._data_lock:
            record = self._jobs.get(job_id)
            return None if record is None else dict(record)

    async def update_job_record(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        async with self._data_lock:
            if job_id not in self._jobs:
                return None
            self._jobs[job_id].update(updates)
            return dict(self._jobs[job_id])

    async def increment_job_metric(self, job_id: str, field: str, amount: int = 1) -> int:
        async with self._data_lock:
            if job_id not in self._jobs:
                return 0
            next_value = int(self._jobs[job_id].get(field, 0)) + amount
            self._jobs[job_id][field] = next_value
            return next_value


class RedisCoordinationStore(CoordinationStore):
    def __init__(self, redis_url: str) -> None:
        try:
            import redis.asyncio as redis_async
        except ImportError as exc:
            raise RuntimeError("redis package is required for Redis coordination") from exc

        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._queue_key = "deep_research:queue"

    async def start(self) -> None:
        await self._redis.ping()

    async def close(self) -> None:
        await self._redis.aclose()

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        window_key = f"deep_research:ratelimit:{key}:{int(time.time() // window_seconds)}"
        current = await self._redis.incr(window_key)
        if current == 1:
            await self._redis.expire(window_key, window_seconds)
        if current > limit:
            ttl = await self._redis.ttl(window_key)
            return RateLimitDecision(allowed=False, retry_after_seconds=max(1, ttl))
        return RateLimitDecision(allowed=True, retry_after_seconds=0)

    async def reserve_quota(
        self,
        key: str,
        request_cost: int,
        token_cost: int,
        request_limit: int,
        token_limit: int,
        window_seconds: int,
    ) -> QuotaReservation:
        window_suffix = int(time.time() // window_seconds)
        request_key = f"deep_research:quota:req:{key}:{window_suffix}"
        token_key = f"deep_research:quota:tok:{key}:{window_suffix}"

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(request_key, request_cost)
            pipe.incrby(token_key, token_cost)
            pipe.ttl(request_key)
            requests_used, tokens_used, ttl = await pipe.execute()

        if requests_used == request_cost:
            await self._redis.expire(request_key, window_seconds)
        if tokens_used == token_cost:
            await self._redis.expire(token_key, window_seconds)

        if requests_used > request_limit or tokens_used > token_limit:
            await self._redis.decrby(request_key, request_cost)
            await self._redis.decrby(token_key, token_cost)
            return QuotaReservation(
                allowed=False,
                retry_after_seconds=max(1, ttl if ttl and ttl > 0 else window_seconds),
                requests_used=int(requests_used - request_cost),
                tokens_used=int(tokens_used - token_cost),
            )

        return QuotaReservation(
            allowed=True,
            retry_after_seconds=0,
            requests_used=int(requests_used),
            tokens_used=int(tokens_used),
        )

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        namespaced_key = f"deep_research:data:{key}"
        payload = json.dumps(value)
        if ttl_seconds:
            await self._redis.set(namespaced_key, payload, ex=ttl_seconds)
        else:
            await self._redis.set(namespaced_key, payload)

    async def get_json(self, key: str) -> Any | None:
        namespaced_key = f"deep_research:data:{key}"
        payload = await self._redis.get(namespaced_key)
        return None if payload is None else json.loads(payload)

    async def get_ttl_seconds(self, key: str) -> int:
        ttl = await self._redis.ttl(f"deep_research:data:{key}")
        return max(0, ttl)

    async def increment_counter(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        namespaced_key = f"deep_research:counter:{key}"
        current = await self._redis.incrby(namespaced_key, amount)
        if ttl_seconds and current == amount:
            await self._redis.expire(namespaced_key, ttl_seconds)
        return int(current)

    async def enqueue_job(self, job_id: str, payload: dict[str, Any]) -> None:
        await self.set_json(f"job_payload:{job_id}", payload)
        await self._redis.rpush(self._queue_key, job_id)

    async def dequeue_job(self, timeout_seconds: float) -> tuple[str, dict[str, Any]] | None:
        timeout = max(1, int(timeout_seconds))
        item = await self._redis.blpop(self._queue_key, timeout=timeout)
        if item is None:
            return None
        _, job_id = item
        payload = await self.get_json(f"job_payload:{job_id}")
        return None if payload is None else (job_id, payload)

    async def queue_depth(self) -> int:
        return int(await self._redis.llen(self._queue_key))

    async def queue_position(self, job_id: str) -> int | None:
        queued_ids = await self._redis.lrange(self._queue_key, 0, -1)
        for index, queued_id in enumerate(queued_ids, start=1):
            if queued_id == job_id:
                return index
        return None

    async def set_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        await self.set_json(f"job_record:{job_id}", record)

    async def get_job_record(self, job_id: str) -> dict[str, Any] | None:
        return await self.get_json(f"job_record:{job_id}")

    async def update_job_record(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        record = await self.get_job_record(job_id)
        if record is None:
            return None
        record.update(updates)
        await self.set_job_record(job_id, record)
        return record

    async def increment_job_metric(self, job_id: str, field: str, amount: int = 1) -> int:
        record = await self.get_job_record(job_id)
        if record is None:
            return 0
        next_value = int(record.get(field, 0)) + amount
        record[field] = next_value
        await self.set_job_record(job_id, record)
        return next_value


async def create_coordination_store(settings: Settings) -> CoordinationStore:
    if settings.redis_url:
        try:
            store: CoordinationStore = RedisCoordinationStore(settings.redis_url)
            await store.start()
            return store
        except Exception:
            pass

    store = InMemoryCoordinationStore()
    await store.start()
    return store
