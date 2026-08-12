"""Politeness: token-bucket limiting, backoff on retryable failures, and an
HTTP client that actually reads the rate-limit headers instead of guessing.

The rule for this project: every outbound call goes through here. Instances ban
fast, GDELT throttles, and YouTube counts every unit -- and none of those
failures are recoverable after the fact.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

#: HTTP statuses worth retrying: throttling and transient upstream failures.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TokenBucket:
    """Classic token bucket. Thread-safe, monotonic-clock based.

    ``burst`` lets a source spend a small backlog immediately (useful when a
    paginated fetch wakes up after a long parse) without exceeding the average
    rate over any meaningful window.
    """

    def __init__(self, rate_per_sec: float, burst: int | None = None):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = float(rate_per_sec)
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_sec)))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns seconds actually slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return slept
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            time.sleep(wait)
            slept += wait

    def __enter__(self) -> TokenBucket:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def sleep_until(when: datetime, *, cap_seconds: float = 900.0, label: str = "rate limit") -> float:
    """Sleep until an absolute reset time, capped so a bad header cannot hang a run."""
    now = datetime.now(timezone.utc)
    seconds = (when - now).total_seconds()
    if seconds <= 0:
        return 0.0
    seconds = min(seconds, cap_seconds)
    log.warning("%s: sleeping %.1fs until %s", label, seconds, when.isoformat())
    time.sleep(seconds)
    return seconds


def with_retry(
    *,
    attempts: int = 5,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential backoff with jitter.

    Uses tenacity when it is installed and falls back to an equivalent loop
    otherwise, so the retry policy is identical either way and a missing
    optional dependency cannot silently turn retries off.
    """
    try:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        return retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=lambda state: log.warning(
                "retry %d/%d for %s after %s",
                state.attempt_number,
                attempts,
                getattr(state.fn, "__name__", "call"),
                state.outcome.exception() if state.outcome else "?",
            ),
        )
    except ImportError:  # pragma: no cover - tenacity is a core dependency

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                wait = initial_wait
                for attempt in range(1, attempts + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exceptions as exc:
                        if attempt == attempts:
                            raise
                        delay = min(wait, max_wait) * (1 + random.random())
                        log.warning(
                            "retry %d/%d for %s after %s", attempt, attempts, fn.__name__, exc
                        )
                        time.sleep(delay)
                        wait *= 2
                raise AssertionError("unreachable")

            return wrapper

        return decorator


class RateLimitedSession:
    """``requests.Session`` + token bucket + backoff + honest header handling.

    On a 429 or a depleted ``X-RateLimit-Remaining`` we sleep until the reset
    timestamp the server gave us. Hammering past that is how a research account
    gets an instance-wide ban, which costs far more than the wait.
    """

    def __init__(
        self,
        user_agent: str,
        rate_per_sec: float = 2.0,
        burst: int | None = None,
        timeout: float = 30.0,
    ):
        import requests

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.bucket = TokenBucket(rate_per_sec, burst)
        self.timeout = timeout

    def get(self, url: str, **kwargs: Any):
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs: Any):
        return self.request("HEAD", url, **kwargs)

    def request(self, method: str, url: str, *, max_attempts: int = 5, **kwargs: Any):
        import requests

        kwargs.setdefault("timeout", self.timeout)
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            self.bucket.acquire()
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == max_attempts:
                    raise
                self._backoff(attempt, reason=str(exc))
                continue

            if response.status_code in RETRYABLE_STATUS:
                if attempt == max_attempts:
                    response.raise_for_status()
                self._respect_headers(response, attempt)
                continue

            self._note_remaining(response)
            return response
        if last_exc:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError(f"request to {url} exhausted retries")

    # --- internals -------------------------------------------------------
    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(60.0, 2 ** (attempt - 1)) * (1 + random.random() * 0.25)
        log.warning("http retry %d after %s; sleeping %.1fs", attempt, reason, delay)
        time.sleep(delay)

    def _respect_headers(self, response: Any, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 60.0
            log.warning("server asked for Retry-After=%s; sleeping", retry_after)
            time.sleep(min(delay, 900.0))
            return
        reset = self._reset_time(response)
        if reset is not None:
            sleep_until(reset, label=f"http {response.status_code}")
            return
        self._backoff(attempt, reason=f"status {response.status_code}")

    def _note_remaining(self, response: Any) -> None:
        """Pre-emptively sleep when the quota window is exhausted but not yet 429."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is None:
            return
        try:
            if int(remaining) > 0:
                return
        except ValueError:
            return
        reset = self._reset_time(response)
        if reset is not None:
            sleep_until(reset, label="rate limit window exhausted")

    @staticmethod
    def _reset_time(response: Any) -> datetime | None:
        raw = response.headers.get("X-RateLimit-Reset") or response.headers.get("RateLimit-Reset")
        if not raw:
            return None
        raw = raw.strip()
        # Mastodon sends ISO8601; most others send an epoch or a delta-seconds.
        try:
            if raw.replace(".", "", 1).isdigit():
                value = float(raw)
                if value < 10_000_000:  # a delta, not an epoch
                    return datetime.now(timezone.utc).fromtimestamp(
                        time.time() + value, tz=timezone.utc
                    )
                return datetime.fromtimestamp(value, tz=timezone.utc)
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    def close(self) -> None:
        self.session.close()
