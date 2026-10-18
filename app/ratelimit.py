"""Tiny in-process sliding-window rate limiter for the public JSON endpoints.

It is per worker process and keyed by client address. Behind a reverse proxy run uvicorn with
``--proxy-headers`` so the real client address is used, and apply a proper limit at the proxy
for multi-worker deployments.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional


class RateLimiter:
    def __init__(self, limit_per_window: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self._clock = limit_per_window, window_s, clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> Optional[float]:
        """Return ``None`` if the request is allowed, otherwise seconds until a slot frees up."""
        if self.limit <= 0:
            return None
        now = self._clock()
        if len(self._hits) > 10_000:                       # bound memory
            self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1.0, self.window - (now - hits[0]))
        hits.append(now)
        return None
