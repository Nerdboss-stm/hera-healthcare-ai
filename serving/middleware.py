"""Production middleware — API key authentication, rate limiting, usage metering.

Designed for multi-tenant SaaS deployment where multiple healthcare
organizations share the same HERA instance with isolated data and billing.
"""

from __future__ import annotations

import time
import hashlib
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# In production, these would come from a database/secrets manager.
# For demo purposes, allow unauthenticated access but track usage.
API_KEYS: dict[str, dict] = {
    "hera-demo-key": {
        "tenant": "demo",
        "tier": "free",
        "rate_limit": 60,  # requests per minute
        "daily_limit": 1000,
    },
}


class UsageTracker:
    """In-memory usage tracking for demo. Production would use Redis/PostgreSQL."""

    def __init__(self):
        self._minute_counts: dict[str, list[float]] = defaultdict(list)
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._total_counts: dict[str, int] = defaultdict(int)
        self._endpoint_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def record(self, tenant: str, endpoint: str, latency_ms: float):
        now = time.time()
        self._minute_counts[tenant].append(now)
        self._daily_counts[tenant] += 1
        self._total_counts[tenant] += 1
        self._endpoint_counts[tenant][endpoint] += 1
        self._latencies[tenant].append(latency_ms)
        # Prune old minute entries
        cutoff = now - 60
        self._minute_counts[tenant] = [
            t for t in self._minute_counts[tenant] if t > cutoff
        ]

    def get_minute_count(self, tenant: str) -> int:
        cutoff = time.time() - 60
        self._minute_counts[tenant] = [
            t for t in self._minute_counts[tenant] if t > cutoff
        ]
        return len(self._minute_counts[tenant])

    def get_usage(self, tenant: str) -> dict:
        return {
            "tenant": tenant,
            "requests_this_minute": self.get_minute_count(tenant),
            "requests_today": self._daily_counts.get(tenant, 0),
            "requests_total": self._total_counts.get(tenant, 0),
            "endpoints": dict(self._endpoint_counts.get(tenant, {})),
            "avg_latency_ms": round(
                sum(self._latencies.get(tenant, [0]))
                / max(len(self._latencies.get(tenant, [1])), 1),
                1,
            ),
        }

    def get_all_usage(self) -> dict:
        tenants = set(list(self._total_counts.keys()) + ["anonymous"])
        return {
            t: self.get_usage(t) for t in tenants if self._total_counts.get(t, 0) > 0
        }


usage_tracker = UsageTracker()


class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication with tenant isolation.

    - Requests with valid X-API-Key header get full access
    - Requests without a key get "anonymous" tenant access (for demo)
    - Rate limiting is per-tenant
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for static files, health, docs, and root
        path = request.url.path
        if path in (
            "/",
            "/api/health",
            "/docs",
            "/openapi.json",
            "/metrics",
        ) or path.startswith("/static"):
            response = await call_next(request)
            return response

        # Extract API key
        api_key = request.headers.get("X-API-Key", "")
        tenant = "anonymous"
        tier = "free"
        rate_limit = 30  # anonymous gets 30/min

        if api_key and api_key in API_KEYS:
            key_data = API_KEYS[api_key]
            tenant = key_data["tenant"]
            tier = key_data["tier"]
            rate_limit = key_data["rate_limit"]

        # Rate limiting
        current_count = usage_tracker.get_minute_count(tenant)
        if current_count >= rate_limit:
            logger.warning(
                "Rate limit exceeded for tenant %s (%d/%d)",
                tenant,
                current_count,
                rate_limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "tenant": tenant,
                    "limit": rate_limit,
                    "retry_after_seconds": 60,
                },
            )

        # Attach tenant info to request state
        request.state.tenant = tenant
        request.state.tier = tier

        # Execute request and track usage
        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000
        usage_tracker.record(tenant, path, latency_ms)

        # Add tenant headers to response
        response.headers["X-Tenant"] = tenant
        response.headers["X-Tier"] = tier
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, rate_limit - current_count - 1)
        )
        response.headers["X-Request-Latency-Ms"] = f"{latency_ms:.1f}"

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """HIPAA-compliant audit logging for all patient data access.

    In production, these logs would go to an immutable audit store
    (e.g., AWS CloudTrail, Azure Monitor, or a dedicated HIPAA log database).
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Only audit API calls that touch patient data
        if not path.startswith("/api/") or method == "OPTIONS":
            return await call_next(request)

        tenant = getattr(request.state, "tenant", "unknown")
        start = time.time()

        # Generate a request trace ID
        trace_id = hashlib.sha256(f"{time.time()}{path}{tenant}".encode()).hexdigest()[
            :12
        ]

        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000

        # Log the audit entry
        logger.info(
            "AUDIT | trace=%s tenant=%s method=%s path=%s status=%d latency=%.1fms",
            trace_id,
            tenant,
            method,
            path,
            response.status_code,
            latency_ms,
        )

        response.headers["X-Trace-Id"] = trace_id
        return response
