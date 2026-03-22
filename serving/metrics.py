from prometheus_client import Counter, Histogram, generate_latest
import time

REQUEST_COUNT = Counter(
    "hera_requests_total",
    "Total API requests",
    ["endpoint"],
)

REQUEST_FAILURES = Counter(
    "hera_request_failures_total",
    "Failed API requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "hera_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# Keep old names for backward compatibility with existing Prometheus queries
SUMMARIZER_COUNT = Counter(
    "summarizer_requests_total", "Total summarizer requests"
)
SUMMARIZER_FAILURES = Counter(
    "summarizer_failures_total", "Failed summarizer requests"
)
SUMMARIZER_LATENCY = Histogram(
    "summarizer_request_latency_seconds", "Summarizer request latency"
)


def prometheus_metrics():
    return generate_latest()
