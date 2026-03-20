from prometheus_client import Counter, Histogram, generate_latest
import time

REQUEST_COUNT = Counter(
    "summarizer_requests_total",
    "Total number of requests to the summarizer endpoint"
)

REQUEST_FAILURES = Counter(
    "summarizer_failures_total",
    "Number of failed summarization requests"
)

REQUEST_LATENCY = Histogram(
    "summarizer_request_latency_seconds",
    "Latency of summarization requests"
)

def track_metrics(func):
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        start_time = time.time()
        try:
            response = func(*args, **kwargs)
            return response
        except Exception:
            REQUEST_FAILURES.inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
    return wrapper

def prometheus_metrics():
    return generate_latest()

