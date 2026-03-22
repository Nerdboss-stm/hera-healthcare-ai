from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest

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

# Clinical-specific metrics
RISK_PREDICTIONS = Counter(
    "hera_risk_predictions_total",
    "Risk predictions by level",
    ["risk_level"],
)

ESI_ASSIGNMENTS = Counter(
    "hera_esi_assignments_total",
    "ESI triage assignments by level",
    ["esi_level"],
)

PIPELINE_STAGES = Counter(
    "hera_pipeline_stages_total",
    "Pipeline stage completions",
    ["stage", "status"],
)

ACTIVE_PATIENTS = Gauge(
    "hera_active_patients",
    "Number of patients processed",
)

CDC_EVENTS = Counter(
    "hera_cdc_events_total",
    "CDC change events by table",
    ["table", "change_type"],
)

WAREHOUSE_ENCOUNTERS = Counter(
    "hera_warehouse_encounters_total",
    "Encounters loaded into warehouse",
)

HERA_INFO = Info(
    "hera_build",
    "HERA build information",
)
HERA_INFO.info({"version": "4.0.0", "backend": "postgresql"})

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
