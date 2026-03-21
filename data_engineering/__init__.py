"""HERA Data Engineering Layer — Production-grade data infrastructure.

Modules:
    streaming     - Real-time event pipeline with schema registry
    lineage       - Column-level data lineage DAG tracking
    quality       - Data quality framework (Great Expectations-style)
    warehouse     - Star schema analytics warehouse (SQLite)
    orchestrator  - ETL pipeline orchestrator (Airflow-style DAG)
    cdc           - Change Data Capture audit stream
    catalog       - Data catalog & metadata management
"""
