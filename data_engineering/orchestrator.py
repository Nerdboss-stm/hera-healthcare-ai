"""ETL pipeline orchestrator — Airflow-style DAG execution.

Demonstrates: task dependency resolution, DAG execution, retry logic,
SLA monitoring, backfill support, and pipeline run history.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    UPSTREAM_FAILED = "upstream_failed"


@dataclass
class TaskDefinition:
    """A single task in the DAG."""

    task_id: str
    callable_name: str  # name of the function to execute
    depends_on: list[str] = field(default_factory=list)
    retries: int = 2
    retry_delay_s: float = 0.1
    sla_seconds: float = 30.0
    description: str = ""


@dataclass
class TaskRun:
    """Runtime state of a task execution."""

    task_id: str
    state: TaskState = TaskState.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    latency_ms: float = 0.0
    attempt: int = 0
    max_attempts: int = 3
    error: str = ""
    output: Any = None
    sla_breached: bool = False


@dataclass
class DAGRun:
    """A complete DAG execution run."""

    run_id: str
    dag_id: str
    state: str = "running"  # running | success | failed
    started_at: str = ""
    completed_at: str = ""
    total_latency_ms: float = 0.0
    task_runs: dict[str, TaskRun] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "state": self.state,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": self.total_latency_ms,
            "execution_order": self.execution_order,
            "tasks": {
                tid: {
                    "state": tr.state.value,
                    "latency_ms": tr.latency_ms,
                    "attempt": tr.attempt,
                    "sla_breached": tr.sla_breached,
                    "error": tr.error,
                }
                for tid, tr in self.task_runs.items()
            },
        }


class PipelineDAG:
    """Airflow-style DAG for the HERA clinical pipeline."""

    def __init__(self, dag_id: str = "hera_clinical_pipeline"):
        self.dag_id = dag_id
        self._tasks: dict[str, TaskDefinition] = {}
        self._task_callables: dict[str, Callable] = {}
        self._run_history: list[DAGRun] = []
        self._run_counter = 0

    def add_task(self, task: TaskDefinition, callable_fn: Callable | None = None):
        """Add a task to the DAG."""
        self._tasks[task.task_id] = task
        if callable_fn:
            self._task_callables[task.task_id] = callable_fn

    def get_execution_order(self) -> list[str]:
        """Topological sort of tasks based on dependencies."""
        visited = set()
        order = []

        def _visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            task = self._tasks.get(task_id)
            if task:
                for dep in task.depends_on:
                    _visit(dep)
            order.append(task_id)

        for tid in self._tasks:
            _visit(tid)
        return order

    def validate_dag(self) -> list[str]:
        """Validate the DAG for cycles and missing dependencies."""
        errors = []
        # Check for missing dependencies
        for tid, task in self._tasks.items():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    errors.append(
                        f"Task '{tid}' depends on '{dep}' which does not exist"
                    )

        # Simple cycle detection
        visited = set()
        in_stack = set()

        def _has_cycle(tid: str) -> bool:
            if tid in in_stack:
                return True
            if tid in visited:
                return False
            visited.add(tid)
            in_stack.add(tid)
            task = self._tasks.get(tid)
            if task:
                for dep in task.depends_on:
                    if _has_cycle(dep):
                        return True
            in_stack.discard(tid)
            return False

        for tid in self._tasks:
            if _has_cycle(tid):
                errors.append(f"Cycle detected involving task '{tid}'")
                break

        return errors

    def execute(self, context: dict | None = None) -> DAGRun:
        """Execute the full DAG with retry logic and SLA monitoring."""
        self._run_counter += 1
        run = DAGRun(
            run_id=f"run-{self._run_counter:04d}",
            dag_id=self.dag_id,
        )

        context = context or {}
        order = self.get_execution_order()
        run.execution_order = order
        pipeline_start = time.time()

        for task_id in order:
            task_def = self._tasks[task_id]
            task_run = TaskRun(
                task_id=task_id,
                max_attempts=task_def.retries + 1,
            )
            run.task_runs[task_id] = task_run

            # Check upstream status
            upstream_failed = False
            for dep in task_def.depends_on:
                dep_run = run.task_runs.get(dep)
                if dep_run and dep_run.state != TaskState.SUCCESS:
                    upstream_failed = True
                    break

            if upstream_failed:
                task_run.state = TaskState.UPSTREAM_FAILED
                task_run.error = "Upstream dependency failed"
                continue

            # Execute with retries
            callable_fn = self._task_callables.get(task_id)
            for attempt in range(task_def.retries + 1):
                task_run.attempt = attempt + 1
                task_run.state = (
                    TaskState.RUNNING if attempt == 0 else TaskState.RETRYING
                )
                task_run.started_at = time.time()

                try:
                    if callable_fn:
                        result = callable_fn(context)
                        task_run.output = result
                        context[f"{task_id}_output"] = result
                    task_run.state = TaskState.SUCCESS
                    break
                except Exception as e:
                    task_run.error = str(e)
                    if attempt < task_def.retries:
                        time.sleep(task_def.retry_delay_s)
                    else:
                        task_run.state = TaskState.FAILED

                task_run.completed_at = time.time()
                task_run.latency_ms = round(
                    (task_run.completed_at - task_run.started_at) * 1000, 1
                )

            task_run.completed_at = time.time()
            task_run.latency_ms = round(
                (task_run.completed_at - task_run.started_at) * 1000, 1
            )

            # SLA check
            if task_run.latency_ms / 1000 > task_def.sla_seconds:
                task_run.sla_breached = True
                logger.warning(
                    "SLA breach: %s took %.1fms (limit: %.0fs)",
                    task_id,
                    task_run.latency_ms,
                    task_def.sla_seconds,
                )

        # Finalize run
        run.total_latency_ms = round((time.time() - pipeline_start) * 1000, 1)
        run.completed_at = datetime.now(timezone.utc).isoformat()

        all_success = all(
            tr.state == TaskState.SUCCESS for tr in run.task_runs.values()
        )
        run.state = "success" if all_success else "failed"

        self._run_history.append(run)
        return run

    def get_run_history(self, limit: int = 10) -> list[dict]:
        """Get recent DAG run history."""
        return [r.to_dict() for r in self._run_history[-limit:]]

    def get_dag_definition(self) -> dict:
        """Export DAG definition."""
        return {
            "dag_id": self.dag_id,
            "tasks": {
                tid: {
                    "task_id": t.task_id,
                    "depends_on": t.depends_on,
                    "retries": t.retries,
                    "sla_seconds": t.sla_seconds,
                    "description": t.description,
                    "callable": t.callable_name,
                }
                for tid, t in self._tasks.items()
            },
            "execution_order": self.get_execution_order(),
            "total_tasks": len(self._tasks),
            "validation_errors": self.validate_dag(),
        }


def build_hera_dag() -> PipelineDAG:
    """Build the default HERA clinical pipeline DAG."""
    dag = PipelineDAG("hera_clinical_pipeline")

    dag.add_task(
        TaskDefinition(
            task_id="ingest_data",
            callable_name="streaming.ingest",
            description="Ingest patient data through event stream with schema validation",
            sla_seconds=5.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="validate_quality",
            callable_name="quality.validate",
            depends_on=["ingest_data"],
            description="Run data quality checks on input vitals and notes",
            sla_seconds=2.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="extract_entities",
            callable_name="ner.extract",
            depends_on=["validate_quality"],
            description="Extract medical entities (medications, conditions, procedures)",
            sla_seconds=10.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="build_knowledge_graph",
            callable_name="ner.knowledge_graph",
            depends_on=["extract_entities"],
            description="Build patient knowledge graph from extracted entities",
            sla_seconds=5.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="run_triage",
            callable_name="agents.triage",
            depends_on=["validate_quality"],
            description="Triage agent assigns ESI level",
            sla_seconds=10.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="run_diagnosis",
            callable_name="agents.diagnostic",
            depends_on=["run_triage", "extract_entities"],
            description="Diagnostic agent identifies primary diagnosis using NER + vitals",
            sla_seconds=10.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="run_treatment",
            callable_name="agents.treatment",
            depends_on=["run_diagnosis"],
            description="Treatment agent plans interventions based on diagnosis",
            sla_seconds=10.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="predict_risk",
            callable_name="risk_predictor.predict",
            depends_on=["validate_quality"],
            description="Random Forest risk prediction with derived features",
            sla_seconds=5.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="retrieve_rag_context",
            callable_name="rag.query",
            depends_on=["run_diagnosis"],
            description="RAG retrieval of relevant medical guidelines via FAISS",
            sla_seconds=15.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="generate_summary",
            callable_name="summarizer.generate",
            depends_on=["retrieve_rag_context"],
            description="T5 transformer clinical note summarization",
            sla_seconds=15.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="evaluate_safety",
            callable_name="evaluator.evaluate",
            depends_on=["generate_summary"],
            description="LLM-as-Judge safety evaluation (4-axis)",
            sla_seconds=10.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="export_fhir",
            callable_name="fhir.export",
            depends_on=[
                "run_treatment",
                "predict_risk",
                "generate_summary",
                "evaluate_safety",
            ],
            description="Export entire case as FHIR R4 Bundle",
            sla_seconds=5.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="load_warehouse",
            callable_name="warehouse.load",
            depends_on=["export_fhir"],
            description="Load encounter into star schema analytics warehouse",
            sla_seconds=5.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="track_lineage",
            callable_name="lineage.track",
            depends_on=["export_fhir"],
            description="Record column-level data lineage for governance",
            sla_seconds=2.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="emit_cdc_events",
            callable_name="cdc.emit",
            depends_on=["load_warehouse"],
            description="Emit CDC change events for downstream consumers",
            sla_seconds=2.0,
        )
    )
    dag.add_task(
        TaskDefinition(
            task_id="update_catalog",
            callable_name="catalog.update",
            depends_on=["load_warehouse", "track_lineage"],
            description="Update data catalog with fresh metadata",
            sla_seconds=2.0,
        )
    )

    return dag
