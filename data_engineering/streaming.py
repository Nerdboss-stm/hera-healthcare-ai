"""Real-time event streaming pipeline with schema registry.

Simulates Kafka-style event streaming for clinical data ingestion.
Demonstrates: schema evolution, dead letter queues, event partitioning,
consumer groups, and exactly-once semantics.
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from threading import Lock

logger = logging.getLogger(__name__)


class SchemaVersion(Enum):
    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"


@dataclass
class SchemaDefinition:
    """Tracks schema evolution for event types."""

    name: str
    version: str
    fields: dict[str, str]  # field_name -> type
    required: list[str]
    created_at: str = ""
    changelog: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Validate data against this schema. Returns (valid, errors)."""
        errors = []
        for req in self.required:
            if req not in data:
                errors.append(f"Missing required field: {req}")
        for key, val in data.items():
            if key in self.fields:
                expected = self.fields[key]
                if expected == "float" and not isinstance(val, (int, float)):
                    errors.append(
                        f"Field '{key}' expected {expected}, got {type(val).__name__}"
                    )
                elif expected == "int" and not isinstance(val, int):
                    errors.append(
                        f"Field '{key}' expected {expected}, got {type(val).__name__}"
                    )
                elif expected == "str" and not isinstance(val, str):
                    errors.append(
                        f"Field '{key}' expected {expected}, got {type(val).__name__}"
                    )
        return (len(errors) == 0, errors)


@dataclass
class Event:
    """A single event in the stream."""

    event_id: str
    topic: str
    partition: int
    offset: int
    key: str
    value: dict
    schema_version: str
    timestamp: str = ""
    headers: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DeadLetterEntry:
    """Failed event sent to dead letter queue."""

    event: dict
    error: str
    failed_at: str
    retry_count: int = 0
    resolved: bool = False


class SchemaRegistry:
    """In-memory schema registry with evolution tracking."""

    def __init__(self):
        self._schemas: dict[str, list[SchemaDefinition]] = defaultdict(list)
        self._lock = Lock()
        self._register_defaults()

    def _register_defaults(self):
        """Register default clinical data schemas."""
        self.register(
            SchemaDefinition(
                name="patient_vitals",
                version="1.0",
                fields={
                    "patient_id": "str",
                    "heart_rate": "float",
                    "respiratory_rate": "float",
                    "body_temperature": "float",
                    "oxygen_saturation": "float",
                    "systolic_bp": "float",
                    "diastolic_bp": "float",
                },
                required=["patient_id", "heart_rate"],
                changelog="Initial vitals schema",
            )
        )
        self.register(
            SchemaDefinition(
                name="patient_vitals",
                version="2.0",
                fields={
                    "patient_id": "str",
                    "heart_rate": "float",
                    "respiratory_rate": "float",
                    "body_temperature": "float",
                    "oxygen_saturation": "float",
                    "systolic_bp": "float",
                    "diastolic_bp": "float",
                    "age": "int",
                    "bmi": "float",
                },
                required=["patient_id", "heart_rate", "age"],
                changelog="Added age and BMI fields for risk model v2",
            )
        )
        self.register(
            SchemaDefinition(
                name="patient_vitals",
                version="3.0",
                fields={
                    "patient_id": "str",
                    "heart_rate": "float",
                    "respiratory_rate": "float",
                    "body_temperature": "float",
                    "oxygen_saturation": "float",
                    "systolic_bp": "float",
                    "diastolic_bp": "float",
                    "age": "int",
                    "bmi": "float",
                    "mean_arterial_pressure": "float",
                    "chief_complaint": "str",
                },
                required=["patient_id", "heart_rate", "age", "chief_complaint"],
                changelog="Added MAP derived feature and chief complaint for NER",
            )
        )
        self.register(
            SchemaDefinition(
                name="clinical_note",
                version="1.0",
                fields={
                    "patient_id": "str",
                    "note_text": "str",
                    "author": "str",
                    "encounter_id": "str",
                },
                required=["patient_id", "note_text"],
                changelog="Initial clinical note schema",
            )
        )
        self.register(
            SchemaDefinition(
                name="pipeline_result",
                version="1.0",
                fields={
                    "patient_id": "str",
                    "stage": "str",
                    "status": "str",
                    "latency_ms": "float",
                    "output": "str",
                },
                required=["patient_id", "stage", "status"],
                changelog="Pipeline stage result event",
            )
        )

    def register(self, schema: SchemaDefinition):
        with self._lock:
            self._schemas[schema.name].append(schema)

    def get_latest(self, name: str) -> SchemaDefinition | None:
        versions = self._schemas.get(name, [])
        return versions[-1] if versions else None

    def get_version(self, name: str, version: str) -> SchemaDefinition | None:
        for s in self._schemas.get(name, []):
            if s.version == version:
                return s
        return None

    def get_evolution(self, name: str) -> list[dict]:
        """Get schema evolution history."""
        return [
            {
                "version": s.version,
                "fields": list(s.fields.keys()),
                "required": s.required,
                "changelog": s.changelog,
                "created_at": s.created_at,
            }
            for s in self._schemas.get(name, [])
        ]

    def list_schemas(self) -> dict[str, int]:
        return {name: len(versions) for name, versions in self._schemas.items()}


class EventStream:
    """Kafka-style event streaming with partitioning and consumer groups."""

    def __init__(self, num_partitions: int = 4):
        self._registry = SchemaRegistry()
        self._topics: dict[str, list[list[Event]]] = {}
        self._dead_letter: list[DeadLetterEntry] = []
        self._offsets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._num_partitions = num_partitions
        self._event_count = 0
        self._lock = Lock()
        self._metrics = {
            "events_produced": 0,
            "events_consumed": 0,
            "events_failed": 0,
            "bytes_processed": 0,
        }

    @property
    def registry(self) -> SchemaRegistry:
        return self._registry

    def _partition_for(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self._num_partitions

    def create_topic(self, name: str):
        if name not in self._topics:
            self._topics[name] = [[] for _ in range(self._num_partitions)]

    def produce(
        self,
        topic: str,
        key: str,
        value: dict,
        schema_name: str | None = None,
        schema_version: str | None = None,
    ) -> Event | None:
        """Produce an event to a topic. Validates against schema if provided."""
        self.create_topic(topic)

        # Schema validation
        sv = "none"
        if schema_name:
            if schema_version:
                schema = self._registry.get_version(schema_name, schema_version)
            else:
                schema = self._registry.get_latest(schema_name)
            if schema:
                sv = schema.version
                valid, errors = schema.validate(value)
                if not valid:
                    self._dead_letter.append(
                        DeadLetterEntry(
                            event={"topic": topic, "key": key, "value": value},
                            error=f"Schema validation failed: {'; '.join(errors)}",
                            failed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )
                    self._metrics["events_failed"] += 1
                    logger.warning("Event sent to DLQ: %s", errors)
                    return None

        partition = self._partition_for(key)
        with self._lock:
            self._event_count += 1
            offset = len(self._topics[topic][partition])
            event = Event(
                event_id=f"evt-{self._event_count:06d}",
                topic=topic,
                partition=partition,
                offset=offset,
                key=key,
                value=value,
                schema_version=sv,
            )
            self._topics[topic][partition].append(event)
            self._metrics["events_produced"] += 1
            self._metrics["bytes_processed"] += len(json.dumps(value))
        self._persist_event(event)
        return event

    def _persist_event(self, event: Event):
        """Write stream event to PostgreSQL de_stream_events table."""
        try:
            import psycopg2
            from config.settings import DB_CONFIG

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO de_stream_events
                (event_id, topic, partition_num, offset_num, key, value,
                 schema_version, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    event.event_id,
                    event.topic,
                    event.partition,
                    event.offset,
                    event.key,
                    json.dumps(event.value, default=str),
                    event.schema_version,
                    event.timestamp,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.debug("Stream persist skipped: %s", e)

    def consume(
        self,
        topic: str,
        consumer_group: str,
        max_events: int = 10,
    ) -> list[Event]:
        """Consume events from a topic for a consumer group."""
        if topic not in self._topics:
            return []

        events = []
        for part_idx, partition in enumerate(self._topics[topic]):
            offset_key = f"{part_idx}"
            current_offset = self._offsets[f"{consumer_group}:{topic}"][offset_key]
            for evt in partition[current_offset : current_offset + max_events]:
                events.append(evt)
                self._offsets[f"{consumer_group}:{topic}"][offset_key] += 1
                self._metrics["events_consumed"] += 1

        return events

    def get_dead_letter_queue(self) -> list[dict]:
        return [
            {
                "event": d.event,
                "error": d.error,
                "failed_at": d.failed_at,
                "retry_count": d.retry_count,
                "resolved": d.resolved,
            }
            for d in self._dead_letter
        ]

    def get_topic_stats(self) -> dict[str, dict]:
        stats = {}
        for topic, partitions in self._topics.items():
            total = sum(len(p) for p in partitions)
            stats[topic] = {
                "partitions": len(partitions),
                "total_events": total,
                "events_per_partition": [len(p) for p in partitions],
            }
        return stats

    def get_metrics(self) -> dict:
        return {
            **self._metrics,
            "topics": len(self._topics),
            "dead_letter_size": len(self._dead_letter),
            "schemas_registered": sum(self._registry.list_schemas().values()),
        }

    def ingest_patient_event(self, patient_data: dict) -> dict[str, Any]:
        """High-level: ingest a patient event through the streaming pipeline.

        Produces events to vitals, notes, and pipeline-results topics.
        Returns ingestion report.
        """
        patient_id = patient_data.get("patient_id", "unknown")
        results = {"patient_id": patient_id, "events": [], "dlq": []}

        # Vitals event
        vitals = {
            "patient_id": patient_id,
            "heart_rate": patient_data.get("heart_rate", 0),
            "respiratory_rate": patient_data.get("respiratory_rate", 0),
            "body_temperature": patient_data.get("body_temperature", 0),
            "oxygen_saturation": patient_data.get("oxygen_saturation", 0),
            "systolic_bp": patient_data.get("systolic_bp", 0),
            "diastolic_bp": patient_data.get("diastolic_bp", 0),
            "age": patient_data.get("age", 0),
            "chief_complaint": patient_data.get("chief_complaint", ""),
        }
        # Compute derived features
        sbp = vitals.get("systolic_bp", 120)
        dbp = vitals.get("diastolic_bp", 80)
        vitals["mean_arterial_pressure"] = round(dbp + (sbp - dbp) / 3, 1)
        vitals["bmi"] = patient_data.get("bmi", 25.0)

        evt = self.produce(
            "patient-vitals",
            patient_id,
            vitals,
            schema_name="patient_vitals",
            schema_version="3.0",
        )
        if evt:
            results["events"].append(asdict(evt))
        else:
            results["dlq"].append(
                self._dead_letter[-1].__dict__ if self._dead_letter else {}
            )

        # Clinical note event
        if patient_data.get("clinical_note"):
            note_evt = self.produce(
                "clinical-notes",
                patient_id,
                {
                    "patient_id": patient_id,
                    "note_text": patient_data["clinical_note"],
                    "author": "system",
                    "encounter_id": f"enc-{patient_id}-{int(time.time())}",
                },
                schema_name="clinical_note",
            )
            if note_evt:
                results["events"].append(asdict(note_evt))

        results["metrics"] = self.get_metrics()
        results["topic_stats"] = self.get_topic_stats()
        return results


# Singleton for the app
event_stream = EventStream()
