"""Change Data Capture (CDC) — immutable event stream for audit trail.

Demonstrates: before/after snapshots, diff computation, event sourcing,
replayable event log, and downstream consumer notification.
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    """An immutable CDC event capturing a state change."""

    event_id: str
    table: str
    record_key: str
    change_type: str  # INSERT, UPDATE, DELETE
    before: dict | None  # state before change (None for INSERT)
    after: dict | None  # state after change (None for DELETE)
    diff: dict  # changed fields only
    timestamp: str = ""
    sequence_number: int = 0
    checksum: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.checksum:
            payload = json.dumps(
                {
                    "table": self.table,
                    "key": self.record_key,
                    "type": self.change_type,
                    "after": self.after,
                },
                sort_keys=True,
            )
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()[:16]


class CDCStream:
    """Immutable CDC event log with replay support."""

    def __init__(self):
        self._log: list[ChangeEvent] = []
        self._snapshots: dict[
            str, dict[str, dict]
        ] = {}  # table -> key -> current_state
        self._sequence = 0
        self._consumers: dict[str, int] = {}  # consumer_name -> last_read_sequence

    def _compute_diff(self, before: dict | None, after: dict | None) -> dict:
        """Compute field-level diff between two states."""
        if before is None:
            return {"added": after or {}}
        if after is None:
            return {"removed": before}

        diff = {}
        all_keys = set(list(before.keys()) + list(after.keys()))
        for key in all_keys:
            old_val = before.get(key)
            new_val = after.get(key)
            if old_val != new_val:
                diff[key] = {"before": old_val, "after": new_val}
        return diff

    def capture_change(
        self,
        table: str,
        record_key: str,
        change_type: ChangeType,
        new_state: dict | None = None,
    ) -> ChangeEvent:
        """Capture a state change and emit a CDC event."""
        self._sequence += 1

        # Get current state (before)
        current = self._snapshots.get(table, {}).get(record_key)

        diff = self._compute_diff(current, new_state)

        event = ChangeEvent(
            event_id=f"cdc-{self._sequence:06d}",
            table=table,
            record_key=record_key,
            change_type=change_type.value,
            before=current,
            after=new_state,
            diff=diff,
            sequence_number=self._sequence,
        )
        self._log.append(event)
        self._persist_event(event)

        # Update snapshot
        if table not in self._snapshots:
            self._snapshots[table] = {}
        if change_type == ChangeType.DELETE:
            self._snapshots[table].pop(record_key, None)
        elif new_state:
            self._snapshots[table][record_key] = new_state

        return event

    def _persist_event(self, event: ChangeEvent):
        """Write CDC event to PostgreSQL de_cdc_events table."""
        try:
            import psycopg2
            from config.settings import DB_CONFIG

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO de_cdc_events
                (event_id, table_name, record_key, change_type,
                 before_state, after_state, diff, checksum, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    event.event_id,
                    event.table,
                    event.record_key,
                    event.change_type,
                    json.dumps(event.before, default=str) if event.before else None,
                    json.dumps(event.after, default=str) if event.after else None,
                    json.dumps(event.diff, default=str) if event.diff else None,
                    event.checksum,
                    event.timestamp,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.debug("CDC persist skipped: %s", e)

    def capture_encounter(self, encounter_data: dict) -> list[ChangeEvent]:
        """Capture all changes for a clinical encounter."""
        events = []
        patient_id = encounter_data.get("patient_id", "unknown")

        # Patient record change
        events.append(
            self.capture_change(
                table="patients",
                record_key=patient_id,
                change_type=ChangeType.INSERT,
                new_state={
                    "patient_id": patient_id,
                    "age": encounter_data.get("age", 0),
                    "gender": encounter_data.get("gender", "unknown"),
                },
            )
        )

        # Vitals record
        vitals = {
            k: v
            for k, v in encounter_data.items()
            if k
            in (
                "heart_rate",
                "respiratory_rate",
                "body_temperature",
                "oxygen_saturation",
                "systolic_bp",
                "diastolic_bp",
            )
        }
        if vitals:
            events.append(
                self.capture_change(
                    table="vitals",
                    record_key=f"{patient_id}-vitals",
                    change_type=ChangeType.INSERT,
                    new_state={"patient_id": patient_id, **vitals},
                )
            )

        # Pipeline results
        for stage in encounter_data.get("stages", []):
            stage_name = stage.get("system", stage.get("name", "unknown"))
            events.append(
                self.capture_change(
                    table="pipeline_results",
                    record_key=f"{patient_id}-{stage_name}",
                    change_type=ChangeType.INSERT,
                    new_state={
                        "patient_id": patient_id,
                        "stage": stage_name,
                        "status": stage.get("status", "unknown"),
                        "latency_ms": stage.get("latency_ms", 0),
                    },
                )
            )

        return events

    def consume(self, consumer_name: str, max_events: int = 50) -> list[dict]:
        """Consume events from the log for a named consumer."""
        last_seq = self._consumers.get(consumer_name, 0)
        events = []
        for evt in self._log:
            if evt.sequence_number > last_seq:
                events.append(asdict(evt))
                if len(events) >= max_events:
                    break

        if events:
            self._consumers[consumer_name] = events[-1]["sequence_number"]
        return events

    def replay(self, table: str | None = None, from_sequence: int = 0) -> list[dict]:
        """Replay events from a point in time. Used for rebuilding state."""
        events = []
        for evt in self._log:
            if evt.sequence_number < from_sequence:
                continue
            if table and evt.table != table:
                continue
            events.append(asdict(evt))
        return events

    def get_current_state(self, table: str, record_key: str) -> dict | None:
        """Get current state from the snapshot store."""
        return self._snapshots.get(table, {}).get(record_key)

    def get_record_history(self, table: str, record_key: str) -> list[dict]:
        """Get full change history for a specific record."""
        return [
            asdict(evt)
            for evt in self._log
            if evt.table == table and evt.record_key == record_key
        ]

    def get_stats(self) -> dict:
        """Get CDC stream statistics."""
        tables = set(evt.table for evt in self._log)
        by_table = {}
        for t in tables:
            by_table[t] = sum(1 for evt in self._log if evt.table == t)

        by_type = {}
        for ct in ChangeType:
            by_type[ct.value] = sum(
                1 for evt in self._log if evt.change_type == ct.value
            )

        return {
            "total_events": len(self._log),
            "sequence_number": self._sequence,
            "tables_tracked": sorted(tables),
            "events_by_table": by_table,
            "events_by_type": by_type,
            "consumers": {
                name: {"last_sequence": seq} for name, seq in self._consumers.items()
            },
            "snapshot_tables": {
                t: len(records) for t, records in self._snapshots.items()
            },
        }


# Singleton
cdc_stream = CDCStream()
