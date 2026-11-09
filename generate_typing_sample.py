#!/usr/bin/env python3
"""Append recent typing sessions for p001-sample to a lifelog SQLite database."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, time, timedelta, timezone
import json
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
PARTICIPANT = "p001-sample"
DAY_INSERTS = (100, 110, 120, 130, 140, 150, 160)
DAY_BURSTS = (6, 9, 12, 16, 20, 45, 56)
DAY_SESSIONS = (2, 1, 3, 2, 1, 3, 2)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def stored_at(value: datetime) -> str:
    return iso(value.astimezone(timezone.utc))


def split(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def existing_identity(db: sqlite3.Connection) -> tuple[str, str]:
    sessions = db.execute(
        "SELECT payload FROM records "
        "WHERE participant=? AND kind='session' ORDER BY seq",
        (PARTICIPANT,),
    ).fetchall()
    if not sessions:
        raise RuntimeError(f"no existing session records for {PARTICIPANT}")

    devices: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    for (raw_payload,) in sessions:
        payload = json.loads(raw_payload)
        if payload.get("device_id"):
            devices[payload["device_id"]] += 1
        for event in payload.get("events", []):
            if event.get("type") in {"kana", "text_insert", "delete"}:
                mode = event.get("data", {}).get("input_mode")
                if mode:
                    modes[mode] += 1
    if not devices or not modes:
        raise RuntimeError("existing sample sessions have no typing device/input_mode")
    return devices.most_common(1)[0][0], modes.most_common(1)[0][0]


def session_window_starts(day_index: int, count: int, now: datetime) -> list[datetime]:
    day = now.date() - timedelta(days=6 - day_index)
    if day == now.date():
        latest_end = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
        if latest_end > now - timedelta(minutes=5):
            latest_end -= timedelta(minutes=10)
        latest_start = latest_end - timedelta(minutes=10)
        if latest_end <= datetime.combine(day, time.min, JST):
            raise RuntimeError("too early in the JST day to create a completed fresh window")
        return [latest_start - timedelta(minutes=10 * (count - 1 - i)) for i in range(count)]

    anchors = (time(9, 0), time(14, 0), time(20, 0))
    return [datetime.combine(day, anchors[index], JST) for index in range(count)]


def build_events(
    start: datetime, inserts: int, bursts: int, input_mode: str
) -> list[dict]:
    """Make each correction two rapid deletes; topology counts it as one burst."""
    corrected = set()
    for index in range(bursts):
        corrected.add((index * inserts) // bursts)

    events = []
    for index in range(inserts):
        at = start + timedelta(seconds=index * 2.5)
        if index % 2:
            events.append(
                {
                    "time": iso(at),
                    "type": "text_insert",
                    "data": {"text": "例", "input_mode": input_mode},
                }
            )
        else:
            events.append(
                {
                    "time": iso(at),
                    "type": "kana",
                    "data": {
                        "kana": "あ",
                        "composition": "あ",
                        "input_mode": input_mode,
                    },
                }
            )
        if index in corrected:
            for offset in (0.4, 0.9):
                events.append(
                    {
                        "time": iso(at + timedelta(seconds=offset)),
                        "type": "delete",
                        "data": {
                            "target": "composition",
                            "input_mode": input_mode,
                        },
                    }
                )
    return sorted(events, key=lambda event: event["time"])


def build_session(
    window_start: datetime,
    inserts: int,
    bursts: int,
    device_id: str,
    input_mode: str,
) -> tuple[str, datetime, dict]:
    identifier = str(uuid4())
    started_at = window_start + timedelta(seconds=20)
    events = build_events(started_at + timedelta(seconds=5), inserts, bursts, input_mode)
    last_event = datetime.fromisoformat(events[-1]["time"])
    ended_at = last_event + timedelta(seconds=5)
    if ended_at >= window_start + timedelta(minutes=10):
        raise RuntimeError("generated session does not fit in its 10-minute window")

    delete_count = bursts * 2
    payload = {
        "participant_id": PARTICIPANT,
        "session_id": identifier,
        "device_id": device_id,
        "started_at": iso(started_at),
        "ended_at": iso(ended_at),
        "close_reason": "sample_generation",
        "events": events,
        "typing": {
            "total_typed_characters": inserts,
            "backspace_count": delete_count,
            "backspace_rate": delete_count / inserts,
            "session_duration_sec": (ended_at - started_at).total_seconds(),
        },
        "context": None,
        "received_at": iso(ended_at + timedelta(seconds=1)),
        "provenance": "typing-change-sample-v1",
    }
    return identifier, ended_at, payload


def generate(db_path: str, now: datetime) -> dict:
    db = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(records)").fetchall()
        }
        required = {"participant", "kind", "id", "at", "payload"}
        if not required <= columns:
            raise RuntimeError("records table does not have the expected schema")

        device_id, input_mode = existing_identity(db)
        rows = []
        days = []
        for day_index, (inserts, bursts, count) in enumerate(
            zip(DAY_INSERTS, DAY_BURSTS, DAY_SESSIONS)
        ):
            insert_parts = split(inserts, count)
            burst_parts = split(bursts, count)
            starts = session_window_starts(day_index, count, now)
            for window_start, session_inserts, session_bursts in zip(
                starts, insert_parts, burst_parts
            ):
                identifier, ended_at, payload = build_session(
                    window_start,
                    session_inserts,
                    session_bursts,
                    device_id,
                    input_mode,
                )
                rows.append(
                    (
                        PARTICIPANT,
                        "session",
                        identifier,
                        stored_at(ended_at),
                        json.dumps(payload, ensure_ascii=False),
                    )
                )
            days.append(
                {
                    "date": starts[0].date().isoformat(),
                    "sessions": count,
                    "inserts": inserts,
                    "bursts": bursts,
                    "correction_rate": bursts / inserts,
                }
            )

        with db:
            db.executemany(
                "INSERT INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                rows,
            )
        return {
            "participant": PARTICIPANT,
            "device_id": device_id,
            "input_mode": input_mode,
            "inserted_sessions": len(rows),
            "days": days,
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", help="path to the SQLite database to update")
    parser.add_argument(
        "--now",
        help="optional aware ISO timestamp used as 'now' (for reproducible verification)",
    )
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now).astimezone(JST) if args.now else datetime.now(JST)
    print(json.dumps(generate(args.db, now), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
