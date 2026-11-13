"""Deterministic sample lifelog generation from calendar-tags-v1 input."""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re
import math
import os

import numpy as np

from .data import JST, SCALES


SCHEMA = "calendar-tags-v1"
MANIFEST_SCHEMA = "sample-data-v1"
TAGS = {"lab", "work", "drinking", "travel", "none"}
ASSESSMENTS = (("currentState", 9, 0), ("currentState", 15, 0), ("nightReflection", 22, 15))
BASE = {
    "morning": np.array([3.4, 2.4, 2.3, 3.4, 2.3]),
    "afternoon": np.array([3.3, 2.8, 2.6, 3.3, 2.5]),
    "night": np.array([3.3, 3.0, 2.5, 2.8, 3.2]),
}
TAG_EFFECTS = {
    "lab": np.array([0.1, 0.2, 0.35, 0.75, 0.0]),
    "work": np.array([0.0, 0.2, 0.3, 0.5, 0.0]),
    "drinking": np.array([0.35, 0.65, -0.15, -0.55, 0.65]),
    "travel": np.array([0.25, 0.35, 0.05, -0.25, 0.25]),
    "none": np.zeros(5),
}
ACTIVITY = {"lab": "study", "work": "desk_work", "travel": "commute", "drinking": "friends", "none": "relax"}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _iso(value):
    return value.isoformat(timespec="milliseconds")


def _at(day, hour, minute=0):
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=JST)


def _record(kind, identifier, participant, at, payload):
    return dict(kind=kind, id=identifier, participant=participant, at=_iso(at), payload=payload)


def _calendar_events(day):
    result = []
    for event in day.get("events", []):
        result.append(
            {
                "start": _iso(_at(day["date"], *map(int, event["start"].split(":")))),
                "end": _iso(_at(day["date"], *map(int, event["end"].split(":")))),
                "title": f"sample:{event['tag']}",
            }
        )
    return result


def _scores(rng, day, period, previous_tags):
    values = BASE[period].copy()
    tags = set(day["tags"])
    for tag in tags:
        values += TAG_EFFECTS[tag]
    if "drinking" in previous_tags and period == "morning":
        values += np.array([-0.35, 0.9, 0.2, -0.5, 1.0])
    common = rng.normal(0, 0.35)
    values += common * np.array([0.7, -0.5, -0.35, 0.55, -0.5])
    values += rng.normal(0, 0.18, len(SCALES))
    return {key: int(np.clip(math.floor(value + 0.5), 1, 5)) for key, value in zip(SCALES, values)}


def _context(day, at, index):
    updated = at - timedelta(seconds=20)
    steps = 1200 + index * 137 + at.hour * 83
    measurement = {key: _iso(updated) for key in (
        "step_count", "walking_distance_m", "flights_climbed", "active_energy_kcal",
        "walking_speed_mps", "walking_step_length_m", "walking_asymmetry_pct",
        "battery_level", "calendar_events",
    )}
    return {
        "step_count": steps,
        "walking_distance_m": round(steps * 0.71, 2),
        "flights_climbed": index % 7,
        "active_energy_kcal": round(steps * 0.035, 2),
        "walking_speed_mps": round(1.1 + (index % 5) * 0.04, 2),
        "walking_step_length_m": round(0.66 + (index % 4) * 0.02, 2),
        "walking_asymmetry_pct": round(1.2 + (index % 6) * 0.25, 2),
        "battery_level": round(max(0.2, 0.92 - at.hour * 0.025), 2),
        "battery_state": "unplugged",
        "calendar_events": _calendar_events(day),
        "measurement_updated_at": measurement,
        "updated_at": _iso(updated),
        "living_presence": True,
    }


def _session(day, participant, assessment_index, side, center, day_index):
    if side == "before":
        start = center - timedelta(minutes=5)
    elif side == "fresh":
        start = center - timedelta(minutes=4)
    else:
        start = center + timedelta(minutes=2)
    end = start + timedelta(minutes=2)
    stem = f"sample-{day['date']}-session-{assessment_index}-{side}"
    events = [
        {"time": _iso(start + timedelta(seconds=10)), "type": "key", "data": {"keyID": "さ", "direction": "left"}},
        {"time": _iso(start + timedelta(seconds=11)), "type": "kana", "data": {"kana": "し", "composition": "し", "input_mode": "hiragana"}},
        {"time": _iso(start + timedelta(seconds=18)), "type": "delete", "data": {"target": "composition", "input_mode": "hiragana"}},
        {"time": _iso(start + timedelta(seconds=25)), "type": "key", "data": {"keyID": "か", "direction": "center"}},
        {"time": _iso(start + timedelta(seconds=26)), "type": "kana", "data": {"kana": "か", "composition": "か", "input_mode": "hiragana"}},
        {"time": _iso(start + timedelta(seconds=35)), "type": "commit", "data": {"text": "サンプル観測"}},
    ]
    payload = {
        "participant_id": participant,
        "session_id": stem,
        "device_id": "sample-device",
        "started_at": _iso(start),
        "ended_at": _iso(end),
        "close_reason": "sample_generation",
        "events": events,
        "typing": {
            "total_typed_characters": 7,
            "backspace_count": 1,
            "backspace_rate": 1 / 7,
            "session_duration_sec": 120,
        },
        "context": _context(day, end, day_index * 6 + assessment_index * 2 + (side == "after")),
        "received_at": _iso(end + timedelta(seconds=1)),
        "provenance": MANIFEST_SCHEMA,
    }
    before = _record(
        "heartbeat", stem + "-heartbeat-before", participant, start - timedelta(seconds=20),
        {"event_id": stem + "-heartbeat-before", "participant_id": participant, "occurred_at": _iso(start - timedelta(seconds=20)), "source": "sample", "logging_enabled": True, "questions_enabled": False, "interventions_enabled": False},
    )
    session = _record("session", stem, participant, end, payload)
    after = _record(
        "heartbeat", stem + "-heartbeat-after", participant, end + timedelta(seconds=20),
        {"event_id": stem + "-heartbeat-after", "participant_id": participant, "occurred_at": _iso(end + timedelta(seconds=20)), "source": "sample", "logging_enabled": True, "questions_enabled": False, "interventions_enabled": False},
    )
    return [before, session, after]


def _fresh_session(participant, anchor):
    day = {"date": anchor.date().isoformat(), "tags": ["none"], "events": []}
    center = anchor
    return _session(day, participant, 0, "fresh", center, 99)


def load_calendar(path):
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"calendar schema_version must be {SCHEMA}")
    if value.get("timezone") != "Asia/Tokyo":
        raise ValueError("calendar timezone must be Asia/Tokyo")
    days = value.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("calendar days must be a non-empty list")
    dates = [d.get("date") for d in days]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("calendar dates must be unique and sorted")
    for day in days:
        unknown = set(day.get("tags", [])) - TAGS
        if unknown:
            raise ValueError(f"unknown calendar tags: {sorted(unknown)}")
        for event in day.get("events", []):
            if event.get("tag") not in TAGS or not event.get("start") or not event.get("end"):
                raise ValueError("invalid calendar event")
    return value, hashlib.sha256(raw).hexdigest()


def build_records(calendar, sample_participant="p001-sample", include_today=False, now=None):
    rng = np.random.default_rng(int(calendar["seed"]))
    rows = []
    previous_tags = set()
    for day_index, day in enumerate(calendar["days"]):
        dominant = next((tag for tag in ("lab", "work", "travel", "drinking", "none") if tag in day["tags"]), "none")
        for activity_index, hour in enumerate((8, 14)):
            shown = _at(day["date"], hour, 35)
            answered = shown + timedelta(seconds=25)
            identifier = f"sample-{day['date']}-activity-{activity_index}"
            payload = {
                "participant_id": sample_participant,
                "event_id": identifier,
                "detected_at": _iso(shown),
                "question_shown_at": _iso(shown),
                "answered_at": _iso(answered),
                "question": {"kind": "activityCheck", "prompt": "今、何をしていますか？"},
                "answer": {"label": ACTIVITY[dominant], "free_text": None},
                "source": "sample_generator",
                "event_type": "sample_activity",
                "provenance": MANIFEST_SCHEMA,
            }
            rows.append(_record("narrative", identifier, sample_participant, answered, payload))
        for assessment_index, (kind, hour, minute) in enumerate(ASSESSMENTS):
            shown = _at(day["date"], hour, minute)
            answered = shown + timedelta(seconds=45)
            period = "morning" if hour < 12 else "afternoon" if hour < 20 else "night"
            scores = _scores(rng, day, period, previous_tags)
            identifier = f"sample-{day['date']}-{kind}-{assessment_index}"
            payload = {
                "participant_id": sample_participant,
                "event_id": identifier,
                "detected_at": _iso(shown),
                "question_shown_at": _iso(shown),
                "answered_at": _iso(answered),
                "question": {
                    "kind": kind,
                    "prompt": "今の状態（5項目・各5段階）を教えてください" if kind == "currentState" else "今日全体の状態を教えてください",
                },
                "answer": {
                    "label": "current_state" if kind == "currentState" else "daily_reflection",
                    "free_text": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                },
                "source": "sample_generator",
                "event_type": "state_assessment",
                "provenance": MANIFEST_SCHEMA,
            }
            rows.extend(_session(day, sample_participant, assessment_index, "before", shown, day_index))
            rows.append(_record("narrative", identifier, sample_participant, answered, payload))
            rows.extend(_session(day, sample_participant, assessment_index, "after", answered, day_index))
        for sensor_index, (hour, name, state) in enumerate(((8, "玄関", "close"), (23, "寝室", "close"))):
            at = _at(day["date"], hour, 0)
            identifier = f"sample-{day['date']}-switchbot-{sensor_index}"
            rows.append(_record("switchbot_observation", identifier, sample_participant, at, {
                "state": state, "is_open": state != "close", "battery": 80 - day_index % 10,
                "device_id": f"sample-sensor-{sensor_index}", "name": name, "last_seen": _iso(at), "rssi": -55,
                "provenance": MANIFEST_SCHEMA,
            }))
        previous_tags = set(day["tags"])
    if include_today:
        anchor = (now or datetime.now(JST)).astimezone(JST).replace(second=0, microsecond=0)
        rows.extend(_fresh_session(sample_participant, anchor))
    return sorted(rows, key=lambda r: (r["at"], r["kind"], r["id"]))


def summarize(rows):
    by_kind = Counter(row["kind"] for row in rows)
    narrative = Counter(row["payload"].get("question", {}).get("kind") for row in rows if row["kind"] == "narrative")
    return {
        "total": len(rows),
        "by_kind": dict(sorted(by_kind.items())),
        "narrative": dict(sorted(narrative.items())),
    }


def generate_sample(
    calendar_path,
    sample_participant="p001-sample",
    source_participant="p001",
    store=None,
    manifest_path=None,
    dry_run=False,
    include_today=False,
    now=None,
):
    if sample_participant == source_participant:
        raise ValueError("sample participant must differ from source participant")
    if not sample_participant.endswith('-sample') or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', sample_participant):
        raise ValueError("sample participant must use a reserved *-sample ID")
    if store is not None:
        existing_rows = store.rows(sample_participant)
        if existing_rows and not Path(manifest_path or f"data/sample/{sample_participant}/manifest.json").exists():
            raise ValueError("sample destination already has records without a manifest")
    calendar, input_hash = load_calendar(calendar_path)
    if calendar.get("participant_id") != source_participant:
        raise ValueError("calendar participant_id does not match --participant")
    manifest_path = Path(manifest_path or f"data/sample/{sample_participant}/manifest.json")
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    anchor = None
    if include_today:
        if existing and existing.get("today_anchor"):
            anchor = datetime.fromisoformat(existing["today_anchor"])
        else:
            anchor = (now or datetime.now(JST)).astimezone(JST).replace(second=0, microsecond=0)
    expected = {
        "schema_version": MANIFEST_SCHEMA,
        "input_hash": input_hash,
        "seed": int(calendar["seed"]),
        "source_participant": source_participant,
        "sample_participant": sample_participant,
        "include_today": bool(include_today),
    }
    if existing and any(existing.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Existing sample manifest differs; refusing to modify sample data")
    rows = build_records(calendar, sample_participant, include_today, anchor)
    counts = summarize(rows)
    manifest = {
        **expected,
        "calendar": str(calendar_path),
        "days": len(calendar["days"]),
        "first_date": calendar["days"][0]["date"],
        "last_date": calendar["days"][-1]["date"],
        "today_anchor": _iso(anchor) if anchor else None,
        "counts": counts,
        "record_hash": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if dry_run:
        return {"status": "dry_run", "inserted": 0, "manifest": str(manifest_path), **manifest}
    if store is None:
        raise ValueError("store is required unless --dry-run is used")
    inserted = sum(
        store.save(row["kind"], row["id"], sample_participant, datetime.fromisoformat(row["at"]), row["payload"])
        for row in rows
    )
    _atomic_json(manifest_path, manifest)
    return {"status": "generated", "inserted": inserted, "existing": len(rows) - inserted, "manifest": str(manifest_path), **manifest}
