"""Deterministic artificial cohort, with no participant text or external data."""
from datetime import timedelta
import json
from .data import SCALES, dt


def sample_cohort(days=10):
    rows = []
    start = dt('2026-09-07T00:00:00+09:00')
    for person, shift in [('sample-a', 0), ('sample-b', 1)]:
        for day in range(days):
            for hour in (10, 16, 22):
                at = start + timedelta(days=day, hours=hour)
                prefix = f'{day}-{hour}'  # Deliberately shared IDs test participant isolation.
                def row(kind, suffix, time, payload):
                    rows.append(dict(participant=person, id=prefix+suffix, kind=kind,
                                     at=time.isoformat(), seq=len(rows)+1, payload=payload))
                # Continuous small observation window includes future SSL targets.
                for minute in range(-10, 36, 5):
                    time = at + timedelta(minutes=minute)
                    row('heartbeat', f'-heart-{minute}', time, dict(logging_enabled=True))
                received = at + timedelta(seconds=30)
                row('narrative', '-answer', received, dict(
                    participant_id=person, question_shown_at=at.isoformat(), received_at=received.isoformat(),
                    question=dict(kind='nightReflection' if hour == 22 else 'currentState'),
                    answer=dict(free_text=json.dumps(dict.fromkeys(SCALES, 2+(day+shift) % 3)))))
    return rows
