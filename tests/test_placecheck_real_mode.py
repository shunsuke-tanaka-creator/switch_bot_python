"""Regression tests: placeCheck/taskCheck must be selectable in real (pending) mode.

Production failure (diagnosed 2026-11-18): the context check was never issued
in real mode. While a routine question was unexpired, already_pending returned
it; by the time it expired (60m TTL), the last keyboard session was more than
1h old, so the shared 1h fresh() window had closed and a routine question was
issued instead. The selectable window was only a few minutes wide.

The fix widens the *selection-only* presence window to a same-day session that
ended within 3h (research.questions.context_session_fresh). The shared
context_data.fresh (1h) used by inference/training is unchanged.
"""

from datetime import timedelta

from lifelog_server import Store
from research.data import dt
from research.service import Engine
from research.context_data import fresh


def session(store, key, end, received_delay_sec=90):
    payload = dict(
        started_at=(end - timedelta(minutes=3)).isoformat(),
        ended_at=end.isoformat(),
        received_at=(end + timedelta(seconds=received_delay_sec)).isoformat(),
        events=[
            dict(
                type="commit",
                time=(end - timedelta(seconds=5)).isoformat(),
                data=dict(text="synthetic"),
            )
        ],
        context=dict(calendar_events=[], measurement_updated_at={}),
    )
    store.save("session", key, "p001", end, payload)


def make_engine(tmp_path):
    store = Store(":memory:")
    return store, Engine(store, dict(research_models=str(tmp_path)))


def test_context_selected_after_routine_expiry_with_2h_old_session(tmp_path):
    store, engine = make_engine(tmp_path)
    try:
        # 08:00 fetch before any session today: routine weatherMood is issued.
        q = engine.pending("p001", dt("2026-11-17T08:00:00+09:00"))
        assert q["kind"] == "weatherMood"
        # A keyboard session ends at 09:50; at 12:00 it is >1h old, so the
        # shared inference window is closed, but selection must still ask.
        session(store, "s1", dt("2026-11-17T09:50:00+09:00"))
        rows = store.rows("p001")
        assert not fresh(rows, dt("2026-11-17T12:00:00+09:00"))
        q = engine.pending("p001", dt("2026-11-17T12:00:00+09:00"))
        assert q["kind"] == "placeCheck"
        assert q["eventType"] == "context:place"
        # TTL contract is unchanged: 30 minutes from detectedAt.
        assert dt(q["expiresAt"]) - dt(q["detectedAt"]) == timedelta(minutes=30)
        issued = [
            r for r in store.rows("p001") if r["kind"] == "question_issued"
        ]
        assert sorted(r["payload"]["kind"] for r in issued) == [
            "placeCheck",
            "weatherMood",
        ]
        # Same day after the placeCheck expires: no second context question.
        q = engine.pending("p001", dt("2026-11-17T14:00:00+09:00"))
        assert q is None or q["kind"] not in ("placeCheck", "taskCheck")
        # Next day the axis alternates to taskCheck.
        session(store, "s2", dt("2026-11-18T09:50:00+09:00"))
        q = engine.pending("p001", dt("2026-11-18T12:00:00+09:00"))
        assert q["kind"] == "taskCheck"
    finally:
        store.close()


def test_context_not_selected_when_session_older_than_3h(tmp_path):
    store, engine = make_engine(tmp_path)
    try:
        session(store, "s1", dt("2026-11-17T08:30:00+09:00"))
        q = engine.pending("p001", dt("2026-11-17T12:00:00+09:00"))
        assert q is None or q["kind"] not in ("placeCheck", "taskCheck")
    finally:
        store.close()


def test_context_not_selected_from_previous_day_session(tmp_path):
    store, engine = make_engine(tmp_path)
    try:
        session(store, "s1", dt("2026-11-16T21:30:00+09:00"))
        q = engine.pending("p001", dt("2026-11-17T09:00:00+09:00"))
        assert q is None or q["kind"] not in ("placeCheck", "taskCheck")
    finally:
        store.close()


def test_context_still_selected_within_1h(tmp_path):
    store, engine = make_engine(tmp_path)
    try:
        session(store, "s1", dt("2026-11-17T15:00:00+09:00"))
        q = engine.pending("p001", dt("2026-11-17T15:15:00+09:00"))
        assert q["kind"] == "placeCheck"
    finally:
        store.close()


def test_preview_with_widened_window_does_not_persist(tmp_path):
    store, engine = make_engine(tmp_path)
    try:
        session(store, "s1", dt("2026-11-17T09:50:00+09:00"))
        preview = engine.proposal_preview(
            "p001", "real", dt("2026-11-17T12:00:00+09:00")
        )
        assert preview["question"]["kind"] == "placeCheck"
        assert not [
            r for r in store.rows("p001") if r["kind"] == "question_issued"
        ]
    finally:
        store.close()
