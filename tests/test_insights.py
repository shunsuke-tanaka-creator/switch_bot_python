from datetime import timedelta
import json
import math
import numpy as np
import pytest
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app
from research.data import Timeline, dt, F
from research.events import ResearchEvent
from research.baseline import conditional, weekday_trends
from research.topology import GrowingMap, typing_windows, behavior_map
from research.insights import InsightEngine, fatigue_change, typing_change, recommend
from research.questions import QuestionQueue, settings
from research.service import export_page, ingest_page


AT = dt("2026-10-29T16:40:00+09:00")


def record(kind, id, at, payload):
    return dict(
        kind=kind, id=id, at=at.isoformat(), payload=payload, participant="p001"
    )


def event(kind, id, at, payload):
    return record(
        "research_event",
        id,
        at,
        dict(
            schema_version="research-v2",
            event_id=id,
            participant_id="p001",
            occurred_at=at.isoformat(),
            kind=kind,
            payload=payload,
        ),
    )


def label(at, value=2, id=None):
    return record(
        "narrative",
        id or str(at),
        at + timedelta(seconds=1),
        dict(
            question=dict(kind="currentState"),
            question_shown_at=at.isoformat(),
            answer=dict(free_text=json.dumps({"fatigue": value})),
        ),
    )


def session(at, n=120, corrections=0, purpose=None, device="phone"):
    events = []
    for i in range(n):
        d = {"kana": "あ", "input_mode": "kana"}
        if purpose:
            d["purpose"] = purpose
        t = at + timedelta(seconds=i * 2)
        events.append(dict(type="kana", time=t.isoformat(), data=d))
        if i < corrections:
            events.append(
                dict(
                    type="delete",
                    time=(t + timedelta(seconds=1)).isoformat(),
                    data={
                        "target": "composition",
                        **({"purpose": purpose} if purpose else {}),
                    },
                )
            )
    return record(
        "session",
        str(at) + str(purpose),
        at + timedelta(minutes=5),
        dict(
            session_id=str(at) + str(purpose),
            device_id=device,
            started_at=at.isoformat(),
            ended_at=(at + timedelta(minutes=5)).isoformat(),
            events=events,
        ),
    )


def put(store, rows):
    for r in rows:
        store.save(r["kind"], r["id"], "p001", dt(r["at"]), r["payload"])


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("weather_mood", {"weather": "angry"}),
        ("place_visit", {"place_id": "place_A", "phase": "arrival", "accuracy_m": 40}),
        ("typing_reason", {"question_id": "q", "reason": "rewrite"}),
        (
            "intervention_outcome",
            {
                "question_id": "q",
                "action": "refresh",
                "executed": True,
                "helpfulness": "helpful",
            },
        ),
        ("question_state", {"question_id": "q", "state": "deferred"}),
        ("preferences", {"daily_limit": 4.0, "questions_enabled": False}),
    ],
)
def test_event_contract(kind, payload):
    obj = ResearchEvent(
        event_id="x", participant_id="p001", occurred_at=AT, kind=kind, payload=payload
    )
    assert obj.kind == kind


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("weather_mood", {"weather": "5"}),
        ("weather_mood", {"weather": "happy", "fatigue": 1}),
        (
            "place_visit",
            {"place_id": "p", "phase": "arrival", "accuracy_m": float("nan")},
        ),
        (
            "place_visit",
            {"place_id": "p", "phase": "arrival", "accuracy_m": 10, "latitude": 35},
        ),
        ("preferences", {"daily_limit": True}),
        ("preferences", {"gap_minutes": 0}),
        (
            "intervention_outcome",
            {
                "question_id": "q",
                "action": "refresh",
                "executed": False,
                "helpfulness": "helpful",
            },
        ),
    ],
)
def test_invalid_events_rejected(kind, payload):
    with pytest.raises(ValueError):
        ResearchEvent(
            event_id="x",
            participant_id="p001",
            occurred_at=AT,
            kind=kind,
            payload=payload,
        )


def test_causal_baseline_missing_labels_and_day_weighting():
    rows = [label(AT - timedelta(days=i), 2) for i in range(1, 7)]
    before = conditional(rows, AT)
    assert before["fatigue"]["mean"] == 2 and before["mood"]["mean"] is None
    delayed = label(AT - timedelta(days=2), 5, "late")
    delayed["at"] = (AT + timedelta(minutes=1)).isoformat()
    assert conditional(rows + [delayed], AT) == before
    many = [label(AT - timedelta(hours=1, seconds=i), 5, str(i)) for i in range(100)]
    assert conditional(many, AT)["fatigue"]["status"] == "insufficient_data"
    # The 100 reports on one date have one day's influence, not 100 votes.
    assert conditional(rows + many, AT)["fatigue"]["mean"] < 3


def test_growing_map_masking_capacity_restart_and_novelty():
    model = GrowingMap(radius=0.1, max_nodes=2)
    assert model.add("empty", AT, {"hour": 1}) is None
    a = model.add("a", AT, dict(a=0, b=0, c=0))
    assert model.add("b", AT + timedelta(days=1), dict(a=0.01, b=0, c=0)) == a
    b = model.add("c", AT + timedelta(days=2), dict(a=1, b=1, c=1))
    assert b != a
    assert model.add("over", AT + timedelta(days=3), dict(a=3, b=3, c=3)) is None
    out = model.export(AT + timedelta(days=4))
    assert len(out["nodes"]) == 2 and out["at_capacity"]
    windows = typing_windows([session(AT - timedelta(minutes=10))], AT)
    assert behavior_map(windows, AT) == behavior_map(list(windows), AT)


def test_typing_ignores_assessments_and_preserves_zero_vs_unknown():
    rows = [
        session(AT - timedelta(minutes=10), corrections=30),
        session(AT - timedelta(minutes=20), purpose="assessment"),
    ]
    windows = typing_windows(rows, AT)
    assert (
        len(windows) == 1
        and windows[0]["inserts"] == 120
        and windows[0]["deletes"] == 30
    )
    assert math.isclose(windows[0]["correction_rate"], 0.25)
    assert typing_windows([session(AT - timedelta(minutes=1))], AT) == []
    late = rows[0].copy()
    late["payload"] = dict(
        late["payload"], received_at=(AT + timedelta(seconds=1)).isoformat()
    )
    assert typing_windows([late], AT) == []
    # Adding weather does not alter any of the 98 observation features.
    sample = Timeline(rows).sample(AT)
    changed = Timeline(
        rows
        + [event("weather_mood", "w", AT - timedelta(seconds=1), {"weather": "angry"})]
    ).sample(AT)
    assert F == 98
    for k in sample:
        np.testing.assert_array_equal(sample[k], changed[k])


def typing_window(day, inserts, rate, *, offset=0, device="phone", mode="kana"):
    start = AT.replace(hour=10, minute=0) - timedelta(days=day) + timedelta(
        minutes=offset
    )
    return dict(
        id=f"{day}-{offset}-{device}-{mode}",
        at=start,
        end=start + timedelta(minutes=10),
        inserts=inserts,
        deletes=0,
        correction_rate=rate,
        delete_rate=0,
        speed=0,
        pause_rate=0,
        device=device,
        mode=mode,
    )


def test_typing_daily_pool_available_and_unusual():
    windows = [
        typing_window(day, 30, 0.01, offset=offset)
        for day in range(2, 7)
        for offset in (0, 10)
    ]
    windows += [
        typing_window(1, 30, 0.10, offset=0),
        typing_window(1, 30, 0.20, offset=10),
        typing_window(0, 20, 0.20, offset=0),
        typing_window(0, 40, 0.30, offset=10),
    ]
    result = typing_change(windows, AT)
    assert result["status"] == "available"
    assert result["unusual"]
    assert result["baseline_days"] == 6
    assert result["baseline_windows"] == 12
    assert result["inserts"] == 60
    assert result["correction_rate"] == pytest.approx(4 / 15)
    assert result["evidence_id"] == windows[-1]["id"]
    assert "provisional" not in result


def test_typing_daily_pool_provisional_with_three_or_four_days():
    for days in (3, 4):
        windows = [typing_window(day, 50, 0.01) for day in range(1, days + 1)]
        windows.append(typing_window(0, 50, 0.01))
        result = typing_change(windows, AT)
        assert result["status"] == "available"
        assert result["baseline_days"] == days
        assert result["provisional"] is True


def test_typing_daily_pool_needs_three_valid_days():
    windows = [typing_window(day, 50, 0.01) for day in (1, 2)]
    windows.append(typing_window(0, 50, 0.01))
    result = typing_change(windows, AT)
    assert result == {
        "status": "insufficient_data",
        "unusual": False,
        "baseline_windows": 2,
        "baseline_days": 2,
    }


def test_typing_daily_pool_filters_device_and_mode():
    windows = [typing_window(day, 50, 0.01) for day in range(1, 6)]
    windows += [
        typing_window(day, 500, 0.9, device="laptop")
        for day in range(1, 6)
    ]
    windows += [
        typing_window(day, 500, 0.9, mode="romaji") for day in range(1, 6)
    ]
    windows += [
        typing_window(0, 500, 0.9, device="laptop"),
        typing_window(0, 500, 0.9, mode="romaji"),
        typing_window(0, 50, 0.01),
    ]
    result = typing_change(windows, AT)
    assert result["baseline_windows"] == 5
    assert result["inserts"] == 50
    assert result["correction_rate"] == pytest.approx(0.01)


def test_typing_daily_pool_rejects_stale_latest_window():
    windows = [typing_window(day, 50, 0.01) for day in range(6, 0, -1)]
    result = typing_change(windows, AT, fresh_seconds=60)
    assert result["status"] == "insufficient_data"


def test_typing_daily_pool_requires_current_day_minimum():
    windows = [typing_window(day, 50, 0.01) for day in range(5, 0, -1)]
    windows += [typing_window(0, 20, 0.1), typing_window(0, 29, 0.1, offset=10)]
    result = typing_change(windows, AT)
    assert result["status"] == "insufficient_data"
    assert result["inserts"] == 49


def test_typing_daily_unusual_requires_current_and_previous_valid_day():
    baseline = [typing_window(day, 50, 0.01) for day in range(2, 7)]
    current_only = baseline + [
        typing_window(1, 50, 0.01),
        typing_window(0, 50, 0.2),
    ]
    assert not typing_change(current_only, AT)["unusual"]
    two_days = baseline + [
        typing_window(1, 50, 0.2),
        typing_window(0, 50, 0.2),
    ]
    assert typing_change(two_days, AT)["unusual"]


def test_fatigue_persistence_requires_coverage_and_separate_five_minute_bins():
    t = AT.replace(minute=40)
    rows = [label(t - timedelta(days=i), 2) for i in range(1, 7)]
    prediction = dict(status="ready", current={"fatigue": {"value": 4.5}})
    for offset in (9, 4):
        time = t - timedelta(minutes=offset)
        rows.append(record("inference", "i" + str(offset), time, prediction))
        rows.append(
            record("switchbot_observation", "s" + str(offset), time, {"state": "close"})
        )
    base = conditional(rows, t)
    assert fatigue_change(rows, t, base, prediction)["acute"]
    assert not fatigue_change(
        [r for r in rows if r["kind"] != "switchbot_observation"], t, base, prediction
    )["acute"]


def test_trends_need_weeks_and_have_day_counts():
    t = dt("2026-10-29T16:00:00+09:00")
    start = dt("2026-08-03T14:00:00+09:00")
    rows = [label(start + timedelta(days=i), 4 if i % 7 == 0 else 2) for i in range(77)]
    result = weekday_trends(rows, t)
    monday = next(x for x in result if x["weekday"] == 0 and x["period"] == "afternoon")
    assert monday["status"] == "higher" and monday["target_days"] == 10
    assert weekday_trends(rows[-20:], t) == []


def test_snapshot_restart_and_no_fabricated_prediction(tmp_path):
    store = Store(":memory:")
    put(
        store,
        [
            event(
                "weather_mood",
                "weather",
                AT - timedelta(minutes=1),
                dict(weather="happy"),
            ),
            session(AT - timedelta(minutes=10)),
        ],
    )
    one = InsightEngine(store, {}).snapshot("p001", AT)
    assert (
        one["fatigue"]["status"] == "insufficient_data"
        and one["weather"][0]["weather"] == "happy"
    )
    assert one == InsightEngine(store, {}).snapshot("p001", AT)
    target = Store(":memory:")
    ingest_page(target, export_page(store, "p001"))
    assert InsightEngine(target, {}).snapshot("p001", AT) == one
    store.close()
    target.close()


def test_queue_budget_weather_contract_and_preference_stop(tmp_path):
    store = Store(":memory:")
    queue = QuestionQueue(store, InsightEngine(store, {}))
    t = AT.replace(hour=9, minute=0)
    q = queue.pending("p001", t)
    assert q["kind"] == "weatherMood" and len(q["options"]) == 4
    assert queue.pending("p001", t + timedelta(minutes=1))["id"] == q["id"]
    put(
        store,
        [
            event(
                "question_state", "shown", t, dict(question_id=q["id"], state="shown")
            ),
            event(
                "weather_mood",
                "answer",
                t + timedelta(minutes=1),
                dict(question_id=q["id"], weather="angry"),
            ),
        ],
    )
    assert queue.pending("p001", t + timedelta(minutes=30)) is None
    q2 = queue.pending("p001", t + timedelta(hours=1))
    assert q2["kind"] == "currentState"
    put(
        store,
        [
            event(
                "question_state",
                "shown2",
                t + timedelta(hours=1),
                dict(question_id=q2["id"], state="shown"),
            ),
            event("preferences", "prefs", t + timedelta(hours=1), dict(daily_limit=2)),
        ],
    )
    assert queue.pending("p001", t + timedelta(hours=7)) is None
    put(
        store,
        [
            event(
                "preferences",
                "off",
                t + timedelta(hours=8),
                dict(questions_enabled=False),
            )
        ],
    )
    assert queue.pending("p001", t + timedelta(hours=9)) is None
    pref = settings(
        [
            event("preferences", "x", t, dict(interventions_enabled=False)),
            record(
                "heartbeat",
                "h",
                t + timedelta(minutes=1),
                dict(interventions_enabled=True),
            ),
        ]
    )
    assert pref["interventions_enabled"] is False
    store.close()


def test_api_validation_idempotence_and_complete_export(tmp_path):
    store = Store(":memory:")
    app = create_app(
        dict(
            ble_enabled=False,
            allowed_clients=["testclient"],
            research_models=str(tmp_path),
        ),
        store,
    )
    with TestClient(app) as c:
        body = dict(
            event_id="weather",
            participant_id="p001",
            occurred_at=AT.isoformat(),
            kind="weather_mood",
            payload=dict(weather="sad"),
        )
        assert c.post("/api/research/p001/events", json=body).json()["inserted"]
        assert not c.post("/api/research/p001/events", json=body).json()["inserted"]
        assert c.post("/api/research/p002/events", json=body).status_code == 422
        assert (
            c.post(
                "/api/research/p001/events",
                json={
                    **body,
                    "event_id": "bad",
                    "payload": dict(weather="sad", fatigue=5),
                },
            ).status_code
            == 422
        )
        unknown = dict(
            event_id="q",
            participant_id="p001",
            occurred_at=AT.isoformat(),
            kind="question_state",
            payload=dict(question_id="unknown", state="shown"),
        )
        assert c.post("/api/research/p001/events", json=unknown).status_code == 422
        assert (
            c.get("/api/research/p001/insights").json()["weather"][0]["weather"]
            == "sad"
        )
        assert c.get("/research/p001/insights").status_code == 200
        assert (
            c.get("/api/export/p001").json()["records"][0]["kind"] == "research_event"
        )


def test_no_success_memory_from_acceptance_only():
    fatigue = dict(value=4, difference=1)
    typing = dict(unusual=False)
    rows = [
        record(
            "narrative",
            "click",
            AT - timedelta(hours=1),
            dict(answer=dict(label="accept:refresh")),
        )
    ]
    assert all(
        x["based_on_event_id"] is None for x in recommend(rows, AT, fatigue, typing)
    )


@pytest.mark.parametrize(
    "reason,expected", [("mistype", "breakCheck"), ("rewrite", "weatherMood")]
)
def test_typing_reason_precedes_typing_only_intervention(reason, expected):
    class Fixed:
        def snapshot(self, p, at):
            return dict(
                fatigue={},
                typing=dict(unusual=True, evidence_id="window"),
                places=dict(anomaly=None, time_anomaly=None),
            )

    store = Store(":memory:")
    queue = QuestionQueue(store, Fixed())
    t = AT.replace(hour=9, minute=0)
    put(store, [event("preferences", "prefs", t - timedelta(days=1), dict(daily_limit=6))])
    question = queue.pending("p001", t)
    assert question["kind"] == "typingReason"
    put(
        store,
        [
            event(
                "typing_reason",
                "reason",
                t + timedelta(minutes=1),
                dict(question_id=question["id"], reason=reason),
            )
        ],
    )
    assert queue.pending("p001", t + timedelta(minutes=61))["kind"] == expected
    store.close()


def test_model_refresh_is_not_fatigue_persistence():
    t = AT
    rows = [label(t - timedelta(days=i), 2) for i in range(1, 7)]
    prediction = dict(status="ready", current={"fatigue": {"value": 4.5}})
    for offset, version in [(9, "old"), (4, "new")]:
        time = t - timedelta(minutes=offset)
        rows.append(
            record(
                "inference",
                "i" + str(offset),
                time,
                dict(prediction, model={"fingerprint": version}),
            )
        )
        rows.append(
            record("switchbot_observation", "s" + str(offset), time, {"state": "close"})
        )
    assert not fatigue_change(rows, t, conditional(rows, t), prediction)["acute"]


def test_missing_checkpoint_allows_retry_even_without_new_state_labels(tmp_path):
    from research.service import Engine

    store = Store(":memory:")
    engine = Engine(store, dict(research_models=str(tmp_path)))
    assert engine.train("p001")["status"] == "insufficient_labels"
    put(
        store,
        [
            event("weather_mood", "weather", AT, dict(weather="happy")),
            record(
                "narrative",
                "weather-answer",
                AT,
                dict(question={"kind": "weatherMood"}, answer={"label": "happy"}),
            ),
        ],
    )
    assert engine.train("p001")["status"] == "insufficient_labels"
    store.close()
