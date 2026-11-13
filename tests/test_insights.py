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
from research.insights import (
    InsightEngine,
    exercise_candidates,
    fatigue_change,
    recommend,
    step_shortage,
    typing_change,
)
from research.questions import QuestionQueue, settings
from research.service import Engine, export_page, ingest_page


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


# 追加: 参加者の入力原文を含まない、日累積歩数だけの合成セッション。
def step_session(
    at,
    steps,
    *,
    place_id=None,
    measurement_at=None,
    identifier=None,
    participant="p001",
):
    measured = measurement_at or at - timedelta(minutes=1)
    context = dict(
        step_count=steps,
        measurement_updated_at=dict(step_count=measured.isoformat()),
    )
    if place_id:
        context["place_id"] = place_id
    result = record(
        "session",
        identifier or f"steps-{at.isoformat()}-{place_id}",
        at,
        dict(
            session_id=identifier or f"steps-{at.isoformat()}-{place_id}",
            device_id="synthetic-phone",
            started_at=(at - timedelta(minutes=2)).isoformat(),
            ended_at=at.isoformat(),
            events=[],
            context=context,
        ),
    )
    # 追加: 複数参加者を混在させた基準分離テストに使う。
    result["participant"] = participant
    return result


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
        ("preferences", {"exercise_proposals_enabled": True}),
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


# 追加: 100歩/分とMETs比を使い、候補は最大4件を5分単位で切り上げる。
def test_exercise_candidates_use_mets_and_five_minute_rounding():
    candidates = exercise_candidates(3000)
    assert len(candidates) == 4
    assert [(x["action"], x["mets"], x["minutes"]) for x in candidates] == [
        ("walking", 3.0, 30),
        ("brisk_walking", 4.3, 25),
        ("stair_climbing", 8.0, 15),
        ("cycling", 6.8, 15),
    ]


# 追加: 同曜日・同拠点・同時刻までの分布だけで歩数不足を判定する。
def test_step_shortage_uses_weekday_place_time_context():
    at = dt("2026-11-09T16:00:00+09:00")
    rows = [
        step_session(
            at - timedelta(days=7 * week, minutes=5),
            4000 + week * 20,
            place_id="place_lab",
        )
        for week in range(1, 6)
    ]
    rows += [
        step_session(
            at - timedelta(days=7 * week, minutes=4),
            700,
            place_id="place_home",
            identifier=f"other-{week}",
        )
        for week in range(1, 6)
    ]
    # 追加: 当日の歩数増加で端末未携帯との区別に必要な観測を作る。
    rows.append(
        step_session(
            at - timedelta(hours=1),
            800,
            place_id="place_lab",
            identifier="today-progress",
        )
    )
    rows.append(step_session(at, 1000, place_id="place_lab", identifier="today"))
    result = step_shortage(rows, at)
    assert result["status"] == "available" and result["low"]
    assert result["context"]["basis"] == "weekday_place_time"
    assert result["context"]["place_id"] == "place_lab"
    assert result["baseline_days"] == 5
    assert result["usual_steps"] == 4060
    assert result["shortage_steps"] == 3060
    assert "provisional" in result and result["provisional"] is False


# 追加: 拠点不明時だけ曜日×時刻へフォールバックする。
def test_step_shortage_falls_back_only_when_current_place_is_unknown():
    at = dt("2026-11-09T16:00:00+09:00")
    history = [
        step_session(
            at - timedelta(days=7 * week, minutes=5),
            3000,
            place_id="place_home" if week % 2 else "place_lab",
        )
        for week in range(1, 6)
    ]
    fallback = step_shortage(
        history
        + [
            step_session(
                at - timedelta(hours=1), 800, identifier="fallback-progress"
            ),
            step_session(at, 1000, identifier="today-unknown"),
        ],
        at,
    )
    assert fallback["status"] == "available" and fallback["low"]
    assert fallback["context"]["basis"] == "weekday_time_fallback"
    known = step_shortage(
        history
        + [
            step_session(
                at - timedelta(hours=1),
                800,
                place_id="place_lab",
                identifier="known-progress",
            ),
            step_session(
                at, 1000, place_id="place_lab", identifier="today-known"
            ),
        ],
        at,
    )
    assert known["status"] == "insufficient_data"
    assert known["reason"] == "context_baseline_too_short"
    assert known["baseline_days"] == 2


# 追加: 古い歩数値は通常値にも当日判定にも使用しない。
def test_step_shortage_rejects_stale_measurements():
    at = dt("2026-11-09T16:00:00+09:00")
    rows = [
        step_session(
            at - timedelta(days=7 * week),
            3000,
            measurement_at=at - timedelta(days=7 * week, hours=2),
        )
        for week in range(1, 6)
    ]
    rows.append(
        step_session(at, 1000, measurement_at=at - timedelta(hours=2))
    )
    result = step_shortage(rows, at)
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "current_step_measurement_missing_or_stale"


# 追加: 当日の歩数増加がなく端末未携帯と区別できない場合は提案判定しない。
def test_step_shortage_requires_same_day_step_progress():
    at = dt("2026-11-09T16:00:00+09:00")
    history = [
        step_session(
            at - timedelta(days=7 * week, minutes=5),
            3000,
            place_id="place_lab",
        )
        for week in range(1, 6)
    ]
    result = step_shortage(
        history
        + [step_session(at, 1000, place_id="place_lab", identifier="today")],
        at,
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "device_carry_not_observed"


# 追加: combined表示でも別参加者の歩数を本人の通常値へ混ぜない。
def test_step_shortage_does_not_mix_participants():
    at = dt("2026-11-09T16:00:00+09:00")
    other_history = [
        step_session(
            at - timedelta(days=7 * week, minutes=5),
            4000,
            place_id="place_lab",
            participant="p002",
        )
        for week in range(1, 6)
    ]
    result = step_shortage(
        other_history
        + [
            step_session(
                at - timedelta(hours=1),
                800,
                place_id="place_lab",
                identifier="progress",
            ),
            step_session(at, 1000, place_id="place_lab", identifier="today"),
        ],
        at,
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "context_baseline_too_short"
    assert result["baseline_days"] == 0


# 追加: セッションに拠点がなくても、同時刻のContextLSTM確定拠点を条件に使う。
def test_step_shortage_uses_context_inference_place():
    at = dt("2026-11-09T16:00:00+09:00")
    rows = []
    for week in range(1, 6):
        observed = at - timedelta(days=7 * week, minutes=5)
        rows.extend(
            [
                step_session(observed, 3000, identifier=f"history-{week}"),
                record(
                    "context_inference",
                    f"context-{week}",
                    observed,
                    dict(place=dict(id="place_lab", name="研究拠点")),
                ),
            ]
        )
    rows.extend(
        [
            step_session(
                at - timedelta(hours=1), 800, identifier="today-progress"
            ),
            step_session(at, 1000, identifier="today"),
            record(
                "context_inference",
                "context-today",
                at,
                dict(place=dict(id="place_lab", name="研究拠点")),
            ),
        ]
    )
    result = step_shortage(rows, at)
    assert result["status"] == "available" and result["low"]
    assert result["context"]["basis"] == "weekday_place_time"
    assert result["context"]["place_id"] == "place_lab"
    assert result["context"]["place_name"] == "研究拠点"


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


# 追加: 運動提案は既定無効で、有効時も同日同文脈では一度だけ発行する。
def test_exercise_proposal_preference_pmtt_and_same_context_cooldown(tmp_path):
    exercise = dict(
        status="available",
        low=True,
        provisional=False,
        baseline_stage="formal",
        baseline_days=5,
        context=dict(
            basis="weekday_place_time",
            weekday=0,
            place_id="place_lab",
            place_name="研究室",
            time="09:00",
        ),
        usual_steps=4000,
        lower_quartile_steps=3900,
        today_steps=1000,
        shortage_steps=3000,
        minimum_shortage_steps=500,
        minimum_shortage_fraction=0.25,
        cadence_steps_per_minute=100,
        walking_equivalent_minutes=30.0,
        candidates=exercise_candidates(3000),
        evidence_id="synthetic-today",
        carry_evidence_id="synthetic-progress",
    )

    class Fixed:
        def snapshot(self, p, at):
            return dict(fatigue={}, typing={}, exercise=exercise)

    t = dt("2026-11-09T09:00:00+09:00")
    store = Store(":memory:")
    queue = QuestionQueue(store, Fixed())
    assert queue.pending("p001", t)["kind"] == "weatherMood"
    store.close()

    store = Store(":memory:")
    queue = QuestionQueue(store, Fixed())
    put(
        store,
        [
            event(
                "preferences",
                "exercise-on",
                t - timedelta(days=1),
                dict(exercise_proposals_enabled=True, daily_limit=6),
            )
        ],
    )
    question = queue.pending("p001", t)
    assert question["kind"] == "exerciseProposal"
    assert "100歩/分" in question["prompt"]
    assert question["pmtt"]["condition"]["place_id"] == "place_lab"
    assert question["pmtt"]["cadence_steps_per_minute"] == 100
    assert len(question["pmtt"]["candidates"]) == 4
    put(
        store,
        [
            record(
                "narrative",
                question["id"],
                t + timedelta(minutes=1),
                dict(
                    question=dict(kind="exerciseProposal"),
                    answer=dict(label="declined"),
                ),
            )
        ],
    )
    followup = queue.pending("p001", t + timedelta(minutes=61))
    assert followup["kind"] != "exerciseProposal"
    graph = Engine(store, dict(research_models=str(tmp_path))).pmtt("p001")
    proposal = next(
        node for node in graph["nodes"] if node["kind"] == "ExerciseProposal"
    )
    assert proposal["evidence"]["usual_steps"] == 4000
    assert proposal["evidence"]["candidates"][0]["mets"] == 3.0
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
