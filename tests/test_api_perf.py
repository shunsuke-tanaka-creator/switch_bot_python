"""API equality and freshness for the shared research computation cache."""

from datetime import timedelta
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from lifelog_server import Store, create_app
from research.data import SCALES, dt
from research.data_modes import rows_for_mode
from research.questions import settings
from research.sample_registry import activate_sample, record_hash, register_generation
from research.service import pmtt_projection


AT = dt("2026-11-02T16:40:00+09:00")


@pytest.fixture(params=["legacy", "discovery"])
def api(request, tmp_path, monkeypatch):
    monkeypatch.setattr("research.service.utc", lambda: AT)
    store = Store(tmp_path / "research.sqlite3")
    architecture = request.param
    for participant in ("p001", "p001-sample"):
        for day in range(9):
            at = AT - timedelta(days=day, minutes=20)
            store.save("session", f"s-{day}", participant, at, dict(
                started_at=(at - timedelta(minutes=5)).isoformat(),
                ended_at=at.isoformat(), session_id=f"s-{day}", device_id="phone",
                events=[dict(type="kana", time=(at - timedelta(seconds=i)).isoformat(),
                             data={"input_mode": "kana"}) for i in range(30)],
            ))
            store.save("narrative", f"a-{day}", participant, at, dict(
                question_shown_at=(at - timedelta(minutes=1)).isoformat(),
                question={"kind": "currentState"},
                answer={"free_text": json.dumps(dict.fromkeys(SCALES, 2 + day % 3))},
            ))
            store.save("context_inference", f"c-{day}", participant, at, dict(
                at=at.isoformat(), architecture=architecture,
                place={"id": "place"}, task={"id": "task"}, model={"bulk": [1, 2]},
            ))
        for axis, identity in (("place", "place"), ("task", "task")):
            store.save("context_candidate", identity, participant, AT - timedelta(days=10),
                       dict(axis=axis, class_id=identity, anchor=AT.isoformat()))
        store.save("context_inference", "current", participant, AT, dict(
            at=AT.isoformat(), architecture=architecture,
            place={"id": "place"}, task={"id": "task"}, model={"bulk": [3, 4]},
        ))
        store.save("context_review", "review", participant, AT, dict(
            day="2026-11-01", architecture=architecture,
            episodes=[], original_live=[{"unused": True}], edges={},
        ))
        store.save("inference", "current", participant, AT, dict(
            at=AT.isoformat(), architecture=architecture, data_mode="real",
            current={"fatigue": {"value": 3}}, daily=None,
        ))
    app = create_app(dict(
        ble_enabled=False, allowed_clients=["testclient"],
        research_models=str(tmp_path / "models"), research_architecture=architecture,
    ), store)
    with TestClient(app) as client:
        yield client, app.state.research, store


def reset_insights(engine):
    for name in ("cache", "trend_cache"):
        cache = getattr(engine.insights, name, None)
        if cache is not None:
            cache.clear()


@pytest.mark.parametrize("mode", ["real", "sample", "combined"])
def test_all_endpoint_payloads_equal_with_and_without_shared_cache(api, mode):
    client, engine, store = api
    paths = [f"/api/research/p001/{name}?data_mode={mode}"
             for name in ("overview", "insights", "pmtt")]
    paths.append("/api/research/p001/context")
    engine.api_cache.enabled = False
    uncached = []
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200
        uncached.append(response.json())
    engine.api_cache.enabled = True
    reset_insights(engine)
    for _ in range(2):
        for path, expected in zip(paths, uncached):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json() == expected
    # Independent full-record paths also preserve projections and ordering.
    assert uncached[-1] == engine.context.snapshot("p001", AT, store.rows("p001"))
    records = rows_for_mode(store, "p001", mode, engine.sample_participant)
    assert uncached[2] == pmtt_projection(engine._pmtt("p001", mode, records, AT))
    assert uncached[1]["preferences"] == settings(store.rows("p001"))


def test_answer_and_context_name_invalidate_their_computations(api, monkeypatch):
    client, engine, store = api
    snapshot = Mock(wraps=engine.context.snapshot)
    graph = Mock(wraps=engine._pmtt)
    monkeypatch.setattr(engine.context, "snapshot", snapshot)
    monkeypatch.setattr(engine, "_pmtt", graph)
    before = {name: client.get(f"/api/research/p001/{name}").json()
              for name in ("overview", "insights", "context", "pmtt")}
    store.save("narrative", "new-answer", "p001", AT - timedelta(seconds=1), dict(
        question_shown_at=(AT - timedelta(seconds=30)).isoformat(),
        question={"kind": "currentState"},
        answer={"free_text": json.dumps(dict.fromkeys(SCALES, 5))},
    ))
    after = client.get("/api/research/p001/overview").json()
    assert len(after["answers"]) == len(before["overview"]["answers"]) + 1
    assert client.get("/api/research/p001/insights").json()["baseline"] != before["insights"]["baseline"]
    assert client.get("/api/research/p001/pmtt").status_code == 200
    assert graph.call_count == 2
    store.save("research_event", "name", "p001", AT, dict(
        kind="context_name", payload=dict(axis="place", class_id="place", name="Desk"),
    ))
    context = client.get("/api/research/p001/context").json()
    assert context != before["context"]
    assert context["nodes"]["place"][0]["name"] == "Desk"
    assert snapshot.call_count == 2


def test_heartbeat_keeps_expensive_artifacts_and_refreshes_preferences(api, monkeypatch):
    client, engine, store = api
    snapshot = Mock(wraps=engine.context.snapshot)
    graph = Mock(wraps=engine._pmtt)
    monkeypatch.setattr(engine.context, "snapshot", snapshot)
    monkeypatch.setattr(engine, "_pmtt", graph)
    first = client.get("/api/research/p001/insights").json()
    context = client.get("/api/research/p001/context").json()
    pmtt = client.get("/api/research/p001/pmtt").json()
    trend_cache = dict(engine.insights.trend_cache)
    store.save("heartbeat", "heartbeat", "p001", AT,
               dict(logging_enabled=True, questions_enabled=False, interventions_enabled=False))
    second = client.get("/api/research/p001/insights").json()
    assert second["preferences"]["questions_enabled"] is False
    assert second["preferences"]["interventions_enabled"] is False
    assert {k: v for k, v in first.items() if k != "preferences"} == {
        k: v for k, v in second.items() if k != "preferences"
    }
    assert client.get("/api/research/p001/context").json() == context
    assert client.get("/api/research/p001/pmtt").json() == pmtt
    assert snapshot.call_count == graph.call_count == 1
    assert engine.insights.trend_cache == trend_cache


def test_combined_pmtt_preserves_group_order_after_heartbeat(api):
    client, engine, store = api
    path = "/api/research/p001/pmtt?data_mode=combined"
    before = client.get(path).json()
    # Source group order originally comes from all records, including heartbeats.
    store.save("heartbeat", "newest-real", "p001", AT + timedelta(seconds=1), {})
    records = rows_for_mode(store, "p001", "combined", engine.sample_participant)
    expected = pmtt_projection(engine._pmtt("p001", "combined", records, AT))
    after = client.get(path).json()
    assert after == expected
    assert before != after


def test_sample_activation_refreshes_selection_and_preview_time(api):
    client, engine, store = api
    path = "/api/research/p001/overview?data_mode=sample"
    before = client.get(path).json()
    sample_id = "p001-g" + "0" * 20 + "-sample"
    rows = [dict(r, participant=sample_id) for r in store.rows("p001-sample")]
    preview = AT + timedelta(minutes=1)
    manifest = dict(
        source_participant="p001", sample_participant=sample_id,
        record_hash=record_hash(rows), preview_at=preview.isoformat(),
    )
    register_generation(store, rows, manifest, {}, [])
    activate_sample(store, "p001", sample_id)
    after = client.get(path).json()
    assert before["model"]["sample_generation"] == "p001-sample"
    assert after["model"]["sample_generation"] == sample_id
    assert {answer["participant"] for answer in after["answers"]} == {sample_id}
    assert dt(after["prediction"]["at"]) == preview
    assert engine.rows("p001", "sample") == store.rows(sample_id)


def test_context_expiry_and_model_status_changes_remain_live(api, monkeypatch):
    client, engine, _ = api
    assert client.get("/api/research/p001/context").json()["current"]["at"] == AT.isoformat()
    monkeypatch.setattr("research.service.utc", lambda: AT + timedelta(seconds=181))
    context = client.get("/api/research/p001/context").json()
    assert context["current"]["status"] == "worker_not_running_or_stale"
    path = engine.context.directory("p001") / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="trained", marker="changed")))
    assert client.get("/api/research/p001/context").json()["training"] == {
        "status": "trained", "marker": "changed",
    }
