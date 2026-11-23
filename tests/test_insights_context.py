"""Project context display details without changing stored records or insights."""

from datetime import timedelta
import json

import pytest
from fastapi.testclient import TestClient

from lifelog_server import Store, create_app
from research.context_data import transitions
from research.data import dt
from research.questions import settings


AT = dt("2026-10-29T16:40:00+09:00")
BULK_KEYS = {"reviews", "history"}


@pytest.fixture(params=["legacy", "discovery"])
def context_client(request, tmp_path, monkeypatch):
    monkeypatch.setattr("research.service.utc", lambda: AT)
    architecture = request.param
    store = Store(tmp_path / "context.sqlite3")
    # Large nested vectors make even the existing 7-review/288-inference limits
    # exceed the insights budget. All records and features here are synthetic.
    features = [0.125] * 512
    model = dict(
        status="trained",
        trained_through=(AT - timedelta(days=31)).isoformat(),
        validation=dict(
            place=dict(n=200, accuracy=0.8),
            task=dict(n=200, accuracy=0.7),
        ),
        training_loss=features,
    )

    def inference(at, index):
        return dict(
            at=at.isoformat(),
            architecture=architecture,
            status="ready",
            place=dict(id=f"place_{index % 2}", status="estimated"),
            task=dict(id=f"task_{index % 3}", status="estimated"),
            feature_vector=features,
            model=model,
        )

    for index in range(320):
        at = AT - timedelta(minutes=5 * index)
        store.save(
            "context_inference", f"inference-{index}", "p001", at,
            inference(at, index),
        )
    for index in range(30):
        at = AT - timedelta(days=index + 1)
        episodes = [
            inference(at - timedelta(minutes=5 * step), step)
            for step in range(48)
        ]
        store.save(
            "context_review", f"review-{index}", "p001", at,
            dict(
                day=at.date().isoformat(),
                created_at=at.isoformat(),
                architecture=architecture,
                episodes=episodes,
                original_live=episodes,
                edges=transitions(episodes),
                note="Retrospective reanalysis; never an online accuracy evaluation",
            ),
        )
    app = create_app(
        dict(
            ble_enabled=False,
            allowed_clients=["testclient"],
            research_models=str(tmp_path),
            research_architecture=architecture,
        ),
        store,
    )
    with TestClient(app) as client:
        yield client, app.state.research, store.rows("p001")


def test_insights_omits_bulk_context_and_preserves_other_sections(context_client):
    client, engine, rows = context_client
    response = client.get("/api/research/p001/insights")
    assert response.status_code == 200
    insights = response.json()
    assert BULK_KEYS.isdisjoint(insights["context"])

    full_context = client.get("/api/research/p001/context").json()
    assert insights["context"] == {
        key: value for key, value in full_context.items() if key not in BULK_KEYS
    }
    expected = engine.insights.snapshot("p001", AT, None, rows, "real")
    expected.update(
        preferences=settings(rows), data_mode="real", contains_sample=False,
    )
    assert {key: value for key, value in insights.items() if key != "context"} == expected


def test_context_endpoint_projects_reviews_and_history_for_display(context_client):
    client, _, rows = context_client
    # Calling insights must not affect the separately fetched display details.
    assert client.get("/api/research/p001/insights").status_code == 200
    response = client.get("/api/research/p001/context")
    assert response.status_code == 200
    context = response.json()
    reviews = [row["payload"] for row in rows if row["kind"] == "context_review"]
    history = [
        row["payload"] for row in reversed(rows)
        if row["kind"] == "context_inference"
    ]
    # Preserve limits, order and all fields except the unused bulky data.
    assert context["reviews"] == [
        {key: value for key, value in review.items() if key != "original_live"}
        for review in reviews[:7]
    ]
    assert context["history"] == [
        dict(item, model={}) for item in history[-288:]
    ]
    for review, original in zip(context["reviews"], reviews[:7]):
        assert "original_live" not in review
        for field in ("day", "created_at", "episodes", "edges", "note"):
            assert review[field] == original[field]
    for item, original in zip(context["history"], history[-288:]):
        assert item["model"] == {}
        # table()/describe() in context.html use only these inference fields.
        for field in ("at", "place", "task"):
            assert item[field] == original[field]
    assert context["edges"] == transitions(history[-288:])
    assert len(context["reviews"][0]["episodes"]) == 48
    assert len(context["history"][0]["feature_vector"]) == 512


def test_context_projection_leaves_stored_and_supplied_rows_untouched(context_client):
    client, engine, rows = context_client
    before = json.dumps(rows)
    assert client.get("/api/research/p001/context").status_code == 200
    projected = engine.context.snapshot("p001", AT, rows)
    assert all("original_live" not in review for review in projected["reviews"])
    assert all(item["model"] == {} for item in projected["history"])
    assert json.dumps(rows) == before
    assert engine.store.rows("p001") == rows
    assert all(
        row["payload"]["original_live"]
        for row in rows if row["kind"] == "context_review"
    )
    assert all(
        row["payload"]["model"]["training_loss"]
        for row in rows if row["kind"] == "context_inference"
    )


def test_context_projection_accepts_missing_optional_bulk_fields(context_client):
    _, engine, rows = context_client
    architecture = rows[0]["payload"]["architecture"]
    minimal_rows = [
        dict(
            kind="context_inference", id="minimal-inference", at=AT.isoformat(),
            payload=dict(
                at=AT.isoformat(), architecture=architecture,
                place=None, task=None,
            ),
        ),
        dict(
            kind="context_review", id="minimal-review", at=AT.isoformat(),
            payload=dict(
                day=AT.date().isoformat(), created_at=AT.isoformat(),
                architecture=architecture, episodes=[], edges={}, note="Empty day",
            ),
        ),
    ]
    context = engine.context.snapshot("p001", AT, minimal_rows)
    assert context["history"] == [minimal_rows[0]["payload"]]
    assert context["reviews"] == [minimal_rows[1]["payload"]]


def test_many_context_reviews_keep_serialized_insights_under_200_kb(context_client):
    client, _, _ = context_client
    response = client.get("/api/research/p001/insights")
    assert response.status_code == 200
    assert len(response.content) < 200_000
    assert len(json.dumps(response.json()).encode("utf-8")) < 200_000
    # Even the retained display details must stay out of insights.
    full_context = client.get("/api/research/p001/context")
    assert len(full_context.content) > 1_000_000


def test_context_projection_reduces_serialized_response(context_client):
    client, _, rows = context_client
    response = client.get("/api/research/p001/context")
    assert response.status_code == 200
    context = response.json()
    unprojected = dict(
        context,
        reviews=[
            row["payload"] for row in rows if row["kind"] == "context_review"
        ][:7],
        history=[
            row["payload"] for row in reversed(rows)
            if row["kind"] == "context_inference"
        ][-288:],
    )
    compact = dict(separators=(",", ":"), ensure_ascii=False)
    before = json.dumps(unprojected, **compact).encode("utf-8")
    after = json.dumps(context, **compact).encode("utf-8")
    assert b'"original_live":' in before
    assert b'"original_live":' not in after
    assert len(after) < 0.7 * len(before)


@pytest.mark.parametrize("data_mode", ["sample", "combined"])
def test_sample_insights_keep_real_only_context(context_client, data_mode):
    client, _, _ = context_client
    response = client.get(f"/api/research/p001/insights?data_mode={data_mode}")
    assert response.status_code == 200
    insights = response.json()
    assert insights["context"] == {"status": "real_only"}
    assert insights["data_mode"] == data_mode
    assert insights["contains_sample"] is True
