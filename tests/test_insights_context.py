"""Keep bulky context details on /context without changing other insights."""

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
    store = Store(":memory:")
    # Large nested vectors make even the existing 7-review/288-inference limits
    # exceed the insights budget. All records and features here are synthetic.
    features = [0.125] * 512

    def inference(at, index):
        return dict(
            at=at.isoformat(),
            architecture=architecture,
            status="ready",
            place=dict(id=f"place_{index % 2}", status="estimated"),
            task=dict(id=f"task_{index % 3}", status="estimated"),
            feature_vector=features,
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


def test_context_endpoint_keeps_full_review_and_history_entries(context_client):
    client, _, rows = context_client
    # Calling insights must not mutate or trim the separately fetched details.
    assert client.get("/api/research/p001/insights").status_code == 200
    response = client.get("/api/research/p001/context")
    assert response.status_code == 200
    context = response.json()
    reviews = [row["payload"] for row in rows if row["kind"] == "context_review"]
    history = [
        row["payload"] for row in reversed(rows)
        if row["kind"] == "context_inference"
    ]
    # Preserve the existing endpoint's limits, order and complete nested payloads.
    assert context["reviews"] == reviews[:7]
    assert context["history"] == history[-288:]
    assert context["edges"] == transitions(history[-288:])
    assert len(context["reviews"][0]["episodes"]) == 48
    assert len(context["history"][0]["feature_vector"]) == 512


def test_many_context_reviews_keep_serialized_insights_under_200_kb(context_client):
    client, _, _ = context_client
    response = client.get("/api/research/p001/insights")
    assert response.status_code == 200
    assert len(response.content) < 200_000
    assert len(json.dumps(response.json()).encode("utf-8")) < 200_000
    # Prove that the fixture would reproduce the oversized embedded context.
    full_context = client.get("/api/research/p001/context")
    assert len(full_context.content) > 1_000_000


@pytest.mark.parametrize("data_mode", ["sample", "combined"])
def test_sample_insights_keep_real_only_context(context_client, data_mode):
    client, _, _ = context_client
    response = client.get(f"/api/research/p001/insights?data_mode={data_mode}")
    assert response.status_code == 200
    insights = response.json()
    assert insights["context"] == {"status": "real_only"}
    assert insights["data_mode"] == data_mode
    assert insights["contains_sample"] is True
