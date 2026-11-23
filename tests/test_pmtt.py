"""Prune one-off nodes only from the PMTT API's display projection."""

from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from lifelog_server import Store, create_app
from research.data import dt
from research.service import pmtt_projection


AT = dt("2026-10-29T16:40:00+09:00")


@pytest.fixture
def full_graph():
    return dict(
        nodes=[
            dict(id="isolated", kind="ContextPlace"),
            dict(id="leaf-a", kind="Observation", source="correction"),
            dict(id="a", kind="StateEpisode", at=AT.isoformat()),
            dict(id="b", kind="Day"),
            dict(id="leaf-b", kind="Pattern"),
            dict(id="hub", kind="ContextTask", name="Analysis", confirmed=True),
            dict(id="leaf-c", kind="ContextEpisode"),
            dict(id="leaf-d", kind="ContextEpisode"),
        ],
        edges=[
            dict(source="leaf-a", target="a", kind="provenance"),
            dict(source="a", target="b", kind="part_of", provenance="synthetic"),
            dict(source="b", target="leaf-b", kind="part_of"),
            dict(source="leaf-c", target="hub", kind="task"),
            dict(source="hub", target="leaf-d", kind="task"),
        ],
    )


def expected_projection(graph):
    # a, b and hub each have degree 2 in the full graph. After projection,
    # a and b have degree 1 and hub is isolated; all three must still survive.
    return dict(
        nodes=[node for node in graph["nodes"] if node["id"] in {"a", "b", "hub"}],
        edges=[graph["edges"][1]],
    )


def test_pmtt_projection_uses_full_degree_and_preserves_payloads(full_graph):
    before = deepcopy(full_graph)
    projected = pmtt_projection(full_graph)
    assert set(projected) == {"nodes", "edges"}
    assert projected == expected_projection(before)
    assert full_graph == before


@pytest.mark.parametrize(
    "graph",
    [
        dict(nodes=[], edges=[]),
        dict(nodes=[dict(id="isolated", kind="Pattern")], edges=[]),
        dict(
            nodes=[dict(id="a", kind="Observation"), dict(id="b", kind="Pattern")],
            edges=[dict(source="a", target="b", kind="provenance")],
        ),
    ],
)
def test_pmtt_projection_accepts_empty_or_only_one_off_nodes(graph):
    before = deepcopy(graph)
    assert pmtt_projection(graph) == dict(nodes=[], edges=[])
    assert graph == before


@pytest.mark.parametrize("data_mode", ["real", "sample", "combined"])
def test_pmtt_endpoint_projects_without_mutating_full_graph(
    tmp_path, monkeypatch, full_graph, data_mode
):
    store = Store(":memory:")
    app = create_app(
        dict(
            ble_enabled=False,
            allowed_clients=["testclient"],
            research_models=str(tmp_path),
        ),
        store,
    )
    calls = []

    def full_pmtt(participant, mode="real"):
        calls.append((participant, mode))
        return full_graph

    monkeypatch.setattr(app.state.research, "pmtt", full_pmtt)
    before = deepcopy(full_graph)
    with TestClient(app) as client:
        response = client.get("/api/research/p001/pmtt", params=dict(data_mode=data_mode))
        assert response.status_code == 200
        assert set(response.json()) == {"nodes", "edges"}
        assert response.json() == expected_projection(before)
        assert calls == [("p001", data_mode)]
        assert full_graph == before
        assert app.state.research.pmtt("p001", data_mode) == before


@pytest.mark.parametrize("architecture", ["legacy", "discovery"])
def test_pmtt_endpoint_keeps_built_graph_and_stored_records_complete(
    tmp_path, monkeypatch, architecture
):
    monkeypatch.setattr("research.service.utc", lambda: AT)
    store = Store(":memory:")
    store.save(
        "context_candidate", "unused", "p001", AT,
        dict(axis="place", class_id="place_unused", anchor=AT.isoformat()),
    )
    for key, payload in [
        ("two-axes", dict(place=dict(id="place_shared"), task=dict(id="task_shared"))),
        ("another", dict(place=dict(id="place_shared"), task=dict(id="task_shared"))),
        ("one-axis", dict(place=dict(id="place_shared"))),
        ("empty", {}),
    ]:
        store.save(
            "context_inference", key, "p001", AT,
            dict(at=AT.isoformat(), architecture=architecture, **payload),
        )
    store.save(
        "question_issued", "unsupported-proposal", "p001", AT,
        dict(
            kind="exerciseProposal",
            pmtt=dict(
                evidence_id="missing", usual_steps=4000,
                candidates=[dict(mets=3.0)],
            ),
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
    engine = app.state.research
    before_rows = store.rows("p001")
    before_graph = engine.pmtt("p001")
    omitted = {
        "place_unused", "context:one-axis", "context:empty",
        "proposal:unsupported-proposal",
    }
    assert omitted.issubset({node["id"] for node in before_graph["nodes"]})
    with TestClient(app) as client:
        response = client.get("/api/research/p001/pmtt")
        assert response.status_code == 200
        projected = response.json()
        assert set(projected) == {"nodes", "edges"}
        assert {node["id"] for node in projected["nodes"]} == {
            "place_shared", "task_shared", "context:two-axes", "context:another",
        }
        assert len(projected["edges"]) == 4
        assert projected == pmtt_projection(before_graph)
        assert engine.pmtt("p001") == before_graph
        assert store.rows("p001") == before_rows


@pytest.mark.parametrize("upstream_status", [200, 503])
def test_pmtt_proxy_preserves_upstream_projection_or_error(
    tmp_path, monkeypatch, full_graph, upstream_status
):
    # An upstream projects its own complete graph. Pruning its response again
    # would incorrectly remove a/b/hub, whose projected degrees are only 1/1/0.
    payload = (
        expected_projection(full_graph)
        if upstream_status == 200 else dict(detail="Upstream unavailable")
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(upstream_status, json=payload)

    async_client = httpx.AsyncClient

    def upstream_client(**kwargs):
        return async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("research.service.httpx.AsyncClient", upstream_client)
    store = Store(":memory:")
    app = create_app(
        dict(
            ble_enabled=False,
            allowed_clients=["testclient"],
            research_models=str(tmp_path),
            research_upstream="http://research-worker",
        ),
        store,
    )

    def unexpected_local_graph(*args):
        pytest.fail("A proxied PMTT response must not build the local graph")

    monkeypatch.setattr(app.state.research, "pmtt", unexpected_local_graph)
    with TestClient(app) as client:
        response = client.get("/api/research/p001/pmtt?data_mode=combined")
        assert response.status_code == upstream_status
        assert response.json() == payload
        assert len(requests) == 1
        assert requests[0].url.path == "/api/research/p001/pmtt"
        assert requests[0].url.params["data_mode"] == "combined"
