from datetime import timedelta
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app
from research.data import dt, Timeline, F
from research.train import batch
from research.context_data import observations, annotations, label_at, catalog, fresh
from research.context_model import ContextLSTM, relation_loss
from research.context_learning import train_context
from research.context_service import ContextEngine
from research.service import Engine
from research.transfer import export_bundle, import_bundle


def record(kind, key, at, payload):
    return dict(
        kind=kind, id=key, participant="p001", at=at.isoformat(), payload=payload
    )


def label(axis, key, start, end, received=None):
    received = received or end + timedelta(seconds=1)
    return record(
        "research_event",
        axis + key + end.isoformat(),
        received,
        dict(
            kind="context_label",
            payload=dict(
                axis=axis, class_id=key, start=start.isoformat(), end=end.isoformat()
            ),
        ),
    )


def data(days=8):
    rows = []
    start = dt("2026-09-07T08:00:00+09:00")
    for d in range(days):
        for h in range(8):
            at = start + timedelta(days=d, hours=h)
            rows.append(
                record(
                    "session",
                    f"{d}-{h}",
                    at,
                    dict(
                        started_at=(at - timedelta(minutes=1)).isoformat(),
                        ended_at=at.isoformat(),
                        events=[
                            dict(
                                type="commit",
                                time=(at - timedelta(seconds=5)).isoformat(),
                                data=dict(text="仕事です" if h % 2 else "遊んでいます"),
                            ),
                            dict(
                                type="key",
                                time=(at - timedelta(seconds=6)).isoformat(),
                                data={},
                            ),
                        ],
                    ),
                )
            )
            rows += [
                label(
                    "place",
                    "place_home" if h < 4 else "place_lab",
                    at - timedelta(minutes=1),
                    at,
                ),
                label(
                    "task",
                    "task_work" if h % 2 else "task_play",
                    at - timedelta(minutes=1),
                    at,
                ),
            ]
    return rows, start + timedelta(days=days - 1, hours=9)


def test_annotations_are_independent_and_no_future_labels():
    rows, at = data(1)
    t = dt("2026-09-07T08:00:00+09:00")
    assert label_at(annotations(rows, t), "place", t) is None
    a = annotations(rows, at)
    assert label_at(a, "place", t)["class_id"] == "place_home"
    changed = rows + [label("task", "task_break", t - timedelta(seconds=10), t, at)]
    b = annotations(changed, at)
    assert label_at(b, "task", t)["class_id"] == "task_break"
    assert label_at(b, "place", t)["class_id"] == "place_home"
    # Same place / different task; same task / different place occur in actual labels.
    assert (
        label_at(a, "task", t + timedelta(hours=2))["class_id"]
        == label_at(a, "task", t + timedelta(hours=4))["class_id"]
    )
    assert (
        label_at(a, "place", t + timedelta(hours=2))["class_id"]
        != label_at(a, "place", t + timedelta(hours=4))["class_id"]
    )


def test_gps_questions_and_future_observations_do_not_change_features():
    rows, at = data(1)
    t = at - timedelta(hours=2)
    extra = [
        record(
            "research_event",
            "gps",
            t,
            dict(kind="place_visit", payload=dict(place_id="gps_cheat", latitude=99)),
        ),
        label("place", "place_fake", t - timedelta(minutes=5), t),
    ]
    a = Timeline(observations(rows)).sample(t)
    b = Timeline(observations(rows + extra)).sample(t)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not fresh(
        [record("switchbot_observation", "door", t, dict(state="open"))], t
    )


def test_gradients_and_separate_metric_objectives():
    rows, at = data(1)
    sample = Timeline(observations(rows)).sample(at)
    model = ContextLSTM(hidden=8)
    out = model(batch([sample, sample], "cpu", torch.zeros(F), torch.ones(F)))
    loss = out["next"].square().mean() + out["place"].sum() + out["task"].sum()
    loss.backward()
    for lstm in (model.encoder.micro, model.encoder.meso, model.encoder.macro):
        assert lstm.weight_ih_l0.grad.abs().sum() > 0
    for head in (model.place, model.task):
        assert head[0].weight.grad.abs().sum() > 0
    same = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    assert relation_loss(same, ["place_a", "place_a"]) == 0
    assert relation_loss(same, ["place_a", "place_b"]) > 0
    assert relation_loss(same, [None, "place_b"]) == 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("context_model")
    rows, at = data()
    report = train_context(
        rows,
        root / "p001" / "real" / "context",
        at,
        epochs=2,
        hidden=8,
        evaluate_test=True,
    )
    assert report["status"] == "trained"
    return root, rows, at, report


def test_training_has_held_out_days_and_untrained_gate(trained, tmp_path):
    root, rows, at, report = trained
    assert report["train_before"] < report["test_from"]
    assert report["validation"]["place"]["n"] > 0 and report["test"]["task"]["n"] > 0
    store = Store(":memory:")
    try:
        engine = ContextEngine(store, dict(research_models=str(root)))
        old = engine.infer("p001", at - timedelta(days=2), rows)
        assert old["status"] == "checkpoint_from_future"
        result = engine.infer("p001", at + timedelta(days=1), rows)
        assert result["status"] == "observation_unknown" and result["place"] is None
        blank = ContextEngine(store, dict(research_models=str(tmp_path))).infer(
            "p001", at
        )
        assert blank["status"] == "untrained" and blank["task"] is None
    finally:
        store.close()


def test_merge_split_and_rename_preserve_originals():
    rows, at = data(1)
    t = dt("2026-09-07T08:00:00+09:00")
    original = json.dumps(rows)
    rows2 = rows + [
        record(
            "research_event",
            "merge",
            at,
            dict(
                kind="context_merge",
                payload=dict(
                    axis="place", source_id="place_lab", target_id="place_home"
                ),
            ),
        ),
        label("task", "task_new", t - timedelta(seconds=10), t, at),
    ]
    nodes, _ = catalog(rows2, at)
    assert "place_lab" not in nodes["place"] and "task_new" in nodes["task"]
    assert all(
        l["class_id"] == "place_home"
        for l in annotations(rows2, at)
        if l["axis"] == "place"
    )
    assert json.dumps(rows) == original


def test_api_budget_answer_idempotence_and_legacy_gps(tmp_path, monkeypatch):
    monkeypatch.setattr("research.service.utc", lambda: dt("2026-09-07T15:16:00+09:00"))
    store = Store(":memory:")
    rows, at = data(1)
    for r in rows:
        store.save(r["kind"], r["id"], "p001", dt(r["at"]), r["payload"])
    engine = Engine(store, dict(research_models=str(tmp_path)))
    q = engine.pending("p001", dt("2026-09-07T15:15:00+09:00"))
    assert q["kind"] == "placeCheck"
    app = create_app(
        dict(
            ble_enabled=False,
            allowed_clients=["testclient"],
            research_models=str(tmp_path),
        ),
        store,
    )
    with TestClient(app) as client:
        event = dict(
            schema_version="research-v2",
            event_id="answer",
            participant_id="p001",
            occurred_at="2026-09-07T15:16:00+09:00",
            kind="context_label",
            payload=dict(
                axis="place",
                class_id=q["options"][0]["value"],
                start="2026-09-07T15:10:00+09:00",
                end=q["detectedAt"],
                question_id=q["id"],
            ),
        )
        assert client.post("/api/research/p001/events", json=event).json()["inserted"]
        assert not client.post("/api/research/p001/events", json=event).json()[
            "inserted"
        ]
        assert engine.pending("p001", dt("2026-09-07T15:17:00+09:00")) is None
        event["event_id"] = "wrong-axis"
        event["payload"]["axis"] = "task"
        assert client.post("/api/research/p001/events", json=event).status_code == 422
        event.update(
            event_id="gps",
            kind="place_visit",
            payload=dict(place_id="old", phase="arrival", accuracy_m=10),
        )
        assert (
            client.post("/api/research/p001/events", json=event).json()["ignored"]
            == "gps_disabled"
        )
        assert not any(r["id"] == "gps" for r in store.rows("p001"))
        assert client.get("/research/p001/context").status_code == 200
        assert client.get("/api/research/p001/context").json()["gps_used"] is False


def test_review_retains_live_and_transfer_preserves_ids_and_weights(trained, tmp_path):
    root, rows, at, _ = trained
    source = Store(":memory:")
    dest = Store(":memory:")
    try:
        for r in rows:
            source.save(r["kind"], r["id"], "p001", dt(r["at"]), r["payload"])
        engine = ContextEngine(source, dict(research_models=str(root)))
        t = dt("2026-09-07T08:00:00+09:00")
        source.save(
            "context_inference",
            "original",
            "p001",
            t,
            dict(at=t.isoformat(), place=None, task=None, status="untrained"),
        )
        review = engine.review("p001", "2026-09-07", at)
        assert review["original_live"][0]["status"] == "untrained"
        assert review["episodes"][0]["place"]["source"] == "self_report"
        assert (
            next(r for r in source.rows("p001") if r["id"] == "original")["payload"][
                "place"
            ]
            is None
        )
        bundle = tmp_path / "transfer.zip"
        export_bundle(source, "p001", root, bundle)
        import_bundle(dest, "p001", tmp_path / "models", bundle)
        before = engine.infer("p001", at, rows)
        after = ContextEngine(
            dest, dict(research_models=str(tmp_path / "models"))
        ).infer("p001", at, rows)
        assert before == after
        assert catalog(source.rows("p001"), at) == catalog(dest.rows("p001"), at)
        with pytest.raises(ValueError, match="empty"):
            import_bundle(dest, "p001", tmp_path / "models", bundle)
    finally:
        source.close()
        dest.close()


def test_late_merge_does_not_relabel_past_and_corrupt_transfer_is_rejected(tmp_path):
    import zipfile

    rows, at = data(2)
    before = at - timedelta(hours=1)
    rows.append(
        record(
            "research_event",
            "later-merge",
            at,
            dict(
                kind="context_merge",
                payload=dict(
                    axis="place", source_id="place_lab", target_id="place_home"
                ),
            ),
        )
    )
    assert any(l["class_id"] == "place_lab" for l in annotations(rows, before))
    assert not any(l["class_id"] == "place_lab" for l in annotations(rows, at))
    source = Store(":memory:")
    dest = Store(":memory:")
    try:
        r = rows[0]
        source.save(r["kind"], r["id"], "p001", dt(r["at"]), r["payload"])
        good = tmp_path / "good.zip"
        bad = tmp_path / "bad.zip"
        export_bundle(source, "p001", tmp_path / "empty-models", good)
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
            for name in src.namelist():
                dst.writestr(
                    name, b"corrupt" if name == "records.json" else src.read(name)
                )
        with pytest.raises(ValueError, match="integrity"):
            import_bundle(dest, "p001", tmp_path / "models", bad)
        assert dest.rows("p001") == [] and not (tmp_path / "models" / "p001").exists()
    finally:
        source.close()
        dest.close()
