from datetime import timedelta, timezone
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app
from research.data import dt, Timeline, F
from research.train import batch
from research.context_data import (
    observations,
    annotations,
    label_at,
    catalog,
    fresh,
    node_profiles,
    node_descriptions,
    question_presentation,
)
from research.context_model import ContextLSTM, relation_loss
from research.context_learning import train_context
from research.context_service import ContextEngine
from research.service import Engine
from research.transfer import export_bundle, import_bundle
from research.questions import render_context_check


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


def test_node_profiles_use_self_reports_with_correction_and_receipt_cutoffs():
    end = dt("2026-09-07T10:00:00+09:00")
    original = label("place", "place_old", end - timedelta(hours=1), end)
    duplicate = dict(original, id="duplicate", at=(end + timedelta(seconds=2)).isoformat())
    correction = label(
        "place", "place_new", end - timedelta(hours=1), end, end + timedelta(minutes=10)
    )
    correction["payload"]["received_at"] = (end + timedelta(days=2)).isoformat()
    rows = [
        original,
        duplicate,
        correction,
        label("task", "task_work", end - timedelta(hours=1), end),
        record(
            "context_inference",
            "model-only",
            end,
            dict(place=dict(id="place_old", count=1000, days=1000)),
        ),
        record(
            "research_event",
            "merge",
            end + timedelta(hours=1),
            dict(
                kind="context_merge",
                received_at=(end + timedelta(days=3)).isoformat(),
                payload=dict(
                    axis="place", source_id="place_new", target_id="place_merged"
                ),
            ),
        ),
    ]
    before = node_profiles(rows, end + timedelta(days=1))
    assert before["place"]["place_old"] == dict(
        confirmed_count=1,
        confirmed_days=1,
        last_confirmed_at=end,
        dominant_band=None,
    )
    assert "place_new" not in before["place"]
    after = node_profiles(rows, end + timedelta(days=2))
    assert after["place"]["place_old"]["confirmed_count"] == 0
    assert after["place"]["place_new"]["confirmed_count"] == 1
    merged = node_profiles(rows, end + timedelta(days=3))
    assert "place_new" not in merged["place"]
    assert merged["place"]["place_merged"]["last_confirmed_at"] == end
    assert merged["task"]["task_work"]["confirmed_count"] == 1


def test_node_profiles_partial_corrections_do_not_multiply_confirmations():
    start = dt("2026-09-07T09:00:00+09:00")
    end = start + timedelta(hours=3)
    rows = [
        label("place", "place_a", start, end),
        label(
            "place",
            "place_b",
            start + timedelta(hours=1),
            start + timedelta(hours=1, minutes=30),
            end + timedelta(minutes=1),
        ),
        label(
            "place",
            "unknown",
            end - timedelta(hours=1),
            end,
            end + timedelta(minutes=2),
        ),
    ]
    original = json.dumps(rows)
    profiles = node_profiles(rows, end + timedelta(minutes=3))["place"]
    assert profiles["place_a"]["confirmed_count"] == 1
    assert profiles["place_a"]["last_confirmed_at"] == end - timedelta(hours=1)
    assert profiles["place_b"]["confirmed_count"] == 1
    assert "unknown" not in profiles
    assert json.dumps(rows) == original
    rows.append(label("place", "moving", start, end, end + timedelta(minutes=4)))
    profiles = node_profiles(rows, end + timedelta(minutes=5))["place"]
    assert all(p["confirmed_count"] == 0 for p in profiles.values())
    assert "moving" not in profiles


def test_node_profiles_aggregate_aliases_and_deduplicate_merged_intervals():
    end = dt("2026-09-07T10:00:00+09:00")
    rows = [
        label("place", key, end - timedelta(minutes=5), end)
        for key in ("place_a", "place_b")
    ]
    rows += [
        label(
            "place", key,
            end + timedelta(days=day, minutes=-5),
            end + timedelta(days=day),
        )
        for day, key in [(1, "place_a"), (2, "place_b")]
    ]
    at = end + timedelta(days=3)
    rows.append(record(
        "research_event", "merge", at,
        dict(kind="context_merge", payload=dict(
            axis="place", source_id="place_a", target_id="place_b",
        )),
    ))
    profile = node_profiles(rows, at)["place"]
    assert "place_a" not in profile
    assert profile["place_b"] == dict(
        confirmed_count=3,
        confirmed_days=3,
        last_confirmed_at=end + timedelta(days=2),
        dominant_band="morning",
    )


def test_node_profiles_count_jst_days_and_require_three_confirmations():
    end = dt("2026-09-06T23:30:00+00:00")
    rows = []
    for hour in range(3):
        cutoff = end + timedelta(hours=hour)
        report = label("place", "place_a", cutoff - timedelta(minutes=5), cutoff)
        for field in ("start", "end"):
            report["payload"]["payload"][field] = dt(
                report["payload"]["payload"][field]
            ).astimezone(timezone.utc).isoformat()
        rows.append(report)
    assert node_profiles(rows[:2], end + timedelta(days=1))["place"]["place_a"][
        "dominant_band"
    ] is None
    profile = node_profiles(rows, end + timedelta(days=1))["place"]["place_a"]
    assert profile["confirmed_count"] == 3
    assert profile["confirmed_days"] == 1
    assert profile["last_confirmed_at"] == end + timedelta(hours=2)
    assert profile["dominant_band"] == "morning"


@pytest.mark.parametrize(
    "hour,band", [(8, "morning"), (13, "afternoon"), (19, "evening"), (23, "night")]
)
def test_node_profiles_band_uses_only_confirmations_and_sixty_percent(hour, band):
    base = dt("2026-09-07T00:00:00+09:00")
    rows = []
    for day, report_hour in enumerate([hour, hour, hour, (hour + 12) % 24, (hour + 12) % 24]):
        end = base + timedelta(days=day, hours=report_hour)
        rows.append(label("task", "task_a", end - timedelta(minutes=5), end))
    at = base + timedelta(days=6)
    profile = node_profiles(rows, at)["task"]["task_a"]
    assert profile["confirmed_count"] == 5
    assert profile["confirmed_days"] == 5
    assert profile["dominant_band"] == band
    rendered = render_context_check(rows, at, "task")
    band_name = dict(morning="午前", afternoon="午後", evening="夕方", night="夜")[band]
    assert rendered["options"][0]["label"].endswith(f"（{band_name}の回答で多い）")
    other = base + timedelta(days=5, hours=(hour + 12) % 24)
    rows.append(label("task", "task_a", other - timedelta(minutes=5), other))
    assert node_profiles(rows, at)["task"]["task_a"]["dominant_band"] is None


@pytest.mark.parametrize("axis,noun", [("place", "拠点"), ("task", "作業")])
def test_context_labels_use_name_then_confirmation_then_anchor(axis, noun, tmp_path):
    at = dt("2026-09-07T15:00:00+09:00")
    confirmed = at - timedelta(days=1)
    anchor = at - timedelta(days=2)
    ids = [axis + "_" + suffix for suffix in ("a12345", "b67890", "cabcde")]
    rows = [
        record(
            "context_candidate",
            key,
            anchor,
            dict(axis=axis, class_id=key, anchor=anchor.isoformat()),
        )
        for key in ids
    ]
    rows += [
        label(axis, key, confirmed - timedelta(minutes=5), confirmed)
        for key in ids[:2]
    ]
    rows += [
        record(
            "research_event",
            "name",
            at - timedelta(minutes=1),
            dict(kind="context_name", payload=dict(axis=axis, class_id=ids[0], name="名前付き")),
        ),
        record(
            "context_inference",
            "current",
            at,
            {axis: dict(candidate_id=ids[2])},
        ),
    ]
    original = json.dumps(rows)
    rendered = render_context_check(rows, at, axis)
    options = {o["value"]: o["label"] for o in rendered["options"]}
    assert options[ids[0]] == "名前付き"
    assert options[ids[1]] == f"前回9月6日の15時に確認した{noun}"
    assert options[ids[2]] == f"9月5日の15時ごろの記録に似た{noun}候補（未確認）"
    assert [o["value"] for o in rendered["options"][:3]] == [ids[2], ids[0], ids[1]]
    assert "説明から判断できなければ『わからない』を選んでください。" in rendered["prompt"]
    for option in rendered["options"]:
        assert all(key not in option["label"] and key[-5:] not in option["label"] for key in ids)
    assert rendered["presentation"]["candidate_id"] == ids[2]
    assert [rendered["presentation"]["label_sources"][key] for key in ids] == [
        "name", "last_confirmed", "anchor"
    ]
    with_store = Store(":memory:")
    try:
        snapshot = ContextEngine(with_store, dict(research_models=str(tmp_path))).snapshot(
            "p001", at, rows
        )
        assert {n["id"]: n["description"] for n in snapshot["nodes"][axis]} == {
            key: options[key] for key in ids
        }
    finally:
        with_store.close()
    assert json.dumps(rows) == original


def test_context_labels_disambiguate_same_names_and_times_without_ids():
    at = dt("2026-09-07T15:00:00+09:00")
    keys = ["place_" + suffix for suffix in ("abcde", "f0123", "45678", "9abcd")]
    nodes = {
        key: dict(id=key, name="同じ名前", anchor=at.isoformat(), confirmed=index < 2)
        for index, key in enumerate(keys)
    }
    profiles = {
        keys[0]: dict(last_confirmed_at=at - timedelta(minutes=5)),
        keys[1]: dict(last_confirmed_at=at - timedelta(minutes=45)),
    }
    descriptions = node_descriptions("place", nodes, profiles)
    labels = [descriptions[key]["label"] for key in keys]
    assert len(set(labels)) == len(keys)
    assert "14時55分" in labels[0] and "14時15分" in labels[1]
    assert all("本人確認なし" in label for label in labels[2:])
    assert all(key[-5:] not in text for key in keys for text in labels)
    assert descriptions == node_descriptions("place", dict(reversed(list(nodes.items()))), profiles)
    # Equal unnamed anchors still remain unique, and literal special names are reserved.
    for node in nodes.values():
        node["name"] = None
    nodes[keys[0]]["name"] = "わからない"
    descriptions = node_descriptions("place", nodes, {})
    labels = [d["label"] for d in descriptions.values()]
    assert len(set(labels + ["わからない", "新しい拠点", "移動中"])) == len(keys) + 3


def test_api_budget_answer_idempotence_and_legacy_gps(tmp_path, monkeypatch):
    monkeypatch.setattr("research.service.utc", lambda: dt("2026-09-07T15:16:00+09:00"))
    store = Store(":memory:")
    rows, at = data(1)
    for r in rows:
        store.save(r["kind"], r["id"], "p001", dt(r["at"]), r["payload"])
    engine = Engine(store, dict(research_models=str(tmp_path)))
    q = engine.pending("p001", dt("2026-09-07T15:15:00+09:00"))
    assert q["kind"] == "placeCheck"
    presentation = question_presentation(store.rows("p001"), q["id"], dt(q["detectedAt"]))
    assert presentation["format"] == "descriptive-v1"
    assert presentation["question_id"] == q["id"]
    assert presentation["ordered_option_values"] == [o["value"] for o in q["options"]]
    assert "presentation" not in q
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
        event["payload"]["class_id"] = "place_not_offered"
        assert client.post("/api/research/p001/events", json=event).status_code == 422
        event["payload"]["class_id"] = q["options"][0]["value"]
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
