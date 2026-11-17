"""Delivery regressions; pure serving/audit tests also run without torch or HTTP.

python3 -m unittest discover -s tests -p test_baseline_delivery.py -v
"""
from datetime import timedelta
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from lifelog.store import Store
from research.data import Label, SCALES, Timeline, dt
from research.data_modes import rows_for_mode
from research.delivery import (
    SCOPES, apply_delivery, answered_predictions, decide, evaluate_delivery,
    item_state, load_state, validation_evidence,
)
from research.experiment_metrics import sequential_baselines
from research.inference import infer_state
from research.insights import InsightEngine, fatigue_change
from research.io import atomic_json
from research.questions import QuestionQueue, settings


AT = dt("2026-11-01T10:00:00+09:00")
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_HTTP = all(importlib.util.find_spec(package) is not None for package in ("fastapi", "httpx", "pydantic"))


class LocalEngine:
    """The storage/path contract, without the optional service dependencies."""
    sample_participant = "p001-sample"
    config = {}

    def __init__(self, store, root):
        self.store, self.root = store, Path(root)

    def rows(self, participant, mode="real"):
        return rows_for_mode(self.store, participant, mode, self.sample_participant)

    def directory(self, participant, mode="real"):
        return self.root / participant / mode

    def read_path(self, participant, mode, name):
        return self.directory(participant, mode) / name

    def status(self, participant, mode="real"):
        path = self.read_path(participant, mode, "status.json")
        return json.loads(path.read_text()) if path.exists() else dict(status="untrained")


def answer(store, key, at, scores, scope="current", received=None, participant="p001"):
    received = received or at + timedelta(seconds=30)
    store.save("narrative", key, participant, at + timedelta(seconds=30), dict(
        question_shown_at=at.isoformat(), received_at=received.isoformat(),
        question=dict(kind="currentState" if scope == "current" else "nightReflection"),
        answer=dict(free_text=json.dumps(scores))))


def preferences(store, at=AT - timedelta(days=20), participant="p001", **values):
    store.save("research_event", at.isoformat(), participant, at,
               dict(kind="preferences", payload=values, received_at=at.isoformat()))


def prediction(store, key, at, mean=2., last=2., lstm=4., scope="current", name="fatigue",
               source="lstm", saved=None, mode="real"):
    store.save("inference", key, "p001", saved or at, dict(
        at=at.isoformat(), data_mode=mode, status="ready", model=dict(fingerprint="model-a"),
        **{scope: {name: dict(value=lstm if source == "lstm" else last, source=source,
                             provisional=source == "baseline", baseline_method=None if source == "lstm" else "last",
                             lstm_value=lstm, model_id="model-a",
                             baselines=dict(mean=dict(value=mean), last=dict(value=last)))}}))


def validation(ids=None, train_n=10, n=10, days=3, mean=1., last=1., lstm=.8):
    return dict(train_n=train_n, validation_n=n, validation_days=days,
                answer_ids=ids or [f"answer-{i}" for i in range(n)],
                maes=dict(mean=mean, last=last, lstm=lstm))


def evidence(qualified="current.mood", ids=None):
    items = {f"{scope}.{name}": validation(train_n=0, n=0, days=0, mean=None, last=None, lstm=None)
             for scope in SCOPES for name in SCALES}
    items[qualified] = validation(ids=ids)
    return dict(model_id="active-model", split_id="split-identity",
                split=dict(train_before="2026-10-20", test_from="2026-11-01",
                           train=[], validation=[dict(id=key) for key in items[qualified]["answer_ids"]]),
                items=items)


class DeliveryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.engine = LocalEngine(self.store, self.temp.name)

    def enable(self):
        preferences(self.store, baseline_estimates_enabled=True)

    def active(self, mode="real"):
        path = self.engine.read_path("p001", mode, "active.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mocked checkpoint")


class BaselineDeliveryTests(DeliveryFixture):
    def test_sequential_baselines_ignore_future_and_delayed_answers(self):
        answer(self.store, "early", AT - timedelta(hours=2), dict(mood=1))
        answer(self.store, "recent", AT - timedelta(hours=1), dict(mood=3))
        answer(self.store, "future", AT + timedelta(hours=1), dict(mood=5))
        answer(self.store, "late", AT - timedelta(minutes=30), dict(mood=5),
               received=AT + timedelta(days=1))
        answer(self.store, "equal-reception", AT - timedelta(minutes=20), dict(mood=5), received=AT)
        baseline = sequential_baselines(Timeline(self.engine.rows("p001")).labels, AT, "current")
        self.assertEqual(baseline["mood"], dict(n=2, mean=2., last=3.))
        self.assertEqual(baseline["fatigue"], dict(n=0, mean=None, last=None))

    def test_sequential_baselines_update_with_prior_validation_answers_only(self):
        labels = [Label(AT + timedelta(hours=i), AT + timedelta(hours=i, seconds=30),
                        "current", np.array([v, np.nan, np.nan, np.nan, np.nan]), id=str(i))
                  for i, v in enumerate((1., 5., 3.))]
        second = sequential_baselines(labels, labels[1].at, "current")["mood"]
        third = sequential_baselines(labels, labels[2].at, "current")["mood"]
        self.assertEqual(second, dict(n=1, mean=1., last=1.))
        self.assertEqual(third, dict(n=2, mean=3., last=5.))

    def test_default_off_and_explicit_off_produce_no_provisional_estimate(self):
        answer(self.store, "first", AT - timedelta(minutes=1), dict(mood=4))
        self.assertFalse(settings(self.engine.rows("p001"))["baseline_estimates_enabled"])
        self.assertIsNone(infer_state(self.engine, "p001", AT)["current"])
        preferences(self.store, baseline_estimates_enabled=False)
        self.assertIsNone(infer_state(self.engine, "p001", AT)["current"])
        self.assertFalse(self.engine.directory("p001").exists())

    def test_first_received_answer_enables_provisional_estimates_per_item(self):
        self.enable()
        answer(self.store, "first", AT, dict(mood=4, fatigue=2))
        self.assertIsNone(infer_state(self.engine, "p001", AT)["current"])
        result = infer_state(self.engine, "p001", AT + timedelta(minutes=1))
        self.assertEqual(result["status"], "ready")
        item = result["current"]["mood"]
        self.assertEqual(item["value"], 4.)
        self.assertEqual(item["source"], "baseline")
        self.assertTrue(item["provisional"])
        self.assertEqual(item["baseline_method"], "last")
        self.assertEqual(item["baselines"], dict(mean=dict(value=4., mae=None, n=0),
                                                last=dict(value=4., mae=None, n=0)))
        self.assertIsNone(result["current"]["stress"]["value"])
        self.assertIsNone(result["daily"])
        self.assertIsNone(result["activity"])

    def test_settings_use_received_cutoff_and_real_participant_in_sample_mode(self):
        self.store.save("research_event", "late-setting", "p001", AT - timedelta(days=1),
                        dict(kind="preferences", received_at=(AT + timedelta(days=1)).isoformat(),
                             payload=dict(baseline_estimates_enabled=True)))
        answer(self.store, "sample-first", AT - timedelta(minutes=1), dict(mood=4),
               participant="p001-sample")
        self.assertIsNone(infer_state(self.engine, "p001", AT, "sample")["current"])
        result = infer_state(self.engine, "p001", AT + timedelta(days=2), "sample")
        self.assertEqual(result["current"]["mood"]["value"], 4.)
        self.assertTrue(result["contains_sample"])

    def test_past_saved_prequential_maes_choose_mean(self):
        self.enable()
        prediction(self.store, "pred", AT - timedelta(minutes=2), mean=1., last=4., lstm=None,
                   name="mood", source="baseline")
        answer(self.store, "answered", AT - timedelta(minutes=1), dict(mood=1))
        result = infer_state(self.engine, "p001", AT)
        item = result["current"]["mood"]
        self.assertEqual(item["baseline_method"], "mean")
        self.assertEqual(item["baselines"]["mean"]["mae"], 0.)
        self.assertEqual(item["baselines"]["last"]["mae"], 3.)
        self.assertEqual(item["baselines"]["mean"]["n"], 1)

    def test_tie_uses_last_and_unanswered_predictions_do_not_choose_mean(self):
        self.enable()
        answer(self.store, "first", AT - timedelta(hours=1), dict(mood=2))
        prediction(self.store, "tie", AT - timedelta(minutes=20), mean=1., last=3., lstm=None,
                   name="mood", source="baseline")
        answer(self.store, "tie-answer", AT - timedelta(minutes=19), dict(mood=2))
        prediction(self.store, "pending", AT - timedelta(minutes=10), mean=1., last=5., lstm=None,
                   name="mood", source="baseline")
        answer(self.store, "pending-answer", AT - timedelta(minutes=9), dict(mood=1),
               received=AT + timedelta(days=1))
        item = infer_state(self.engine, "p001", AT)["current"]["mood"]
        self.assertEqual(item["baseline_method"], "last")
        self.assertEqual(item["baselines"]["mean"]["mae"], 1.)
        self.assertEqual(item["baselines"]["last"]["mae"], 1.)
        self.assertEqual(item["baselines"]["last"]["n"], 1)
        self.assertEqual(item["prior_answers"], 2)

    def test_predictions_persisted_after_answer_are_never_scored(self):
        prediction(self.store, "after-answer", AT - timedelta(minutes=2),
                   saved=AT + timedelta(minutes=1))
        answer(self.store, "target", AT - timedelta(minutes=1), dict(fatigue=2))
        labels = Timeline(self.engine.rows("p001")).labels
        points = answered_predictions(self.store.rows("p001"), labels, AT + timedelta(minutes=2))
        self.assertEqual(points["current.fatigue"], [])

    def test_answer_does_not_score_a_later_prediction_or_another_data_mode(self):
        prediction(self.store, "future", AT + timedelta(minutes=1))
        prediction(self.store, "sample", AT - timedelta(minutes=1), mode="sample")
        answer(self.store, "target", AT, dict(fatigue=2))
        labels = Timeline(self.engine.rows("p001")).labels
        self.assertEqual(answered_predictions(self.store.rows("p001"), labels, AT + timedelta(minutes=2))
                         ["current.fatigue"], [])

    def test_disabled_retains_validated_lstm_values_with_metadata(self):
        response = dict(status="ready", current=dict(mood=dict(value=3.2, status="estimated")),
                        daily=None, comparisons={})
        delivered = apply_delivery(self.engine, "p001", AT, "real", [], response)
        self.assertEqual(delivered["current"]["mood"],
                         dict(value=3.2, status="estimated", source="lstm", provisional=False,
                              baseline_method=None))

    def test_switch_streak_requires_three_distinct_days_and_new_validation(self):
        item, decision, _, _ = decide(item_state(), validation(), [], AT)
        self.assertEqual(item["win_streak"], 1)
        self.assertEqual(decision, "hold")
        same_day = validation(ids=[f"new-{i}" for i in range(10)])
        item, _, _, _ = decide(item, same_day, [], AT + timedelta(hours=1))
        self.assertEqual(item["win_streak"], 1)
        item, _, _, _ = decide(item, validation(), [], AT + timedelta(days=1))
        self.assertEqual(item["win_streak"], 1)
        item, _, _, _ = decide(item, same_day, [], AT + timedelta(days=1, hours=1))
        self.assertEqual(item["win_streak"], 2)
        final = validation(ids=["third-day"] + same_day["answer_ids"])
        item, decision, _, _ = decide(item, final, [], AT + timedelta(days=2))
        self.assertEqual(decision, "switch")
        self.assertEqual(item["source"], "lstm")
        self.assertEqual(item["win_streak"], 3)

    def test_no_switch_below_any_quantity_requirement(self):
        for overrides in (dict(train_n=9), dict(n=9), dict(days=2)):
            with self.subTest(overrides=overrides):
                item = item_state()
                for day in range(3):
                    item, decision, reason, _ = decide(
                        item, validation(ids=[str(day)], **overrides), [], AT + timedelta(days=day))
                    self.assertEqual(decision, "hold")
                    self.assertEqual(reason, "quantity_failed")
                    self.assertEqual(item["win_streak"], 0)

    def test_no_switch_without_absolute_and_relative_margins_against_both(self):
        for scores in (dict(mean=1., last=.5, lstm=.45),   # fails absolute against last
                       dict(mean=.5, last=1., lstm=.45),   # fails absolute against mean
                       dict(mean=2., last=2., lstm=1.85),  # fails relative
                       dict(mean=.08, last=.08, lstm=0.), # impossible .1 improvement
                       dict(mean=1., last=None, lstm=.1)):
            with self.subTest(scores=scores):
                item = item_state()
                for day in range(3):
                    item, decision, reason, _ = decide(
                        item, validation(ids=[str(day)], **scores), [], AT + timedelta(days=day))
                    self.assertEqual(decision, "hold")
                    self.assertEqual(reason, "quality_failed")
                    self.assertEqual(item["win_streak"], 0)

    def test_exact_margin_boundaries_qualify(self):
        for scores in (dict(mean=1., last=1., lstm=.9), dict(mean=2., last=2., lstm=1.8)):
            item, _, _, _ = decide(item_state(), validation(**scores), [], AT)
            self.assertEqual(item["win_streak"], 1)

    def test_failed_evaluation_resets_winning_streak(self):
        item, _, _, _ = decide(item_state(), validation(ids=["one"]), [], AT)
        item, _, _, _ = decide(item, validation(ids=["two"]), [], AT + timedelta(days=1))
        self.assertEqual(item["win_streak"], 2)
        item, _, _, _ = decide(item, validation(ids=["failure"], lstm=1.), [], AT + timedelta(days=2))
        self.assertEqual(item["win_streak"], 0)
        item, decision, _, _ = decide(item, validation(ids=["three"]), [], AT + timedelta(days=3))
        self.assertEqual(item["win_streak"], 1)
        self.assertEqual(decision, "hold")

    def test_per_item_switching_persists_and_audits_every_evaluation(self):
        self.enable()
        self.active()
        reports = []
        for day in range(3):
            current = evidence(ids=[f"day-{day}-{i}" for i in range(10)])
            with patch("research.delivery.validation_evidence", return_value=current):
                reports.extend(evaluate_delivery(self.engine, "p001", at=AT + timedelta(days=day)))
            self.engine = LocalEngine(self.store, self.temp.name)  # restart
        state = load_state(self.engine.directory("p001"))
        self.assertEqual(state["items"]["current.mood"]["source"], "lstm")
        self.assertTrue(all(item["source"] == "baseline" for key, item in state["items"].items()
                            if key != "current.mood"))
        self.assertFalse(self.engine.directory("p001", "sample").exists())
        self.assertFalse(self.engine.directory("other").exists())
        audit = self.engine.directory("p001") / "delivery-audit.jsonl"
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        self.assertEqual(len(records), 30)
        self.assertEqual(len([r for r in records if r["decision"] == "switch"]), 1)
        self.assertEqual(reports, records)
        for record in records:
            self.assertTrue({"model_id", "split_id", "split", "train_n", "validation_n",
                             "validation_days", "validation_maes", "thresholds", "win_streak",
                             "decision", "reason", "at"} <= record.keys())
            self.assertEqual(set(record["validation_maes"]), {"mean", "last", "lstm"})
        answer(self.store, "current", AT, dict.fromkeys(SCALES, 2))
        answer(self.store, "daily", AT, dict.fromkeys(SCALES, 4), scope="daily")
        response = dict(status="ready", current={name: dict(value=3., status="estimated") for name in SCALES},
                        daily={name: dict(value=3., status="estimated") for name in SCALES}, model_id="active-model")
        delivered = apply_delivery(self.engine, "p001", AT + timedelta(days=3), "real",
                                   self.engine.rows("p001"), response)
        self.assertEqual(delivered["current"]["mood"]["source"], "lstm")
        self.assertFalse(delivered["current"]["mood"]["provisional"])
        self.assertIsNone(delivered["current"]["mood"]["baseline_method"])
        self.assertEqual(delivered["current"]["mood"]["value"], 3.)
        self.assertEqual(delivered["current"]["fatigue"]["source"], "baseline")
        self.assertEqual(delivered["current"]["fatigue"]["value"], 2.)
        self.assertEqual(delivered["daily"]["mood"]["source"], "baseline")
        self.assertEqual(delivered["daily"]["mood"]["value"], 4.)
        self.assertIn("baselines", delivered["current"]["mood"])

    def test_audit_recovers_state_when_atomic_state_write_was_interrupted(self):
        self.enable()
        self.active()
        with patch("research.delivery.validation_evidence", return_value=evidence()):
            with patch("research.delivery.atomic_json", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    evaluate_delivery(self.engine, "p001", at=AT)
        directory = self.engine.directory("p001")
        self.assertFalse((directory / "delivery-state.json").exists())
        recovered = load_state(directory, recover=True)
        self.assertEqual(recovered["items"]["current.mood"]["win_streak"], 1)
        with patch("research.delivery.validation_evidence", return_value=evidence()):
            evaluate_delivery(self.engine, "p001", at=AT + timedelta(hours=1))
        self.assertEqual(load_state(directory)["items"]["current.mood"]["win_streak"], 1)
        self.assertEqual(len((directory / "delivery-audit.jsonl").read_text().splitlines()), 20)

    def test_disabled_and_missing_active_model_never_evaluate_or_write_state(self):
        with patch("research.delivery.validation_evidence") as score:
            self.active()
            self.assertIsNone(evaluate_delivery(self.engine, "p001", at=AT))
            self.enable()
            self.engine.read_path("p001", "real", "active.pt").unlink()
            self.assertIsNone(evaluate_delivery(self.engine, "p001", at=AT))
            score.assert_not_called()
        self.assertFalse((self.engine.directory("p001") / "delivery-state.json").exists())

    def test_live_degradation_reverts_only_on_two_days_with_new_answers(self):
        self.enable()
        self.active()
        directory = self.engine.directory("p001")
        switched = dict(item_state(), source="lstm", win_streak=3, switched_at=AT.isoformat())
        atomic_json(directory / "delivery-state.json", dict(items={"current.fatigue": switched}, audit_id=None))
        for i in range(10):
            t = AT + timedelta(minutes=2 * i + 1)
            prediction(self.store, f"live-{i}", t, mean=2., last=2., lstm=4.)
            answer(self.store, f"answer-{i}", t + timedelta(seconds=10), dict(fatigue=2))
        with patch("research.delivery.validation_evidence", return_value=evidence()):
            evaluate_delivery(self.engine, "p001", at=AT + timedelta(hours=1))
            state = load_state(directory)["items"]["current.fatigue"]
            self.assertEqual(state["revert_streak"], 1)
            self.assertEqual(state["source"], "lstm")
            # Same day, and next day without new answers, cannot increment.
            evaluate_delivery(self.engine, "p001", at=AT + timedelta(hours=2))
            evaluate_delivery(self.engine, "p001", at=AT + timedelta(days=1))
            self.assertEqual(load_state(directory)["items"]["current.fatigue"]["revert_streak"], 1)
            t = AT + timedelta(days=1, minutes=1)
            prediction(self.store, "live-new", t, mean=2., last=2., lstm=4.)
            answer(self.store, "answer-new", t + timedelta(seconds=10), dict(fatigue=2))
            reports = evaluate_delivery(self.engine, "p001", at=AT + timedelta(days=1, hours=1))
        state = load_state(directory)["items"]["current.fatigue"]
        self.assertEqual(state["source"], "baseline")
        self.assertEqual(state["win_streak"], 0)
        record = next(r for r in reports if r["item_key"] == "current.fatigue")
        self.assertEqual(record["decision"], "revert")
        self.assertEqual(record["checks"]["live_n"], 10)
        self.assertEqual(record["checks"]["live_maes"], dict(mean=0., last=0., lstm=2.))
        self.assertEqual(record["revert_streak"], 2)

    def test_revert_uses_same_recent_ten_points_and_better_baseline(self):
        state = dict(item_state(), source="lstm", switched_at=None)
        points = [dict(answer_id=str(i), source="lstm", lstm=2.1, mean=3., last=2.,
                       target=2., saved_at=AT.isoformat()) for i in range(10)]
        state, _, _, _ = decide(state, validation(), points, AT)
        self.assertEqual(state["revert_streak"], 1)
        state, decision, _, _ = decide(state, validation(), points[1:] + [dict(points[-1], answer_id="new")],
                                      AT + timedelta(days=1))
        self.assertEqual(decision, "revert")
        self.assertEqual(state["source"], "baseline")

    def test_good_live_window_resets_degradation_and_nine_points_are_insufficient(self):
        state = dict(item_state(), source="lstm", switched_at=None)
        points = [dict(answer_id=str(i), source="lstm", lstm=4., mean=2., last=2.,
                       target=2., saved_at=AT.isoformat()) for i in range(10)]
        state, _, _, _ = decide(state, validation(), points[:9], AT)
        self.assertEqual(state["revert_streak"], 0)
        state, _, _, _ = decide(state, validation(), points, AT)
        self.assertEqual(state["revert_streak"], 1)
        good = [dict(p, lstm=2.) for p in points]
        state, decision, _, _ = decide(state, validation(), good, AT + timedelta(days=1))
        self.assertEqual(decision, "hold")
        self.assertEqual(state["revert_streak"], 0)
        state, _, _, _ = decide(state, validation(), [dict(p, source="baseline") for p in points],
                               AT + timedelta(days=2))
        self.assertEqual(state["revert_streak"], 0)

    def test_provisional_fatigue_is_excluded_from_insights_and_breakcheck(self):
        at = AT.replace(hour=15, minute=40)
        preferences(self.store, daily_limit=6)
        for day in range(1, 7):
            answer(self.store, f"prior-{day}", at - timedelta(days=day), dict(fatigue=2))
        for offset in (9, 4, 1):
            t = at - timedelta(minutes=offset)
            prediction(self.store, f"provisional-{offset}", t, last=4.5, source="baseline")
            self.store.save("switchbot_observation", f"coverage-{offset}", "p001", t, dict(state="close"))
        rows = self.engine.rows("p001")
        baseline = dict(fatigue=dict(status="available", mean=2., sd=.5))
        provisional = dict(current=dict(fatigue=dict(value=4.5, source="baseline", provisional=True)))
        fatigue = fatigue_change(rows, at, baseline, provisional)
        self.assertFalse(fatigue["acute"])
        self.assertFalse(fatigue["high"])
        queue = QuestionQueue(self.store, InsightEngine(self.store, {}))
        self.assertNotEqual(queue.pending("p001", at)["kind"], "breakCheck")
        # Baseline history must also be ignored when the latest item is LSTM.
        lstm = dict(current=dict(fatigue=dict(value=4.5, source="lstm", provisional=False)))
        self.assertFalse(fatigue_change(rows, at, baseline, lstm)["acute"])
        self.assertTrue(fatigue_change(rows, at, baseline, lstm)["high"])


@unittest.skipUnless(HAS_TORCH, "torch is unavailable")
class LSTMDeliveryTests(DeliveryFixture):
    def test_active_model_validation_matches_sequential_baselines_on_exact_split(self):
        import torch
        from research.data import F, FEATURES, VERSION
        from research.model import HierarchicalLSTM
        self.enable()
        for i in range(10):
            answer(self.store, f"train-{i}", AT - timedelta(days=5, minutes=i), dict(mood=1))
        for i in range(10):
            answer(self.store, f"valid-{i}", AT + timedelta(days=i // 4, minutes=i), dict(mood=3))
        answer(self.store, "test-future", AT + timedelta(days=10), dict(mood=5))
        labels = Timeline(self.engine.rows("p001")).labels
        train_labels = [l for l in labels if l.id.startswith("train")]
        validation_labels = [l for l in labels if l.id.startswith("valid")]
        model = HierarchicalLSTM(hidden=4)
        for parameter in model.parameters():
            parameter.data.zero_()  # exactly 3 on every head
        report = dict(support=dict(current=[10, 0, 0, 0, 0], daily=[0] * 5),
                      train_before="2026-11-01", test_from="2026-11-04",
                      run=dict(splits={name: [dict(id=l.id, at=l.at.isoformat(), available=l.available.isoformat())
                                              for l in group]
                                       for name, group in (("train", train_labels), ("validation", validation_labels))}))
        path = self.engine.directory("p001")
        path.mkdir(parents=True)
        torch.save(dict(weights=model.state_dict(), settings=model.settings, mean=[0.] * F,
                        std=[1.] * F, report=report, version=VERSION, features=FEATURES), path / "active.pt")
        evaluated = validation_evidence(self.engine, "p001", "real", self.engine.rows("p001"),
                                        AT + timedelta(days=4))
        item = evaluated["items"]["current.mood"]
        self.assertEqual(item["train_n"], 10)
        self.assertEqual(item["validation_n"], 10)
        self.assertEqual(item["validation_days"], 3)
        self.assertEqual(item["maes"]["lstm"], 0.)
        self.assertAlmostEqual(item["maes"]["last"], .2)
        self.assertEqual(item["points"][0]["mean"], 1.)
        self.assertAlmostEqual(item["points"][1]["mean"], 13 / 11)
        self.assertEqual(item["points"][1]["last"], 3.)
        # Future target values cannot affect any of these baseline predictions.
        self.assertTrue(all("test-future" not in point["answer_id"] for point in item["points"]))
        with patch("research.delivery.validation_evidence", wraps=validation_evidence):
            first = evaluate_delivery(self.engine, "p001", at=AT + timedelta(days=4))
        self.assertEqual(next(r for r in first if r["item_key"] == "current.mood")["win_streak"], 1)
        # A rejected candidate supplies a newer validation window, while the
        # evaluated weights, support, and training provenance stay active's.
        candidate_report = copy.deepcopy(report)
        candidate_report["support"]["current"][0] = 99
        candidate_report["run"]["splits"]["train"] = candidate_report["run"]["splits"]["validation"]
        atomic_json(path / "status.json", candidate_report)
        for parameter in model.parameters():
            parameter.data.fill_(1.)
        torch.save(dict(weights=model.state_dict()), path / "candidate.pt")
        retained = validation_evidence(self.engine, "p001", "real", self.engine.rows("p001"),
                                       AT + timedelta(days=4))
        self.assertEqual(retained["model_id"], evaluated["model_id"])
        self.assertEqual(retained["items"]["current.mood"]["train_n"], 10)
        self.assertEqual(retained["items"]["current.mood"]["maes"]["lstm"], 0.)
        self.assertEqual(retained["split"]["train"], report["run"]["splits"]["train"])

    def test_incompatible_active_model_is_audited_without_switching(self):
        import torch
        from research.data import FEATURES
        self.enable()
        path = self.engine.directory("p001")
        path.mkdir(parents=True)
        torch.save(dict(version="old", features=FEATURES), path / "active.pt")
        records = evaluate_delivery(self.engine, "p001", at=AT)
        self.assertEqual(len(records), 10)
        self.assertTrue(all(r["validation_status"] == "incompatible_checkpoint"
                            and r["decision"] == "hold" and r["win_streak"] == 0 for r in records))


@unittest.skipUnless(HAS_HTTP, "fastapi/pydantic test environment is unavailable")
class HTTPDeliveryTests(unittest.TestCase):
    def test_setting_event_validation_overview_metadata_and_training_hook(self):
        from fastapi.testclient import TestClient
        from lifelog_server import create_app
        from research.events import ResearchEvent
        with tempfile.TemporaryDirectory() as root:
            store = Store(":memory:")
            app = create_app(dict(ble_enabled=False, allowed_clients=["testclient"],
                                  research_models=root), store)
            event = dict(event_id="enable", participant_id="p001", occurred_at=AT.isoformat(),
                         kind="preferences", payload=dict(baseline_estimates_enabled=True))
            self.assertTrue(ResearchEvent(**event).payload["baseline_estimates_enabled"])
            with self.assertRaises(ValueError):
                ResearchEvent(**dict(event, payload=dict(baseline_estimates_enabled=1)))
            with TestClient(app) as client, patch("research.service.utc", return_value=AT + timedelta(minutes=2)):
                self.assertEqual(client.post("/api/research/p001/events", json=event).status_code, 200)
                answer(store, "first", AT, dict(mood=4))
                result = app.state.research.infer("p001", AT + timedelta(minutes=2))
                store.save("inference", "persisted", "p001", AT + timedelta(minutes=2), result)
                item = client.get("/api/research/p001/overview").json()["prediction"]["current"]["mood"]
                self.assertEqual(item["source"], "baseline")
                self.assertTrue(item["provisional"])
                with patch.object(app.state.research, "_train", return_value=dict(status="no_new_labels")):
                    with patch("research.delivery.evaluate_delivery") as evaluate:
                        response = client.post("/api/research/p001/train", json=dict(data_mode="real"))
                        self.assertEqual(response.status_code, 200)
                        evaluate.assert_called_once_with(app.state.research, "p001", "real")


if __name__ == "__main__":
    unittest.main()
