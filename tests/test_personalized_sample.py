"""Core tests also run with: python3 -m unittest discover -s tests -p test_personalized_sample.py -v"""
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lifelog.store import Store
from research.baseline import conditional
from research.data import BASE, SCALES, Timeline, dt, inference_rows, training_examples
from research.data_modes import rows_for_mode, source_identity
from research.evaluation import chronological_split
from research.personalized_sample import _bind_steps, build_personalized_records, generate_personalized_sample, rng_stream
from research.questions import QuestionQueue
from research.sample_registry import activate_sample, active_generation, generation, record_hash, real_participants
from research.sample_validation import gate_preflight, validate_records, validate_sample
from research.tendency_profile import P001_PROFILE, category_probabilities, load_profile, validate_profile
from research.topology import behavior_map, typing_windows
from research.insights import InsightEngine, step_shortage


def without_identity(value):
    if isinstance(value, dict):
        return {k: without_identity(v) for k, v in value.items()
                if k not in ("participant", "participant_id", "provenance")}
    if isinstance(value, list):
        return [without_identity(v) for v in value]
    return value


def question_content(value):
    value = copy.deepcopy(value)
    if value.get("question"):
        value["question"].pop("id", None)  # Preview IDs are intentionally random UUIDs.
    return value


class PersonalizedSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile()
        cls.rows, cls.details = build_personalized_records(cls.profile, seed=17, weeks=1)
        cls.report = validate_records(cls.rows, cls.profile, cls.details)

    def test_profile_validation_and_aggregate_only_input(self):
        self.assertEqual(load_profile()["measured"]["scope_ratio"], dict(current=12, daily=8))
        for field, value in (("schema_version", "unknown"), ("participant_id", "p001-sample"),
                             ("participant_id", 1), ("raw_answers", ["private text"])):
            p = copy.deepcopy(self.profile)
            p[field] = value
            with self.assertRaises(ValueError):
                validate_profile(p)
        for field, value in (("rho", 1), ("noise_sd", float("nan")), ("time_mixture", [.4, .4, .4])):
            p = copy.deepcopy(self.profile)
            p["assumed"][field] = value
            with self.assertRaises(ValueError):
                validate_profile(p)
        p = copy.deepcopy(self.profile)
        p["measured"]["scales"]["stress"]["value_counts"] = {"1": 18, "2": 1, "3": 1, "4": 0, "5": 0}
        self.assertEqual(validate_profile(p)["measured"]["scales"]["stress"]["value_counts"]["1"], 18)
        q = category_probabilities(p["measured"]["scales"]["stress"])
        self.assertGreater(q[0], .85)
        for scale in self.profile["measured"]["scales"].values():
            q = category_probabilities(scale)
            self.assertAlmostEqual(q.sum(), 1)
            self.assertAlmostEqual(float(q @ np.arange(1, 6)), scale["mean"], places=5)
            self.assertAlmostEqual(float(np.sqrt(q @ (np.arange(1, 6) - scale["mean"]) ** 2)), scale["sd"], places=5)

    def test_determinism_and_component_stream_isolation(self):
        rows, _ = build_personalized_records(self.profile, seed=17, weeks=1)
        self.assertEqual(record_hash(rows), record_hash(self.rows))
        rng_stream(17, "weather").normal(size=10000)
        np.testing.assert_array_equal(rng_stream(17, "scores").normal(size=10),
                                      rng_stream(17, "scores").normal(size=10))
        p = copy.deepcopy(self.profile)
        p["assumed"]["switchbot_enabled"] = True
        changed, _ = build_personalized_records(p, seed=17, weeks=1)
        self.assertEqual(without_identity([r for r in changed if r["kind"] != "switchbot_observation"]),
                         without_identity(self.rows))
        self.assertNotEqual(changed[0]["participant"], self.rows[0]["participant"])

    def test_discrete_values_missingness_and_late_reflection_cutoff(self):
        answers = [r for r in self.rows if r["kind"] == "narrative"]
        daily_targets = []
        late = []
        for r in answers:
            p = r["payload"]
            values = json.loads(p["answer"]["free_text"])
            self.assertEqual(set(values), set(SCALES))
            self.assertTrue(all(type(v) is int and 1 <= v <= 5 for v in values.values()))
            self.assertLessEqual(dt(p["question_shown_at"]), dt(p["answered_at"]))
            self.assertLessEqual(dt(p["answered_at"]), dt(p["received_at"]))
            if p["question"]["kind"] == "nightReflection":
                daily_targets.append(p["target_day"])
                self.assertEqual(str(dt(p["question_shown_at"]).date()), p["target_day"])
                if dt(p["answered_at"]).date() > dt(p["question_shown_at"]).date():
                    late.append(r)
                    label = Timeline([r]).labels[0]
                    self.assertEqual(label.at.date(), dt(p["question_shown_at"]).date())
                    self.assertEqual(label.available, dt(p["received_at"]))
        self.assertTrue(late)
        self.assertEqual(len(daily_targets), len(set(daily_targets)))
        self.assertGreater(self.report["missingness"]["no_answer_day_ratio"], 0)
        self.assertGreater(self.report["missingness"]["incomplete_opportunities"], 0)
        self.assertFalse(any(r["kind"] == "question_issued" for r in self.rows))

    def test_consumer_typing_metrics_reproduce_ranges(self):
        windows = typing_windows(self.rows, max(dt(r["at"]) for r in self.rows) + timedelta(days=1))
        self.assertTrue(windows)
        self.assertTrue(all(22 <= w["inserts"] <= 80 for w in windows))
        for w in windows:
            self.assertLessEqual(w["pause_rate"], .17)
            self.assertGreaterEqual(w["pause_rate"], .03)
            self.assertLessEqual(w["correction_rate"] / 2, .065)
            self.assertGreaterEqual(w["correction_rate"] / 2, .005)
            self.assertTrue(.39 <= np.log1p(w["inserts"]) / 8 <= .55)
        session = next(r for r in self.rows if r["kind"] == "session")
        p = session["payload"]
        self.assertEqual(sum(e["type"] == "kana" for e in p["events"]), p["typing"]["total_typed_characters"])
        # Training's physical key feature is populated too.
        feature = Timeline([session]).features(dt(p["started_at"]), dt(p["ended_at"]) + timedelta(seconds=2))
        self.assertGreater(feature[BASE.index("key_count")], 0)

    def test_step_binding_is_exact_production_replay(self):
        base = copy.deepcopy([r for r in self.rows if r["kind"] not in ("step_snapshot", "state_step_snapshot")])
        for r in base:
            r["payload"].pop("provenance", None)
        derived = _bind_steps(base, self.details["participant"])
        oracle = _bind_steps(base, self.details["participant"], cache_windows=False)
        self.assertEqual(record_hash(derived), record_hash(oracle))
        expected = without_identity([r for r in self.rows if r["kind"] in ("step_snapshot", "state_step_snapshot")])
        self.assertEqual(sorted(without_identity(derived), key=lambda r: (r["kind"], r["id"])),
                         sorted(expected, key=lambda r: (r["kind"], r["id"])))
        self.assertTrue(self.report["steps"]["sane"])
        self.assertEqual(self.report["steps"]["snapshots"], self.report["steps"]["bound_snapshots"])
        states = {r["payload"]["state_id"] for r in derived if r["kind"] == "state_step_snapshot"}
        at = max(dt(r["at"]) for r in base) + timedelta(days=1)
        patterns = behavior_map(typing_windows(base, at), at, rows=base)
        self.assertTrue(states <= {n["id"] for n in patterns["nodes"]} | set(patterns["aliases"]))

    def test_preflight_uses_exact_production_split_and_baselines(self):
        _, examples = training_examples(self.rows)
        tr, va, _, train_end, test_start = chronological_split(examples)
        gates = gate_preflight(self.rows)
        self.assertEqual(gates["train_before"], str(train_end))
        self.assertEqual(gates["test_from"], str(test_start))
        for scope in ("current", "daily"):
            for i, name in enumerate(SCALES):
                self.assertEqual(gates["items"][scope][name]["train"],
                                 sum(l.scope == scope and np.isfinite(l.values[i]) for l, _ in tr))
                self.assertEqual(gates["items"][scope][name]["validation"],
                                 sum(l.scope == scope and np.isfinite(l.values[i]) for l, _ in va))
        at = max(dt(r["payload"].get("received_at", r["at"])) for r in self.rows) + timedelta(seconds=1)
        self.assertEqual(self.report["condition_weighted_daily_baselines"][self.details["participant"]],
                         conditional(self.rows, at, lookback_days=7))
        report = validate_records(self.rows, self.profile, self.details, dict(validation=dict(mean_mae=.01)))
        self.assertTrue(any("implausibly small" in w for w in report["warnings"]))

    def test_registry_idempotence_real_invariance_and_question_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "test.sqlite")
            try:
                at = dt("2026-09-28T12:00:00+09:00")
                store.save("heartbeat", "real-heartbeat", "p001", at, dict(logging_enabled=True, questions_enabled=True))
                before_rows = store.rows("p001")
                before_identity = source_identity(store, "p001", "real")
                before_gates = gate_preflight(before_rows)
                queue = QuestionQueue(store, None)
                insight = dict(fatigue={}, typing={}, exercise={})
                before_question = queue.preview("p001", at, before_rows, insight, before_rows)
                first = generate_personalized_sample(self.profile, store, weeks=1, seed=17)
                second = generate_personalized_sample(self.profile, store, weeks=1, seed=17)
                sample_id = first["sample_participant"]
                self.assertEqual(second["inserted"], 0)
                self.assertEqual(first["record_hash"], second["record_hash"])
                selection = activate_sample(store, "p001", sample_id)
                self.assertEqual(selection, activate_sample(store, "p001", sample_id))
                def concurrent_activation(_):
                    other = Store(Path(tmp) / "test.sqlite")
                    try:
                        return activate_sample(other, "p001", sample_id)
                    finally:
                        other.close()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    self.assertEqual(list(pool.map(concurrent_activation, range(2))), [selection, selection])
                self.assertEqual(active_generation(store, "p001"), sample_id)
                self.assertEqual({r["participant"] for r in rows_for_mode(store, "p001", "sample")}, {sample_id})
                self.assertEqual(store.rows("p001"), before_rows)
                self.assertEqual(source_identity(store, "p001", "real"), before_identity)
                self.assertEqual(gate_preflight(store.rows("p001")), before_gates)
                self.assertEqual(question_content(queue.preview("p001", at, store.rows("p001"), insight, store.rows("p001"))),
                                 question_content(before_question))
                for reserved in ("p001-sample", sample_id, "p001-gabcdef-sample", "other-sample"):
                    if reserved != sample_id:
                        store.save("heartbeat", "reserved", reserved, at, dict(questions_enabled=True))
                    self.assertIsNone(queue.pending(reserved, at))
                    self.assertFalse(any(r["kind"] == "question_issued" for r in store.rows(reserved)))
                self.assertEqual(real_participants(store), ["p001"])
                report = validate_sample(store, sample_id)
                self.assertTrue(all(report["integrity"].values()))
                self.assertTrue(report["steps"]["sane"])
                self.assertTrue(Path(first["manifest"]).is_file())
                # An edited generation cannot be activated or silently regenerated.
                with self.assertRaises(RuntimeError):
                    store.save("heartbeat", "tamper", sample_id, at, {})
                # Simulate out-of-band corruption, bypassing the immutable Store guard.
                with store.db:
                    store.db.execute("INSERT INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                                     (sample_id, "heartbeat", "tamper", at.isoformat(), "{}"))
                with self.assertRaises(RuntimeError):
                    activate_sample(store, "p001", sample_id)
            finally:
                store.close()

    def test_legacy_generation_is_retained(self):
        store = Store(":memory:")
        try:
            at = dt("2026-09-28T12:00:00+09:00")
            store.save("heartbeat", "legacy", "p001-sample", at, {})
            before = store.rows("p001-sample")
            identity = source_identity(store, "p001", "sample")
            result = generate_personalized_sample(self.profile, store, weeks=1, seed=17)
            self.assertEqual(source_identity(store, "p001", "sample"), identity)
            activate_sample(store, "p001", result["sample_participant"])
            self.assertEqual(store.rows("p001-sample"), before)
            self.assertTrue(generation(store, "p001-sample")["manifest"]["legacy"])
        finally:
            store.close()

    def test_combined_examples_never_interleave_sensor_or_personal_history(self):
        sample = copy.deepcopy(self.rows)
        answer = copy.deepcopy(next(r for r in sample if r["kind"] == "narrative"))
        answer["participant"] = "p001"
        answer["id"] = "real-answer"
        answer["payload"]["answer"]["free_text"] = json.dumps(dict.fromkeys(SCALES, 5))
        cutoff = dt(answer["payload"]["question_shown_at"])
        heartbeat = dict(participant="p001", kind="heartbeat", id="real-heartbeat",
                         at=(cutoff - timedelta(seconds=1)).isoformat(), payload=dict(logging_enabled=False))
        real = [answer, heartbeat]
        timelines, combined = training_examples(real + sample)
        self.assertEqual(set(timelines), {"p001", self.details["participant"]})
        for label, spec in combined:
            self.assertTrue(all(r["participant"] == label.source for r in spec.timeline.rows))
        real_spec = next(s for l, s in combined if l.source == "p001")
        expected = Timeline(real).sample(cutoff)
        for key in expected:
            np.testing.assert_array_equal(real_spec[key], expected[key])
        selected, source = inference_rows(real + sample, cutoff, "p001")
        self.assertEqual(source, "p001")
        self.assertEqual(selected, real)

    def test_gate_test_dense_seven_days(self):
        rows, details = build_personalized_records(self.profile, weeks=1, scenario="gate-test")
        report = validate_records(rows, self.profile, details)
        self.assertEqual(details["identity"]["days"], 7)
        self.assertTrue(report["gates"]["training_start_eligible"])
        self.assertTrue(report["gates"]["any_display_eligible"])
        self.assertEqual(report["counts"]["scopes"]["daily"], 7)
        self.assertEqual(report["counts"]["scopes"]["current"], 84)

    def test_sample_exercise_uses_the_sample_owner(self):
        bindings = [r for r in self.rows if r["kind"] == "state_step_snapshot"]
        eligible = []
        for row in bindings:
            p = row["payload"]
            past = [r for r in bindings if r["payload"]["state_id"] == p["state_id"]
                    and dt(r["at"]).date() < dt(row["at"]).date()]
            if past and dt(row["payload"]["received_at"]).date() == dt(row["at"]).date():
                eligible.append(row)
        self.assertTrue(eligible)
        row = max(eligible, key=lambda r: dt(r["at"]))
        at = dt(row["payload"]["received_at"]) + timedelta(seconds=1)
        store = Store(":memory:")
        try:
            insights = InsightEngine(store, dict(exercise_baseline_provisional_days=1, exercise_baseline_days=1))
            result = insights.snapshot("p001", at, rows=self.rows, data_mode="sample")
            own = step_shortage(self.rows, at, participant=self.details["participant"],
                               state_aliases=result["patterns"]["aliases"], provisional_days=1, formal_days=1)
            self.assertEqual(result["exercise"], own)
            self.assertEqual(own["status"], "available")
            self.assertNotEqual(own, step_shortage(self.rows, at, participant="p001"))
        finally:
            store.close()


class EightWeekGenerationTests(unittest.TestCase):
    def test_eight_weeks_temp_db_validate_activate_and_real_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "eight-weeks.sqlite")
            try:
                identity = source_identity(store, "p001", "real")
                before_gates = gate_preflight([])
                result = generate_personalized_sample(P001_PROFILE, store, weeks=8, seed=17, require_gates=True)
                sample_id = result["sample_participant"]
                report = validate_sample(store, sample_id)
                self.assertTrue(all(report["integrity"].values()))
                self.assertEqual(result["days"], 56)
                self.assertTrue(report["gates"]["training_start_eligible"])
                self.assertTrue(report["gates"]["display_eligible"])
                self.assertTrue(report["gates"]["trend_eligible"])
                self.assertAlmostEqual(report["timing"]["answers_per_day"], 1.65, delta=.25)
                self.assertAlmostEqual(report["missingness"]["no_answer_day_ratio"], .25, delta=.15)
                self.assertTrue(report["steps"]["sane"])
                self.assertGreater(report["steps"]["bound_snapshots"], 0)
                activate_sample(store, "p001", sample_id)
                self.assertEqual(source_identity(store, "p001", "real"), identity)
                self.assertEqual(gate_preflight(store.rows("p001")), before_gates)
                self.assertEqual(store.rows("p001"), [])
                self.assertEqual({r["participant"] for r in rows_for_mode(store, "p001", "combined")}, {sample_id})
            finally:
                store.close()


ML_AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "fastapi", "httpx", "pydantic"))


@unittest.skipUnless(ML_AVAILABLE, "torch/server dependencies are unavailable")
class PersonalizedAcceptanceTests(unittest.TestCase):
    def test_eight_weeks_validate_activate_train_and_infer_real_untouched(self):
        from research.service import Engine, participants
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "test.sqlite")
            try:
                engine = Engine(store, dict(research_models=str(Path(tmp) / "models"), research_epochs=1))
                identity = source_identity(store, "p001", "real")
                real_gates = gate_preflight(store.rows("p001"))
                result = generate_personalized_sample(P001_PROFILE, store, weeks=8, seed=17)
                sample_id = result["sample_participant"]
                self.assertTrue(all(validate_sample(store, sample_id)["integrity"].values()))
                activate_sample(store, "p001", sample_id)
                self.assertEqual(participants(store), [])
                self.assertIsNone(engine.pending(sample_id))
                report = engine.train("p001", "sample")
                self.assertEqual(report["status"], "trained")
                self.assertTrue((Path(tmp) / "models" / "p001" / "sample" / "active.pt").exists())
                supported = [(scope, name) for scope in ("current", "daily") for i, name in enumerate(SCALES)
                             if report["support"][scope][i] >= 10 and report["validation"][scope][name]["n"] >= 3]
                self.assertTrue(supported)
                at = max(dt(r["at"]) for r in store.rows(sample_id) if r["kind"] == "heartbeat") + timedelta(seconds=1)
                prediction = engine.infer("p001", at, "sample")
                scope, name = supported[0]
                self.assertEqual(prediction[scope][name]["status"], "estimated")
                self.assertTrue(1 <= prediction[scope][name]["value"] <= 5)
                self.assertEqual(engine.infer("p001", data_mode_="sample")[scope][name]["status"], "estimated")
                self.assertEqual(engine.train("p001", "real")["status"], "insufficient_labels")
                self.assertEqual(store.rows("p001"), [])
                self.assertEqual(source_identity(store, "p001", "real"), identity)
                self.assertEqual(gate_preflight(store.rows("p001")), real_gates)
                self.assertEqual(question_content(engine.questions.preview("p001", at, [], dict(fatigue={}, typing={}, exercise={}), [])),
                                 question_content(engine.questions.preview("p001", at, engine.rows("p001"), dict(fatigue={}, typing={}, exercise={}), engine.rows("p001"))))
                other = generate_personalized_sample(P001_PROFILE, store, weeks=1, seed=18)
                activate_sample(store, "p001", other["sample_participant"])
                self.assertEqual(engine.infer("p001", at, "sample")["status"], "generation_changed")
            finally:
                store.close()

    def test_discovery_combined_uses_separate_source_examples(self):
        from research.discovery_learning import examples_for
        profile = load_profile()
        rows, _ = build_personalized_records(profile, weeks=1)
        real = copy.deepcopy(rows)
        for row in real:
            row["participant"] = "p001"
            if row["kind"] == "session":
                row["payload"]["context"]["step_count"] = 50000
        at = max(dt(r["at"]) for r in rows) + timedelta(hours=2)
        _, examples, _ = examples_for(real + rows, at)
        self.assertTrue(examples)
        for e in examples:
            self.assertEqual({r["participant"] for r in e["sample"].timeline.rows}, {e["source"]})

    def test_discovery_sample_train_infer_and_real_remains_insufficient(self):
        from research.service import Engine
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "discovery.sqlite")
            try:
                generated = generate_personalized_sample(P001_PROFILE, store, scenario="gate-test", seed=17)
                activate_sample(store, "p001", generated["sample_participant"])
                engine = Engine(store, dict(research_models=str(Path(tmp) / "models"),
                    research_architecture="discovery", discovery_epochs=1, discovery_hidden=8, context_device="cpu"))
                trained = engine.train("p001", "sample")
                self.assertEqual(trained["status"], "trained")
                estimated = engine.infer("p001", data_mode_="sample")
                self.assertTrue(any(item["status"] == "estimated" for scope in ("current", "daily")
                                    for item in (estimated[scope] or {}).values()))
                self.assertEqual(engine.train("p001", "real")["status"], "insufficient_observations")
                self.assertEqual(store.rows("p001"), [])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
