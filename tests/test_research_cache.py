"""Dependency-light freshness and payload regressions for shared research caching."""

from datetime import timedelta
from heapq import merge
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lifelog.store import Store
from research.api_cache import ResearchCache, row_order
from research.context_service import ContextEngine
from research.data import SCALES, dt
from research.data_modes import rows_for_mode
from research.insights import InsightEngine
from research.discovery_service import DiscoveryInsights
from research.questions import settings


AT = dt("2026-11-02T12:00:00+09:00")
CONTEXT_KINDS = ("context_candidate", "research_event", "context_inference", "context_review")
INSIGHT_KINDS = (
    "session", "narrative", "correction", "research_event", "context_candidate",
    "context_inference", "inference", "step_snapshot", "state_step_snapshot",
)


def answer(at, score=2):
    return dict(
        question_shown_at=at.isoformat(),
        received_at=(at + timedelta(seconds=10)).isoformat(),
        question={"kind": "currentState"},
        answer={"free_text": json.dumps(dict.fromkeys(SCALES, score))},
    )


def populate(store, participant="p001"):
    for days_ago in range(7, 0, -1):
        at = AT - timedelta(days=days_ago)
        store.save("narrative", f"answer-{days_ago}", participant,
                   at + timedelta(seconds=10), answer(at))
        start = at - timedelta(minutes=20)
        store.save("session", f"typing-{days_ago}", participant, start + timedelta(minutes=5), dict(
            session_id=f"typing-{days_ago}", started_at=start.isoformat(),
            ended_at=(start + timedelta(minutes=5)).isoformat(),
            events=[
                dict(time=(start + timedelta(seconds=index)).isoformat(),
                     type="kana", data={"input_mode": "kana"})
                for index in range(30)
            ],
        ))
    old = AT - timedelta(hours=1)
    store.save("context_candidate", "place", participant, old, dict(
        axis="place", class_id="place", anchor=old.isoformat()))
    store.save("context_candidate", "task", participant, old, dict(
        axis="task", class_id="task", anchor=old.isoformat()))
    for axis in ("place", "task"):
        store.save("research_event", f"label-{axis}", participant, old, dict(
            kind="context_label", payload=dict(
                axis=axis, class_id=axis, start=(old - timedelta(minutes=10)).isoformat(),
                end=old.isoformat())))
    store.save("research_event", "name-place", participant, old, dict(
        kind="context_name", payload=dict(axis="place", class_id="place", name="Study")))
    store.save("research_event", "weather", participant, old, dict(
        kind="weather_mood", payload={"weather": "sunny"}))
    current = AT - timedelta(minutes=1)
    store.save("context_inference", "current", participant, current, dict(
        at=current.isoformat(), status="ready",
        place={"id": "place", "name": "Study"}, task={"id": "task"},
        model={"large_report": list(range(30))},
    ))
    store.save("context_review", "review", participant, old, dict(
        day="2026-11-01", created_at=old.isoformat(),
        episodes=[dict(at=old.isoformat(), place={"id": "place"}, task={"id": "task"})],
        original_live=[{"large_report": list(range(30))}],
    ))
    store.save("inference", "prediction", participant, current, dict(
        at=current.isoformat(), status="ready",
        current={"fatigue": {"value": 4, "source": "lstm", "provisional": False}},
        model={"fingerprint": "fixture-model"},
    ))
    store.save("heartbeat", "initial-heartbeat", participant, current, dict(
        logging_enabled=True, questions_enabled=True, interventions_enabled=True))


class ResearchCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "records.sqlite")
        self.addCleanup(self.store.close)
        self.cache = ResearchCache(self.store)
        self.config = {"research_models": str(Path(self.temp.name) / "models")}

    def context_payload(self, cache, engine, at=AT):
        view = cache.records("p001", CONTEXT_KINDS)
        return cache.compute(
            ("context", "p001"), (view.fingerprint, view.time_key(at)),
            lambda: engine.snapshot("p001", at, view.rows),
        )

    def insight_payload(self, cache, engine, at=AT):
        view = cache.records("p001", INSIGHT_KINDS)
        return cache.compute(
            ("insights", "p001"), (view.fingerprint, at),
            lambda: engine.snapshot("p001", at, rows=view.rows),
        )

    def assert_json_equal(self, actual, expected):
        self.assertEqual(
            json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False),
            json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False),
        )

    def test_actual_context_and_insights_payloads_match_disabled_cache(self):
        populate(self.store)
        context = ContextEngine(self.store, self.config)
        insights = InsightEngine(self.store, self.config)
        cached_context = self.context_payload(self.cache, context)
        cached_insights = self.insight_payload(self.cache, insights)
        with patch.object(context, "snapshot", side_effect=AssertionError("cache missed")):
            self.assertIs(self.context_payload(self.cache, context), cached_context)
        with patch.object(insights, "snapshot", side_effect=AssertionError("cache missed")):
            self.assertIs(self.insight_payload(self.cache, insights), cached_insights)

        disabled = ResearchCache(self.store)
        disabled.enabled = False
        uncached_context = self.context_payload(disabled, ContextEngine(self.store, self.config))
        uncached_insights = self.insight_payload(disabled, InsightEngine(self.store, self.config))
        self.assert_json_equal(cached_context, uncached_context)
        self.assert_json_equal(cached_insights, uncached_insights)
        self.assert_json_equal(cached_context, context.snapshot("p001", AT, self.store.rows("p001")))
        self.assert_json_equal(cached_insights, InsightEngine(self.store, self.config).snapshot("p001", AT))
        self.assertNotIn("original_live", cached_context["reviews"][0])
        self.assertEqual(cached_context["history"][0]["model"], {})
        self.assertEqual(cached_insights["baseline"]["fatigue"]["status"], "available")

    def test_new_state_answer_changes_fingerprint_and_recomputes_actual_insights(self):
        populate(self.store)
        engine = InsightEngine(self.store, self.config)
        before_view = self.cache.records("p001", INSIGHT_KINDS)
        before = self.insight_payload(self.cache, engine)
        cutoff = AT - timedelta(minutes=2)
        self.store.save("narrative", "new-state-answer", "p001",
                        cutoff + timedelta(seconds=10), answer(cutoff, score=5))
        after_view = self.cache.records("p001", INSIGHT_KINDS)
        self.assertNotEqual(before_view.fingerprint, after_view.fingerprint)
        after = self.insight_payload(self.cache, engine)
        self.assertIsNot(after, before)
        self.assertNotEqual(before["baseline"]["fatigue"]["mean"], after["baseline"]["fatigue"]["mean"])
        self.assert_json_equal(after, InsightEngine(self.store, self.config).snapshot("p001", AT))

    def test_new_context_name_recomputes_actual_context_payload(self):
        populate(self.store)
        engine = ContextEngine(self.store, self.config)
        before = self.context_payload(self.cache, engine)
        self.store.save("research_event", "renamed-place", "p001", AT, dict(
            kind="context_name", payload=dict(axis="place", class_id="place", name="Office")))
        after = self.context_payload(self.cache, engine)
        self.assertIsNot(after, before)
        self.assertEqual(after["nodes"]["place"][0]["name"], "Office")
        self.assert_json_equal(after, engine.snapshot("p001", AT, self.store.rows("p001")))

    def test_heartbeat_keeps_artifacts_but_updates_heartbeat_view_and_settings(self):
        populate(self.store)
        context = ContextEngine(self.store, self.config)
        insights = InsightEngine(self.store, self.config)
        context_view = self.cache.records("p001", CONTEXT_KINDS)
        insight_view = self.cache.records("p001", INSIGHT_KINDS)
        context_payload = self.context_payload(self.cache, context)
        insight_payload = self.insight_payload(self.cache, insights)
        heartbeat_view = self.cache.records("p001", ("heartbeat",))
        before_preferences = settings(heartbeat_view.rows)
        self.assertTrue(before_preferences["questions_enabled"])

        self.store.save("heartbeat", "next-heartbeat", "p001", AT, dict(
            logging_enabled=False, questions_enabled=False, interventions_enabled=False))
        self.assertIs(self.cache.records("p001", CONTEXT_KINDS), context_view)
        self.assertIs(self.cache.records("p001", INSIGHT_KINDS), insight_view)
        with patch.object(context, "snapshot", side_effect=AssertionError("heartbeat invalidated context")):
            self.assertIs(self.context_payload(self.cache, context), context_payload)
        with patch.object(insights, "snapshot", side_effect=AssertionError("heartbeat invalidated insights")):
            self.assertIs(self.insight_payload(self.cache, insights), insight_payload)
        after_heartbeat = self.cache.records("p001", ("heartbeat",))
        self.assertNotEqual(after_heartbeat.fingerprint, heartbeat_view.fingerprint)
        self.assertEqual(after_heartbeat.rows[0]["id"], "next-heartbeat")
        after_preferences = settings(after_heartbeat.rows)
        self.assertFalse(after_preferences["questions_enabled"])
        self.assertFalse(after_preferences["interventions_enabled"])
        self.assertFalse(after_heartbeat.rows[0]["payload"]["logging_enabled"])
        self.assert_json_equal(after_preferences, settings(self.store.rows("p001")))

    def test_same_sequence_payload_update_invalidates_content(self):
        populate(self.store)
        engine = ContextEngine(self.store, self.config)
        view = self.cache.records("p001", CONTEXT_KINDS)
        before = self.context_payload(self.cache, engine)
        with self.store.lock, self.store.db:
            self.store.db.execute(
                "UPDATE records SET payload=? WHERE participant=? AND kind=? AND id=?",
                (json.dumps(dict(kind="context_name", payload=dict(
                    axis="place", class_id="place", name="Updated"))),
                 "p001", "research_event", "name-place"),
            )
        after_view = self.cache.records("p001", CONTEXT_KINDS)
        self.assertNotEqual(after_view.fingerprint, view.fingerprint)
        after = self.context_payload(self.cache, engine)
        self.assertIsNot(before, after)
        self.assertEqual(after["nodes"]["place"][0]["name"], "Updated")
        self.assert_json_equal(after, engine.snapshot("p001", AT, self.store.rows("p001")))

    def test_same_sequence_state_answer_update_refreshes_insight_engine_cache(self):
        populate(self.store)
        engine = InsightEngine(self.store, self.config)
        before = self.insight_payload(self.cache, engine)
        changed_at = AT - timedelta(days=1)
        with self.store.lock, self.store.db:
            self.store.db.execute(
                "UPDATE records SET payload=? WHERE participant=? AND kind=? AND id=?",
                (json.dumps(answer(changed_at, 5)), "p001", "narrative", "answer-1"),
            )
        after = self.insight_payload(self.cache, engine)
        self.assertIsNot(before, after)
        self.assertNotEqual(before["baseline"]["fatigue"]["mean"], after["baseline"]["fatigue"]["mean"])
        self.assert_json_equal(after, InsightEngine(self.store, self.config).snapshot("p001", AT))

    def test_delete_invalidates_content_and_rebuilds_actual_payload(self):
        populate(self.store)
        engine = ContextEngine(self.store, self.config)
        before = self.context_payload(self.cache, engine)
        with self.store.lock, self.store.db:
            self.store.db.execute("DELETE FROM records WHERE id='name-place'")
        after = self.context_payload(self.cache, engine)
        self.assertIsNot(before, after)
        self.assertIsNone(after["nodes"]["place"][0]["name"])
        self.assert_json_equal(after, engine.snapshot("p001", AT, self.store.rows("p001")))

    def test_discovery_trends_refresh_after_a_same_day_answer(self):
        populate(self.store)
        runtime = SimpleNamespace(
            engine=SimpleNamespace(rows=lambda p, mode: self.store.rows(p)),
            load=lambda p, mode: None,
        )
        engine = DiscoveryInsights(runtime)
        with patch("research.baseline.weekday_trends", side_effect=lambda rows, at: [
            {"answers": sum(r["kind"] == "narrative" for r in rows)}
        ]) as trends:
            first = engine.snapshot("p001", AT)
            self.assertEqual(engine.snapshot("p001", AT)["trends"], first["trends"])
            self.assertEqual(trends.call_count, 1)
            cutoff = AT - timedelta(minutes=1)
            self.store.save("narrative", "new-discovery-answer", "p001",
                            cutoff + timedelta(seconds=10), answer(cutoff, 5))
            second = engine.snapshot("p001", AT)
            self.assertNotEqual(second["trends"], first["trends"])
            self.assertEqual(trends.call_count, 2)

    def test_external_connection_mutations_trigger_content_rescan(self):
        self.store.save("narrative", "state", "p001", AT, answer(AT))
        first = self.cache.records("p001")
        connection = sqlite3.connect(Path(self.temp.name) / "records.sqlite")
        self.addCleanup(connection.close)
        with connection:
            connection.execute("UPDATE records SET payload=? WHERE id='state'", (json.dumps(answer(AT, 5)),))
        updated = self.cache.records("p001")
        self.assertNotEqual(first.fingerprint, updated.fingerprint)
        self.assertEqual(json.loads(updated.rows[0]["payload"]["answer"]["free_text"])["fatigue"], 5)
        with connection:
            connection.execute("INSERT INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                               ("p001", "heartbeat", "external", AT.isoformat(), "{}"))
        inserted = self.cache.records("p001")
        self.assertEqual({record["id"] for record in inserted.rows}, {"state", "external"})
        with connection:
            connection.execute("DELETE FROM records WHERE id='state'")
        deleted = self.cache.records("p001")
        self.assertEqual([record["id"] for record in deleted.rows], ["external"])
        self.assertNotEqual(inserted.fingerprint, deleted.fingerprint)

    def test_append_parses_only_new_rows_and_reuses_existing_payloads(self):
        self.store.save("narrative", "state", "p001", AT, answer(AT))
        with patch("research.api_cache.json.loads", wraps=json.loads) as parse:
            first = self.cache.records("p001")
            self.assertEqual(parse.call_count, 1)
            self.assertIs(self.cache.records("p001"), first)
            self.assertEqual(parse.call_count, 1)
            self.store.save("heartbeat", "next", "p001", AT + timedelta(seconds=1), {})
            appended = self.cache.records("p001")
            self.assertEqual(parse.call_count, 2)
            self.assertIs(appended.rows[-1], first.rows[0])
            self.store.save("heartbeat", "unrelated", "p002", AT, {})
            self.assertIs(self.cache.records("p001"), appended)
            self.assertEqual(parse.call_count, 2)

    def test_time_key_tracks_reception_and_inclusive_context_expiry(self):
        observed = AT - timedelta(seconds=180)
        self.store.save("context_inference", "context", "p001", observed, dict(
            at=observed.isoformat(), received_at=(AT - timedelta(seconds=10)).isoformat(),
            status="ready", place={"id": "place"}, task=None))
        view = self.cache.records("p001", CONTEXT_KINDS)
        before = AT - timedelta(seconds=10, microseconds=1)
        received = AT - timedelta(seconds=10)
        self.assertEqual(view.time_key(before), (0, 0, 0))
        self.assertEqual(view.time_key(received), (0, 1, 0))
        self.assertEqual(view.time_key(AT), (1, 1, 0))
        self.assertEqual(view.time_key(AT + timedelta(microseconds=1)), (1, 1, 1))
        engine = ContextEngine(self.store, self.config)
        self.assertEqual(self.context_payload(self.cache, engine, before)["current"]["status"],
                         "worker_not_running_or_stale")
        self.assertEqual(self.context_payload(self.cache, engine, received)["current"]["status"], "ready")
        self.assertEqual(self.context_payload(self.cache, engine, AT)["current"]["status"], "ready")
        self.assertEqual(self.context_payload(self.cache, engine, AT + timedelta(microseconds=1))["current"]["status"],
                         "worker_not_running_or_stale")

    def test_state_answer_becomes_baseline_evidence_strictly_after_reception(self):
        populate(self.store)
        cutoff = AT - timedelta(minutes=1)
        payload = answer(cutoff, 5)
        payload["received_at"] = AT.isoformat()
        self.store.save("narrative", "just-received", "p001", AT, payload)
        engine = InsightEngine(self.store, self.config)
        at_reception = self.insight_payload(self.cache, engine, AT)
        after_reception = self.insight_payload(self.cache, engine, AT + timedelta(microseconds=1))
        expected = InsightEngine(self.store, self.config).snapshot("p001", AT + timedelta(microseconds=1))
        self.assertNotEqual(at_reception["baseline"]["fatigue"]["mean"],
                            expected["baseline"]["fatigue"]["mean"])
        self.assert_json_equal(after_reception, expected)

    def test_real_sample_combined_record_content_matches_existing_mode_selection(self):
        populate(self.store)
        populate(self.store, "p001-sample")
        for mode, sources in (
            ("real", ("p001",)), ("sample", ("p001-sample",)),
            ("combined", ("p001", "p001-sample")),
        ):
            with self.subTest(mode=mode):
                views = [self.cache.records(source) for source in sources]
                rows = list(merge(*(view.rows for view in views), key=row_order, reverse=True))
                self.assert_json_equal(rows, rows_for_mode(self.store, "p001", mode))
                if mode != "real":
                    actual = InsightEngine(self.store, self.config).snapshot(
                        "p001", AT, rows=rows, data_mode=mode)
                    expected = InsightEngine(self.store, self.config).snapshot(
                        "p001", AT, rows=rows_for_mode(self.store, "p001", mode), data_mode=mode)
                    self.assert_json_equal(actual, expected)

    def test_sources_and_artifacts_have_bounded_recent_use_eviction(self):
        cache = ResearchCache(self.store, limit=2)
        for participant in ("a", "b", "c"):
            self.store.save("heartbeat", participant, participant, AT, {})
        cache.records("a")
        cache.records("b")
        cache.records("a")
        cache.records("c")
        self.assertEqual(list(cache.sources), ["a", "c"])
        first = cache.compute("a", "same", lambda: {"version": 1})
        cache.compute("b", "same", lambda: {"version": 2})
        self.assertIs(cache.compute("a", "same", lambda: None), first)
        cache.compute("c", "same", lambda: {"version": 3})
        self.assertEqual(list(cache.computations), ["a", "c"])
        rebuilt = cache.compute("b", "same", lambda: {"version": 4})
        self.assertEqual(rebuilt, {"version": 4})
        self.assertEqual(len(cache.computations), 2)

    def test_disabled_cache_does_not_reuse_computations_or_record_views(self):
        self.store.save("heartbeat", "one", "p001", AT, {})
        self.cache.enabled = False
        first = self.cache.records("p001")
        second = self.cache.records("p001")
        self.assertIsNot(first, second)
        self.assert_json_equal(first.rows, second.rows)
        self.assertEqual(first.fingerprint, second.fingerprint)
        first_payload = self.cache.compute("same", "same", lambda: {})
        second_payload = self.cache.compute("same", "same", lambda: {})
        self.assertIsNot(first_payload, second_payload)
        self.assertFalse(self.cache.sources)
        self.assertFalse(self.cache.computations)

    def test_portable_store_without_sqlite_counters_reads_current_content(self):
        rows = [dict(kind="narrative", id="state", at=AT.isoformat(), payload=answer(AT))]
        cache = ResearchCache(SimpleNamespace(rows=lambda p: rows))
        first = cache.records("p001")
        rows[0]["payload"] = answer(AT, 5)
        second = cache.records("p001")
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(second.rows[0]["participant"], "p001")
        self.assertEqual(second.rows[0]["seq"], 0)


if __name__ == "__main__":
    unittest.main()
