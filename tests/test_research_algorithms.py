"""Reference regressions for the research API's indexed computations."""

from datetime import timedelta
import json
import random
import unittest
from unittest.mock import patch

from research.data import ACTIVITIES, SCALES, Timeline, dt
from research import topology


AT = dt("2026-11-02T12:00:00+09:00")


def row(kind, key, at, payload, seq=0):
    return dict(kind=kind, id=key, at=at.isoformat(), payload=payload,
                participant="p001", seq=seq)


class ScanTimeline(Timeline):
    """Pre-index implementation, retained as an independent precedence oracle."""

    def activity_at(self, at):
        for available, payload in reversed(self.corrections):
            if (available <= at and dt(payload["start"]) <= at < dt(payload["end"])
                    and payload.get("activity") in ACTIVITIES):
                return ACTIVITIES.index(payload["activity"])
        prior = [label for label in self.labels
                 if label.scope == "activity" and label.available <= at
                 and 0 <= (at - label.at).total_seconds() <= 1800]
        return prior[-1].activity if prior else -1


class ResearchAlgorithmTests(unittest.TestCase):
    def test_activity_index_matches_original_for_delays_ties_and_corrections(self):
        rng = random.Random(29)
        records = []
        for index in range(240):
            cutoff = AT + timedelta(minutes=rng.randrange(-180, 60))
            received = cutoff + timedelta(minutes=rng.randrange(0, 60))
            question = rng.choice(("activityCheck", "currentState", "nightReflection"))
            answer = (dict(label=rng.choice(ACTIVITIES)) if question == "activityCheck"
                      else dict(free_text=json.dumps(dict.fromkeys(SCALES, 3))))
            records.append(row("narrative", str(index), received, dict(
                question_shown_at=cutoff.isoformat(), received_at=received.isoformat(),
                question=dict(kind=question), answer=answer), seq=index))
        for index in range(30):
            start = AT + timedelta(minutes=rng.randrange(-150, 30))
            end = start + timedelta(minutes=rng.randrange(1, 60))
            received = end + timedelta(minutes=rng.randrange(-10, 30))
            records.append(row("correction", f"c{index}", received, dict(
                start=start.isoformat(), end=end.isoformat(), received_at=received.isoformat(),
                activity=rng.choice(ACTIVITIES), scores={"fatigue": 3}), seq=300 + index))
        indexed, reference = Timeline(records), ScanTimeline(records)
        self.assertEqual(
            [(label.id, label.at, label.available, label.activity) for label in indexed.labels],
            [(label.id, label.at, label.available, label.activity) for label in reference.labels],
        )
        for seconds in range(-10801, 7202, 137):
            cutoff = AT + timedelta(seconds=seconds)
            self.assertEqual(indexed.activity_at(cutoff), reference.activity_at(cutoff))

    def test_activity_index_has_inclusive_30_minute_boundaries_without_label_scan(self):
        cutoff = AT - timedelta(minutes=30)
        timeline = Timeline([row("narrative", "activity", cutoff, dict(
            question_shown_at=cutoff.isoformat(), question=dict(kind="activityCheck"),
            answer=dict(label="study")))])

        class NoScan(list):
            def __iter__(self):
                raise AssertionError("activity lookup scanned every state answer")

        timeline.labels = NoScan(timeline.labels)
        self.assertEqual(timeline.activity_at(cutoff), ACTIVITIES.index("study"))
        self.assertEqual(timeline.activity_at(AT), ACTIVITIES.index("study"))
        self.assertEqual(timeline.activity_at(AT + timedelta(microseconds=1)), -1)
        self.assertEqual(timeline.activity_at(cutoff - timedelta(microseconds=1)), -1)

    def test_topology_skips_unrelated_records_and_preserves_context_replay(self):
        start = AT - timedelta(minutes=20)
        session = row("session", "typing", start + timedelta(minutes=5), dict(
            started_at=start.isoformat(), ended_at=(start + timedelta(minutes=5)).isoformat(),
            events=[dict(time=(start + timedelta(seconds=index)).isoformat(),
                         type="kana", data={"input_mode": "kana"}) for index in range(30)]))
        context = [
            row("context_candidate", "place", start, dict(
                axis="place", class_id="old_place", anchor=start.isoformat())),
            row("context_candidate", "task", start, dict(
                axis="task", class_id="task", anchor=start.isoformat())),
            row("context_inference", "context", start + timedelta(minutes=10), dict(
                place={"id": "old_place"}, task={"id": "task"})),
            row("research_event", "merge", AT - timedelta(minutes=1), dict(
                kind="context_merge", payload=dict(
                    axis="place", source_id="old_place", target_id="place"))),
            row("research_event", "name", AT - timedelta(seconds=30), dict(
                kind="context_name", payload=dict(axis="place", class_id="place", name="Study"))),
        ]
        unrelated = [
            row(kind, str(index), AT - timedelta(seconds=index), {"logging_enabled": True})
            for index, kind in enumerate(("heartbeat", "sensor", "inference", "narrative"))
        ]
        delayed = row("session", "delayed", start, dict(session["payload"],
                      received_at=(AT + timedelta(seconds=1)).isoformat()))
        full = unrelated + [session, delayed] + context
        expected_windows = topology.typing_windows([session], AT)
        expected_map = topology.behavior_map(expected_windows, AT, rows=context)
        observed_kinds = []
        original = topology.available_rows

        def available(records, at):
            records = list(records)
            observed_kinds.append({record["kind"] for record in records})
            return original(records, at)

        with patch.object(topology, "available_rows", available):
            actual_windows = topology.typing_windows(full, AT)
            actual_map = topology.behavior_map(actual_windows, AT, rows=full)
        self.assertEqual(actual_windows, expected_windows)
        self.assertEqual(actual_map, expected_map)
        self.assertEqual(observed_kinds, [
            {"session"}, {"context_candidate", "context_inference", "research_event"}])
        self.assertEqual(actual_map["nodes"][0]["place"], {"id": "place", "name": "Study"})


if __name__ == "__main__":
    unittest.main()
