"""LazySample must materialize each example only once per run."""
from datetime import timedelta
import json
import unittest

from research.data import Timeline, SCALES, dt, LazySample


def answer(key, at, received):
    return dict(kind='narrative', id=key, at=(at + timedelta(seconds=10)).isoformat(),
                participant='p001',
                payload=dict(question_shown_at=at.isoformat(),
                             received_at=received.isoformat(),
                             question={'kind': 'currentState'},
                             answer={'free_text': json.dumps(dict.fromkeys(SCALES, 3))}))


class LazySampleCacheTests(unittest.TestCase):
    def test_repeated_access_materializes_once(self):
        at = dt('2026-11-01T10:00:00+09:00')
        timeline = Timeline([answer('a', at, at + timedelta(minutes=1))])
        examples = timeline.examples()
        self.assertEqual(len(examples), 1)
        label, sample = examples[0]
        self.assertIsInstance(sample, LazySample)

        calls = []
        original = timeline.sample

        def counting(at_, scope='current'):
            calls.append((at_, scope))
            return original(at_, scope)

        timeline.sample = counting
        try:
            first = sample.materialize()
            second = sample.materialize()
            for key in sample.keys_order:
                sample[key]
            dict(sample)
        finally:
            timeline.sample = original

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
