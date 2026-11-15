"""Participant-isolated samples and strict, global chronological boundaries."""
from dataclasses import dataclass
from collections import defaultdict
from .data import Label, Timeline, dt, midnight


@dataclass
class Example:
    participant: str
    label: Label
    sample: object

    @property
    def key(self):
        return (self.participant, self.label.id, self.label.scope)


def cohort(rows, text_model=None):
    grouped = defaultdict(list)
    for row in rows:
        participant = row.get('participant') or row['payload'].get('participant_id')
        if not participant:
            raise ValueError('Every experimental row requires a participant identifier')
        grouped[str(participant)].append(row)
    timelines = {p: Timeline(items, text_model=text_model) for p, items in sorted(grouped.items())}
    examples = [Example(p, label, sample) for p, timeline in timelines.items()
                for label, sample in timeline.examples()]
    examples.sort(key=lambda e: (e.label.at, e.participant, e.label.id, e.label.scope))
    return timelines, examples


def chronological_split(examples, train_before=None, test_from=None, pretrain_before=None):
    if (train_before is None) != (test_from is None):
        raise ValueError('Supply both train_before and test_from, or neither')
    if train_before is None:
        days = sorted({midnight(e.label.at) for e in examples if e.label.scope in ('current', 'daily')})
        if len(days) < 5:
            raise ValueError('At least five distinct labeled days are required')
        a = max(1, int(len(days) * .6))
        b = min(max(a + 1, int(len(days) * .8)), len(days) - 1)
        train_before, test_from = days[a], days[b]
    train_before, test_from = dt(train_before), dt(test_from)
    pretrain_before = dt(pretrain_before) if pretrain_before else train_before
    if not pretrain_before <= train_before < test_from:
        raise ValueError('Require pretrain_before <= train_before < test_from')
    train = [e for e in examples if e.label.at < train_before and e.label.available < train_before]
    valid = [e for e in examples if train_before <= e.label.at < test_from and e.label.available < test_from]
    test = [e for e in examples if e.label.at >= test_from]
    if not all(any(e.label.scope in ('current', 'daily') for e in part) for part in (train, valid, test)):
        raise ValueError('Train, validation and test each require state labels')
    metadata = dict(train_before=train_before.isoformat(), test_from=test_from.isoformat(),
                    pretrain_before=pretrain_before.isoformat(), timezone='Asia/Tokyo',
                    boundary='exclusive: event time AND label availability < training cutoff',
                    normalization='training samples only',
                    conditioning='strictly earlier received answers; evaluation weights stay frozen unless prequential is enabled')
    return train, valid, test, metadata


def eligible_updates(examples, used, at):
    # Includes delayed pre-training-period answers once they actually arrive.
    return sorted((e for e in examples if e.key not in used and e.label.at < at and e.label.available < at),
                  key=lambda e: (e.label.available, e.label.at, e.participant, e.label.id, e.label.scope))


def sensor_timelines(timelines):
    # SSL never consumes questionnaire answers or correction-derived activity boundaries.
    return {p: Timeline([r for r in timeline.rows if r['kind'] not in ('narrative', 'correction')])
            for p, timeline in timelines.items()}
