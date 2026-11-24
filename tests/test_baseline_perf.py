"""Preserve the statistical result while reusing bootstrap week blocks."""

from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from research import baseline
from research.data import dt


def scan_reference(labels, at):
    """Original bootstrap: find each selected week's samples by a fresh scan."""
    cells = defaultdict(list)
    for label in labels:
        cells[(label.at.date(), baseline.period(label.at), label.activity)].append(
            float(label.values[1])
        )
    samples = [(d, p, a, float(np.mean(v))) for (d, p, a), v in cells.items()]
    result = []
    alpha = 0.05 / 28
    for day in range(7):
        for part in ("morning", "afternoon", "evening", "night"):
            target = [s for s in samples if s[0].weekday() == day and s[1] == part]
            reference = [
                s for s in samples if s[0].weekday() != day and s[1] == part
            ]
            strata = {s[2] for s in target} & {s[2] for s in reference}
            target = [s for s in target if s[2] in strata]
            reference = [s for s in reference if s[2] in strata]
            td, rd = {s[0] for s in target}, {s[0] for s in reference}
            if len(td) < 4 or len(rd) < 8:
                continue

            def effect(t, r):
                terms = []
                for activity in strata:
                    tv = [s[3] for s in t if s[2] == activity]
                    rv = [s[3] for s in r if s[2] == activity]
                    if tv and rv:
                        terms.append((float(np.mean(tv) - np.mean(rv)), len(tv)))
                return (
                    sum(value * weight for value, weight in terms)
                    / sum(weight for _, weight in terms)
                    if terms else None
                )

            difference = effect(target, reference)
            weeks = sorted(
                {date - timedelta(days=date.weekday())
                 for date, _, _, _ in target + reference}
            )
            rng = np.random.default_rng(1700 + day)
            boot = []
            for _ in range(2000):
                selected = rng.choice(len(weeks), len(weeks), replace=True)
                t, r = [], []
                for index in selected:
                    week = weeks[int(index)]
                    t.extend(
                        s for s in target
                        if s[0] - timedelta(days=s[0].weekday()) == week
                    )
                    r.extend(
                        s for s in reference
                        if s[0] - timedelta(days=s[0].weekday()) == week
                    )
                value = effect(t, r)
                if value is not None:
                    boot.append(value)
            if len(boot) < 1000:
                continue
            lo, hi = [float(v) for v in np.quantile(boot, [alpha / 2, 1 - alpha / 2])]
            result.append(dict(
                weekday=day, period=part, difference=difference, interval=[lo, hi],
                target_days=len(td), reference_days=len(rd), weeks=len(weeks),
                status="higher" if lo > 0 else "lower" if hi < 0 else "uncertain",
                note="exploratory association, activity-stratified; week bootstrap; 28-comparison correction",
            ))
    return result


def test_weekday_bootstrap_matches_original_scan_exactly(monkeypatch):
    at = dt("2026-10-29T16:40:00+09:00")
    start = dt("2026-08-10T14:00:00+09:00")
    labels = []
    for week in range(10):
        for day in (0, 1):
            for activity in (-1, 2, 4):
                # Missing cells exercise partially shared strata and weeks.
                if (week + day + activity) % 7 == 0:
                    continue
                cutoff = start + timedelta(weeks=week, days=day, minutes=activity)
                for report in range(2):
                    labels.append(SimpleNamespace(
                        at=cutoff,
                        activity=activity,
                        values=np.array([np.nan, 1.13 + (week % 3) / 7
                                         + day / 5 + activity / 31 + report / 13]),
                    ))
    # Week blocks must preserve ordering even when inputs arrive out of order.
    labels = labels[::2] + list(reversed(labels[1::2]))
    monkeypatch.setattr(baseline, "current_answers", lambda rows, cutoff: labels)
    expected = scan_reference(labels, at)
    assert len(expected) == 2
    assert baseline.weekday_trends([], at) == expected
