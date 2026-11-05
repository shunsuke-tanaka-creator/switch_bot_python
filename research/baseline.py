"""Causal, day-weighted personal comparisons; no model predictions become labels."""

from collections import defaultdict
from datetime import timedelta
import math
import numpy as np
from .data import Timeline, dt, SCALES


def period(t):
    return (
        "morning"
        if 5 <= t.hour < 12
        else "afternoon"
        if t.hour < 17
        else "evening"
        if t.hour < 22
        else "night"
    )


def current_answers(rows, at):
    """One last available report per cutoff, so a correction is not a second vote."""
    result = {}
    for l in sorted(Timeline(rows).labels, key=lambda x: x.available):
        if l.scope == "current" and l.available < at and l.at < at:
            result[l.at] = l
    return list(result.values())


def conditional(rows, at, activity=-1, lookback_days=56, min_days=5):
    at = dt(at)
    labels = [
        l
        for l in current_answers(rows, at)
        if l.at >= at - timedelta(days=lookback_days)
    ]
    output = {}
    for j, name in enumerate(SCALES):
        daily = defaultdict(list)
        for l in labels:
            if not np.isfinite(l.values[j]):
                continue
            # Close conditions carry more weight; repeated reports on one day do not dominate.
            w = (
                1.0
                + 2 * (period(l.at) == period(at))
                + 2 * (l.at.weekday() == at.weekday())
            )
            w += 3 * (period(l.at) == period(at) and l.at.weekday() == at.weekday())
            w += 2 * (activity >= 0 and l.activity == activity)
            daily[l.at.date()].append((float(l.values[j]), w))
        values = [
            (
                sum(v * w for v, w in vs) / sum(w for _, w in vs),
                sum(w for _, w in vs) / len(vs),
            )
            for vs in daily.values()
        ]
        if len(values) < min_days:
            output[name] = dict(
                status="insufficient_data", days=len(values), mean=None, sd=None
            )
            continue
        v, w = np.array(values).T
        mu = float(np.average(v, weights=w))
        sd = max(0.5, math.sqrt(float(np.average((v - mu) ** 2, weights=w))))
        output[name] = dict(
            status="available",
            days=len(values),
            mean=mu,
            sd=sd,
            method="condition-weighted daily mean; sparse conditions borrow global history",
        )
    return output


def weekday_trends(rows, at, min_weeks=6, min_target_days=4, min_reference_days=8):
    """Exploratory weekday × period effects. Week-block bootstrap and Bonferroni CIs.

    Use one observation per date/period/activity; compare within known activity.
    Unknown activity is its own stratum, never silently matched to a known one.
    """
    at = dt(at)
    labels = [
        l for l in current_answers(rows, at) if at - timedelta(days=84) <= l.at < at
    ]
    if not labels:
        return []
    if (at - min(l.at for l in labels)).days < min_weeks * 7:
        return []
    cells = defaultdict(list)
    for l in labels:
        if np.isfinite(l.values[1]):
            cells[(l.at.date(), period(l.at), l.activity)].append(float(l.values[1]))
    samples = [(d, p, a, float(np.mean(v))) for (d, p, a), v in cells.items()]
    results = []
    # Fixed exploration family: seven weekdays × four periods (28 comparisons).
    alpha = 0.05 / 28
    for day in range(7):
        for part in ("morning", "afternoon", "evening", "night"):
            target = [s for s in samples if s[0].weekday() == day and s[1] == part]
            ref = [s for s in samples if s[0].weekday() != day and s[1] == part]
            strata = {s[2] for s in target} & {s[2] for s in ref}
            target = [s for s in target if s[2] in strata]
            ref = [s for s in ref if s[2] in strata]
            td, rd = {s[0] for s in target}, {s[0] for s in ref}
            if len(td) < min_target_days or len(rd) < min_reference_days:
                continue

            def effect(t, r):
                terms = []
                for a in strata:
                    tv = [s[3] for s in t if s[2] == a]
                    rv = [s[3] for s in r if s[2] == a]
                    if tv and rv:
                        terms.append((float(np.mean(tv) - np.mean(rv)), len(tv)))
                return (
                    sum(v * w for v, w in terms) / sum(w for _, w in terms)
                    if terms
                    else None
                )

            diff = effect(target, ref)
            weeks = sorted(
                {d - timedelta(days=d.weekday()) for d, _, _, _ in target + ref}
            )
            rng = np.random.default_rng(1700 + day)
            boot = []
            for _ in range(2000):
                selected = rng.choice(len(weeks), len(weeks), replace=True)
                t = []
                r = []
                for index in selected:
                    week = weeks[int(index)]
                    t.extend(
                        s
                        for s in target
                        if s[0] - timedelta(days=s[0].weekday()) == week
                    )
                    r.extend(
                        s for s in ref if s[0] - timedelta(days=s[0].weekday()) == week
                    )
                e = effect(t, r)
                if e is not None:
                    boot.append(e)
            if len(boot) < 1000:
                continue
            lo, hi = [float(x) for x in np.quantile(boot, [alpha / 2, 1 - alpha / 2])]
            results.append(
                dict(
                    weekday=day,
                    period=part,
                    difference=diff,
                    interval=[lo, hi],
                    target_days=len(td),
                    reference_days=len(rd),
                    weeks=len(weeks),
                    status="higher" if lo > 0 else "lower" if hi < 0 else "uncertain",
                    note="exploratory association, activity-stratified; week bootstrap; 28-comparison correction",
                )
            )
    return results
