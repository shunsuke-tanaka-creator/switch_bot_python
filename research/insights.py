"""Personal change detectors and automatic topology alongside the three LSTMs."""

from datetime import timedelta
import numpy as np
from .data import dt
from .baseline import conditional, weekday_trends
from .topology import available_rows, typing_windows, behavior_map, place_map


def fatigue_change(rows, at, base, prediction, delta_min=1.0, z_min=2.0):
    reference = base["fatigue"]
    value = ((prediction or {}).get("current") or {}).get("fatigue", {}).get("value")
    if reference["status"] != "available" or value is None:
        return dict(status="insufficient_data", acute=False, high=False)
    delta = value - reference["mean"]
    z = delta / reference["sd"]
    buckets = {}
    for r in available_rows(rows, at):
        if r["kind"] != "inference" or dt(r["at"]) < at - timedelta(minutes=15):
            continue
        p = r["payload"]
        v = (p.get("current") or {}).get("fatigue", {}).get("value")
        if p.get("status") != "ready" or v is None:
            continue
        t = dt(r["at"])
        bucket = t.replace(minute=(t.minute // 5) * 5, second=0, microsecond=0)
        if bucket + timedelta(minutes=5) > at:
            continue
        buckets[bucket] = (float(v), p.get("model", {}).get("fingerprint", "legacy"))
    times = sorted(buckets)
    # Each bucket requires its own recent observed typing/sensor coverage, not repeated idle predictions.
    observed = [
        r
        for r in available_rows(rows, at)
        if r["kind"] in ("session", "switchbot_observation", "switchbot")
    ]
    valid = []
    for t in times:
        if not any(t <= dt(r["at"]) < t + timedelta(minutes=5) for r in observed):
            continue
        b = conditional(rows, t, min_days=5)["fatigue"]
        if (
            b["status"] == "available"
            and buckets[t][0] - b["mean"] >= delta_min
            and (buckets[t][0] - b["mean"]) / b["sd"] >= z_min
        ):
            valid.append(t)
    persistent = any(
        (b - a).total_seconds() == 300 and buckets[a][1] == buckets[b][1]
        for a, b in zip(valid, valid[1:])
    )
    return dict(
        status="available",
        value=value,
        expected=reference["mean"],
        difference=delta,
        z=z,
        acute=bool(delta >= delta_min and z >= z_min and persistent),
        high=bool(value >= 4),
        evidence_windows=len(valid),
        note="research thresholds; not calibrated probabilities",
    )


def typing_change(windows, at, minimum=100):
    if not windows:
        return dict(status="insufficient_data", unusual=False)
    current = windows[-1]
    if (at - current["end"]).total_seconds() > 900 or current["inserts"] < minimum:
        return dict(status="insufficient_data", unusual=False)
    past = [
        w
        for w in windows[:-1]
        if w["device"] == current["device"]
        and w["mode"] == current["mode"]
        and w["inserts"] >= minimum
        and w["at"].date() < current["at"].date()
    ]
    if len(past) < 20 or len({w["at"].date() for w in past}) < 5:
        return dict(
            status="insufficient_data", unusual=False, baseline_windows=len(past)
        )
    threshold = max(
        float(np.quantile([w["correction_rate"] for w in past], 0.99)), 0.02
    )
    previous = next(
        (
            w
            for w in reversed(windows[:-1])
            if w["device"] == current["device"] and w["mode"] == current["mode"]
        ),
        None,
    )
    persistent = bool(
        previous
        and previous["end"] == current["at"]
        and previous["inserts"] >= minimum
        and previous["correction_rate"] > threshold
    )
    return dict(
        status="available",
        unusual=bool(current["correction_rate"] > threshold and persistent),
        correction_rate=current["correction_rate"],
        threshold=threshold,
        inserts=current["inserts"],
        at=current["end"].isoformat(),
        evidence_id=current["id"],
        baseline_windows=len(past),
        baseline_days=len({w["at"].date() for w in past}),
        note="input corrections, not confirmed mistakes",
    )


class InsightEngine:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.cache = {}
        self.trend_cache = {}

    def snapshot(self, p, at, prediction=None):
        at = dt(at)
        rows = available_rows(self.store.rows(p), at)
        source = [
            r
            for r in rows
            if r["kind"] in ("session", "narrative", "correction", "research_event")
        ]
        signature = tuple((r.get("seq"), r["kind"], r["id"]) for r in source)
        bucket = at.replace(minute=at.minute // 5 * 5, second=0, microsecond=0)
        key = (p, signature, bucket)
        if self.cache.get(p, {}).get("key") == key:
            fixed = self.cache[p]["data"]
        else:
            windows = typing_windows(rows, at)
            trend_key = (
                p,
                tuple(
                    (r.get("seq"), r["id"])
                    for r in source
                    if r["kind"] in ("narrative", "correction")
                ),
                at.date(),
            )
            if self.trend_cache.get(p, {}).get("key") != trend_key:
                self.trend_cache[p] = dict(
                    key=trend_key,
                    data=weekday_trends(
                        rows, at, min_weeks=self.config.get("trend_min_weeks", 6)
                    ),
                )
            fixed = dict(
                baseline=conditional(rows, at),
                trends=self.trend_cache[p]["data"],
                typing=typing_change(
                    windows, at, minimum=self.config.get("typing_min_inserts", 100)
                ),
                patterns=behavior_map(
                    windows,
                    at,
                    radius=self.config.get("pattern_radius", 0.35),
                    rows=source,
                ),
                places=place_map(rows, at),
                weather=[
                    dict(
                        at=r["at"],
                        weather=r["payload"]["payload"]["weather"],
                        source="self_report",
                    )
                    for r in source
                    if r["kind"] == "research_event"
                    and r["payload"]["kind"] == "weather_mood"
                ][-200:],
            )
            self.cache[p] = dict(key=key, data=fixed)
        if prediction is None:
            latest = next(
                (
                    r
                    for r in reversed(rows)
                    if r["kind"] == "inference"
                    and (at - dt(r["at"])).total_seconds() <= 180
                ),
                None,
            )
            prediction = latest["payload"] if latest else None
        fatigue = fatigue_change(
            rows,
            at,
            fixed["baseline"],
            prediction,
            delta_min=self.config.get("fatigue_delta", 1.0),
            z_min=self.config.get("fatigue_z", 2.0),
        )
        return dict(version="insights-v1", at=at.isoformat(), **fixed, fatigue=fatigue)


ACTIONS = {
    "short_break": "短い休憩",
    "refresh": "軽い気分転換",
    "switch_task": "作業の切り替え",
}


def recommend(rows, at, fatigue, typing):
    """Singleton-consequent weighted rules, ranked by explicit self-reported usefulness."""
    f = max(0.0, min(1.0, ((fatigue.get("value") or 1) - 2) / 2))
    change = max(0.0, min(1.0, (fatigue.get("difference") or 0) / 2))
    correction = 1.0 if typing.get("unusual") else 0.0
    # Rule firing strengths and singleton consequents (short_break, refresh, switch_task).
    rules = [
        (f, (1.0, 0.6, 0.3)),
        (change, (0.8, 0.7, 0.5)),
        (correction, (0.9, 0.6, 0.8)),
    ]
    total = sum(w for w, _ in rules)
    scores = {
        a: sum(w * v[i] for w, v in rules) / total if total else 0.0
        for i, a in enumerate(ACTIONS)
    }
    evidence = {}
    outcomes = {a: [] for a in ACTIONS}
    for r in reversed(available_rows(rows, at)):
        if (
            r["kind"] != "research_event"
            or r["payload"]["kind"] != "intervention_outcome"
        ):
            continue
        p = r["payload"]["payload"]
        if p["executed"] and p.get("helpfulness") in ("helpful", "unchanged", "worse"):
            # Only use an actually offered intervention; never fabricate a successful memory.
            q = next(
                (
                    x["payload"]
                    for x in rows
                    if x["kind"] == "question_issued" and x["id"] == p["question_id"]
                ),
                None,
            )
            if q and q.get("kind") == "breakCheck":
                reason = "typing" if typing.get("unusual") else "fatigue"
                if q.get("eventType") == "support_" + reason:
                    quality = {"helpful": 1, "unchanged": 0, "worse": -1}[
                        p["helpfulness"]
                    ]
                    outcomes[p["action"]].append(quality)
                    if quality > 0:
                        evidence.setdefault(p["action"], r["id"])
    for a, values in outcomes.items():
        if values:
            quality = sum(values) / len(values)
            scores[a] += 0.15 * quality
            if quality <= 0:
                evidence.pop(a, None)
    ranked = sorted(scores, key=lambda a: (-scores[a], list(ACTIONS).index(a)))
    return [
        dict(
            action=a,
            label=ACTIONS[a],
            score=round(scores[a], 3),
            based_on_event_id=evidence.get(a),
        )
        for a in ranked
    ]
