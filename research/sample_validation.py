"""Production-parser preflight and aggregate realism reports (no torch required)."""
from collections import Counter, defaultdict
from datetime import timedelta
import math

import numpy as np

from .baseline import conditional, current_answers, period
from .data import SCALES, dt, training_examples
from .evaluation import chronological_split, baseline_metrics
from .sample_registry import generation, record_hash
from .tendency_profile import canonical_hash
from .topology import behavior_map, typing_windows


def gate_preflight(rows, at=None):
    timelines, examples = training_examples(rows)
    labels = [l for l, _ in examples if l.scope in ("current", "daily")]
    dates = {l.at.date() for l in labels}
    at = dt(at) if at else max((l.available for l in labels), default=dt("2026-01-01T00:00:00+09:00")) + timedelta(seconds=1)
    tr, va, te, train_end, test_start = chronological_split(examples)
    items = {}
    for scope in ("current", "daily"):
        items[scope] = {}
        for i, name in enumerate(SCALES):
            train_n = int(sum(l.scope == scope and np.isfinite(l.values[i]) for l, _ in tr))
            valid = [l for l, _ in va if l.scope == scope and np.isfinite(l.values[i])]
            items[scope][name] = dict(train=int(train_n), validation=len(valid),
                validation_days=len({l.at.date() for l in valid}), display_eligible=train_n >= 10 and len(valid) >= 3,
                switch_support_eligible=train_n >= 10 and len(valid) >= 10 and len({l.at.date() for l in valid}) >= 3)
    # Same period, activity strata, dates and thresholds as weekday_trends,
    # without doing its expensive bootstrap just to count eligibility.
    trend_cells = []
    for source, timeline in timelines.items():
        source_rows = timeline.rows
        current = [l for l in current_answers(source_rows, at) if at - timedelta(days=84) <= l.at < at]
        age = (at - min((l.at for l in current), default=at)).days
        samples = {(l.at.date(), period(l.at), l.activity) for l in current if np.isfinite(l.values[1])}
        for weekday in range(7):
            for band in ("morning", "afternoon", "evening", "night"):
                target = [s for s in samples if s[0].weekday() == weekday and s[1] == band]
                reference = [s for s in samples if s[0].weekday() != weekday and s[1] == band]
                shared = {s[2] for s in target} & {s[2] for s in reference}
                td = {s[0] for s in target if s[2] in shared}
                rd = {s[0] for s in reference if s[2] in shared}
                trend_cells.append(dict(source=source, weekday=weekday, period=band,
                                        target_days=len(td), reference_days=len(rd), age_days=age,
                                        eligible=age >= 42 and len(td) >= 4 and len(rd) >= 8))
    training_eligible = len(labels) >= 20 and len(dates) >= 5 and bool(tr) and bool(va)
    return dict(state_answers=len(labels), answer_days=len(dates), training_start_eligible=training_eligible,
                train_before=str(train_end) if train_end else None, test_from=str(test_start) if test_start else None,
                items=items, display_eligible=training_eligible and all(v["display_eligible"] for s in items.values() for v in s.values()),
                any_display_eligible=training_eligible and any(v["display_eligible"] for s in items.values() for v in s.values()),
                trend_eligible=any(c["eligible"] for c in trend_cells), trend_cells=trend_cells,
                preview_support=True, switch_note="support only; accuracy margins and three distinct win days still required")


def _summary(values):
    values = list(values)
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                sd=float(np.std(values)) if values else None,
                min=float(min(values)) if values else None, max=float(max(values)) if values else None)


def validate_records(rows, profile, details=None, training_report=None):
    details = details or {}
    timelines, examples = training_examples(rows)
    labels = [l for l, _ in examples if l.scope in ("current", "daily")]
    now = max((dt(r["payload"].get("received_at", r["at"])) for r in rows),
              default=dt(profile["assumed"]["start_date"] + "T00:00:00+09:00")) + timedelta(seconds=1)
    days = details.get("identity", {}).get("days") or ((max(l.at.date() for l in labels) - min(l.at.date() for l in labels)).days + 1 if labels else 0)
    scopes = Counter(l.scope for l in labels)
    answered = [r for r in rows if r["kind"] == "narrative" and r["payload"].get("question", {}).get("kind") in ("currentState", "nightReflection")]
    completion_days = {dt(r["payload"]["answered_at"]).date() for r in answered}
    target_days = {r["payload"].get("target_day", str(dt(r["payload"]["question_shown_at"]).date())) for r in answered}
    scales = {}
    warnings = []
    for i, name in enumerate(SCALES):
        values = [float(l.values[i]) for l in labels if np.isfinite(l.values[i])]
        target = profile["measured"]["scales"][name]
        scales[name] = dict(**_summary(values), histogram={str(v): sum(x == v for x in values) for v in range(1, 6)},
                            target=target, mean_difference=float(np.mean(values) - target["mean"]) if values else None)
        if values and abs(scales[name]["mean_difference"]) > .2:
            warnings.append(f"{name}: raw mean differs by more than 0.2; target is a weighted daily baseline")
    # Run the actual production conditional aggregator at each weekday × period,
    # and at the end of every day with the brief's seven-day lookback.
    baselines = {}
    daily_baselines = {}
    weighted_comparison = {}
    for source, timeline in timelines.items():
        baselines[source] = conditional(timeline.rows, now, lookback_days=7)
        daily_baselines[source] = [
            dict(at=(now - timedelta(days=offset)).isoformat(),
                 scales=conditional(timeline.rows, now - timedelta(days=offset), lookback_days=7))
            for offset in range(min(days, 56))
        ]
        weighted_comparison[source] = {}
        for name in SCALES:
            observed = baselines[source][name]
            target = profile["measured"]["scales"][name]
            weighted_comparison[source][name] = dict(target=target, latest=observed,
                mean_difference=observed["mean"] - target["mean"] if observed["mean"] is not None else None,
                sd_difference=observed["sd"] - target["sd"] if observed["sd"] is not None else None,
                historical_baseline_means=_summary(point["scales"][name]["mean"] for point in daily_baselines[source]
                                                   if point["scales"][name]["mean"] is not None))
    hours = Counter(dt(r["payload"]["answered_at"]).hour for r in answered)
    delays = [(dt(r["payload"]["answered_at"]) - dt(r["payload"]["question_shown_at"])).total_seconds() for r in answered]
    uploads = [(dt(r["payload"]["received_at"]) - dt(r["payload"]["answered_at"])).total_seconds() for r in answered]
    windows = typing_windows(rows, now)
    patterns = behavior_map(windows, now, rows=rows)
    step_groups = defaultdict(list)
    for r in rows:
        if r["kind"] == "step_snapshot":
            p = r["payload"]
            step_groups[(r["participant"], p.get("device_id"), dt(p["measurement_at"]).date())].append(
                (dt(p["measurement_at"]), p["step_count"]))
    step_errors = []
    totals = []
    for key, snapshots in step_groups.items():
        snapshots.sort()
        totals.append(snapshots[-1][1])
        if any(v < 0 or not math.isfinite(v) for _, v in snapshots) or any(a[1] > b[1] for a, b in zip(snapshots, snapshots[1:])):
            step_errors.append(str(key))
    bound = [r for r in rows if r["kind"] == "state_step_snapshot"]
    # A state can later merge; canonicalize historical bindings through final aliases.
    ids = {n["id"] for n in patterns["nodes"]}
    aliases = patterns["aliases"]
    def canonical(key):
        while key in aliases:
            key = aliases[key]
        return key
    invalid_bindings = [r["id"] for r in bound if canonical(r["payload"]["state_id"]) not in ids]
    calendar = details.get("calendar", [])
    if not calendar:
        by_day = {}
        for r in rows:
            if r["kind"] == "session":
                events = r["payload"].get("context", {}).get("calendar_events", [])
                by_day[str(dt(r["at"]).date())] = dict(date=str(dt(r["at"]).date()), events=events,
                    scheduled=bool(events), is_work_day=any(e["title"] == "sample:work" for e in events),
                    is_lab_day=any(e["title"] == "sample:lab" for e in events))
        calendar = list(by_day.values())
    work = Counter(dt(c["date"] + "T00:00:00+09:00").weekday() for c in calendar if c["is_work_day"])
    lab = Counter(dt(c["date"] + "T00:00:00+09:00").weekday() for c in calendar if c["is_lab_day"])
    duration = [(dt(e["end"]) - dt(e["start"])).total_seconds() / 3600 for c in calendar for e in c["events"]]
    weather = Counter(r["payload"]["payload"]["weather"] for r in rows if r["kind"] == "research_event" and r["payload"].get("kind") == "weather_mood")
    autocorrelation = {}
    weekday_means = {}
    for i, name in enumerate(SCALES):
        daily = defaultdict(list)
        for l in labels:
            if l.scope == "current" and np.isfinite(l.values[i]):
                daily[l.at.date()].append(float(l.values[i]))
        pairs = [(np.mean(v), np.mean(daily[d + timedelta(days=1)])) for d, v in daily.items() if d + timedelta(days=1) in daily]
        x = np.array(pairs)
        autocorrelation[name] = float(np.corrcoef(x.T)[0, 1]) if len(x) > 2 and (x.std(0) > 0).all() else None
        weekday_means[name] = {str(d): _summary(v for day, values in daily.items() if day.weekday() == d for v in values) for d in range(7)}
    _, va, _, _, _ = chronological_split(examples)
    evaluation = dict(validation_baselines_by_source=baseline_metrics(va))
    if training_report:
        evaluation.update(validation=training_report.get("validation"), support=training_report.get("support"),
                          support_by_source=training_report.get("support_by_source"),
                          validation_by_source=training_report.get("validation_by_source"))
        if training_report.get("validation", {}).get("mean_mae") is not None and training_report["validation"]["mean_mae"] < .15:
            warnings.append("Validation MAE < 0.15: implausibly small; check leakage/noise and shuffled-label control")
    return dict(schema_version="personalized-sample-validation-v1", record_hash=record_hash(rows),
                counts=dict(total=len(rows), by_kind=dict(Counter(r["kind"] for r in rows)), scopes=dict(scopes)),
                scales=scales, condition_weighted_daily_baselines=baselines, daily_baseline_series=daily_baselines,
                condition_weighted_comparison=weighted_comparison,
                timing=dict(hours={str(h): hours[h] for h in range(24)}, answers_per_day=len(labels) / max(days, 1),
                            completion_delay_seconds=_summary(delays), upload_delay_seconds=_summary(uploads)),
                missingness=dict(days=days, target_days=len(target_days),
                    no_answer_day_ratio=1 - len(target_days) / max(days, 1), completion_days=len(completion_days),
                    incomplete_opportunities=sum(not o["completed"] for o in details.get("opportunities", []))),
                calendar=dict(scheduled_day_ratio=sum(c["scheduled"] for c in calendar) / max(len(calendar), 1),
                              work_weekdays=dict(work), lab_weekdays=dict(lab),
                              block_median_hours=float(np.median(duration)) if duration else None),
                weather=dict(weather), autocorrelation=autocorrelation, weekday_means=weekday_means,
                topology=dict(windows=len(windows), eligible_windows=sum(w["inserts"] >= 20 for w in windows),
                    nodes=len(patterns["nodes"]), established_nodes=sum(n["status"] == "established" for n in patterns["nodes"]),
                    dominant_node=max(patterns["nodes"], key=lambda n: n["count"], default=None),
                    features={key: _summary(n["center"][key] for n in patterns["nodes"]) for key in ("input", "corrections", "pauses")},
                    typing_metrics={key: _summary(w[key] for w in windows) for key in ("inserts", "correction_rate", "pause_rate")},
                    synthetic_state_count=len({s["state"] for s in details.get("states", [])}),
                    dwell_minutes=_summary((dt(s["end"]) - dt(s["start"])).total_seconds() / 60 for s in details.get("states", [])),
                    transitions=sum(a["state"] != b["state"] for a, b in zip(details.get("states", []), details.get("states", [])[1:]))),
                steps=dict(sane=not step_errors and not invalid_bindings, decreasing_or_invalid=step_errors,
                           invalid_bindings=invalid_bindings, snapshots=sum(len(v) for v in step_groups.values()),
                           bound_snapshots=len(bound), daily_total=_summary(totals)),
                gates=gate_preflight(rows, now), evaluation=evaluation, warnings=warnings,
                limitations=["Synthetic evaluation is a functional preview, not evidence of predictive performance on real data.",
                             "Brief scale moments are weighted daily baselines; raw marginal conversion is assumed.",
                             "Five synthetic place/action states may produce additional time-conditioned topology nodes."])


def validate_sample(store, sample_id, training_report=None):
    info = generation(store, sample_id)
    if not info or info["manifest"].get("legacy"):
        raise ValueError("a registered personalized generation is required")
    manifest = info["manifest"]
    rows = store.rows(sample_id)
    intact = record_hash(rows) == manifest["record_hash"]
    # Retain generation-time state/calendar summaries; recompute data-dependent
    # sections and gates from DB rows, never trust manifest support counts.
    report = validate_records(rows, manifest["profile"],
                              dict(identity=manifest, opportunities=info["opportunities"]), training_report)
    report["integrity"] = dict(records=intact, saved_report=canonical_hash(info["report"]) == manifest["validation_hash"],
                               opportunities=canonical_hash(info["opportunities"]) == manifest["opportunities_hash"])
    report["topology"].update({k: info["report"]["topology"][k] for k in
                              ("synthetic_state_count", "dwell_minutes", "transitions")})
    if not all(report["integrity"].values()):
        report["warnings"].append("Immutable generation integrity check failed")
    return report
