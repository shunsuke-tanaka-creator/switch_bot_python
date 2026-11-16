"""Personal change detectors and automatic topology alongside the three LSTMs."""

from datetime import timedelta
import math
import numpy as np
from .data import dt
from .baseline import conditional, weekday_trends
from .topology import VERSION as TOPOLOGY_VERSION
from .topology import available_rows, typing_windows, behavior_map


# 追加: Compendium系の一般的なMETs参照値。医療的な処方値ではない。
EXERCISE_METS = (
    ("walking", "歩行", 3.0),
    ("brisk_walking", "速歩", 4.3),
    ("stair_climbing", "階段上り", 8.0),
    ("cycling", "自転車", 6.8),
    ("jogging", "ジョギング", 7.0),
    ("stretching", "ストレッチ", 2.3),
    ("radio_calisthenics", "ラジオ体操", 4.0),
)


# 追加: 不足歩数を100歩/分の歩行相当にし、活動別METs比で5分単位へ切り上げる。
def exercise_candidates(shortage_steps, limit=4):
    walking_minutes = shortage_steps / 100
    return [
        dict(
            action=action,
            label=label,
            mets=mets,
            minutes=max(5, math.ceil(walking_minutes * (3.0 / mets) / 5) * 5),
        )
        for action, label, mets in EXERCISE_METS[:limit]
    ]


# 追加: セッションContextから鮮度を確認できる日累積歩数だけを保存する。
def persist_step_snapshot(store, participant, session_id, payload, fresh_seconds=900):
    context = payload.get("context") or {}
    measured_at = (context.get("measurement_updated_at") or {}).get("step_count")
    steps = context.get("step_count")
    if measured_at is None or type(steps) not in (int, float) or not math.isfinite(steps):
        return False
    # 追加: 任意Contextの不正な時刻で、保存済みセッションのAPI応答を失敗させない。
    try:
        measured_at = dt(measured_at)
        ended_at = dt(payload["ended_at"])
    except (TypeError, ValueError, AttributeError):
        return False
    if steps < 0 or not timedelta(0) <= ended_at - measured_at <= timedelta(
        seconds=fresh_seconds
    ):
        return False
    place = context.get("place")
    place_id = (
        context.get("place_id")
        or context.get("context_place_id")
        or (place.get("id") if isinstance(place, dict) else None)
    )
    return store.save(
        "step_snapshot",
        session_id,
        participant,
        measured_at,
        dict(
            step_count=float(steps),
            measurement_at=measured_at.isoformat(),
            session_at=ended_at.isoformat(),
            source_session_id=session_id,
            place_id=place_id,
            device_id=payload.get("device_id"),
            received_at=payload.get("received_at", ended_at.isoformat()),
        ),
    )


# 追加: GrowingMapの状態割当てと鮮度付き歩数を、今後の状態別基準として結合保存する。
def persist_state_step_snapshots(store, participant, at, fresh_seconds=900):
    at = dt(at)
    rows = available_rows(store.rows(participant), at)
    saved = {
        r["id"] for r in rows if r["kind"] == "state_step_snapshot"
    }
    snapshots = [
        r
        for r in rows
        if r["kind"] == "step_snapshot"
        and r["id"] not in saved
        and timedelta(0) <= at - dt(r["payload"]["session_at"]) <= timedelta(
            seconds=fresh_seconds
        )
    ]
    if not snapshots:
        return 0
    patterns = behavior_map(typing_windows(rows, at), at, rows=rows)
    nodes = {node["id"]: node for node in patterns["nodes"]}
    assignments = [
        assignment
        for assignment in patterns["assignments"]
        if (assignment.get("place") or {}).get("id")
        and (assignment.get("action") or {}).get("id")
    ]
    inserted = 0
    for row in snapshots:
        payload = row["payload"]
        session_at = dt(payload["session_at"])
        nearby = [
            assignment
            for assignment in assignments
            if abs((dt(assignment["end"]) - session_at).total_seconds())
            <= fresh_seconds
        ]
        if not nearby:
            continue
        assignment = min(
            nearby,
            key=lambda value: abs(
                (dt(value["end"]) - session_at).total_seconds()
            ),
        )
        node = nodes[assignment["state_id"]]
        inserted += store.save(
            "state_step_snapshot",
            row["id"],
            participant,
            dt(payload["measurement_at"]),
            dict(
                state_version=TOPOLOGY_VERSION,
                state_id=assignment["state_id"],
                state_observed_at=assignment["end"],
                place=assignment["place"],
                action=assignment["action"],
                state_features=node["center"],
                step_count=payload["step_count"],
                measurement_at=payload["measurement_at"],
                session_at=payload["session_at"],
                source_session_id=payload["source_session_id"],
                device_id=payload.get("device_id"),
                received_at=at.isoformat(),
            ),
        )
    return inserted


# 追加: 既存sessionと新しいstep_snapshotを同じ鮮度付き歩数観測にする。
def step_observations(rows, at, measurement_fresh_seconds=900):
    visible = available_rows(rows, at)
    observations = {}
    for r in visible:
        payload = r["payload"]
        if r["kind"] == "step_snapshot":
            steps = payload.get("step_count")
            if type(steps) not in (int, float) or not math.isfinite(steps) or steps < 0:
                continue
            # 追加: 旧・外部レコードの不正な任意時刻は観測だけ除外する。
            try:
                measured_at = dt(payload["measurement_at"])
                session_at = dt(payload["session_at"])
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            source_id = payload["source_session_id"]
            device_id = payload.get("device_id")
            explicit = True
        elif r["kind"] == "session":
            context = payload.get("context") or {}
            measured = (context.get("measurement_updated_at") or {}).get("step_count")
            steps = context.get("step_count")
            if (
                measured is None
                or type(steps) not in (int, float)
                or not math.isfinite(steps)
                or steps < 0
            ):
                continue
            # 追加: 保存済みセッション内の不正な任意時刻でpreviewを失敗させない。
            try:
                measured_at = dt(measured)
                session_at = dt(payload["ended_at"])
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            source_id = payload.get("session_id", r["id"])
            device_id = payload.get("device_id")
            explicit = False
        else:
            continue
        if measured_at > at or session_at > at:
            continue
        if not timedelta(0) <= session_at - measured_at <= timedelta(
            seconds=measurement_fresh_seconds
        ):
            continue
        observation = dict(
            at=measured_at,
            session_at=session_at,
            steps=float(steps),
            source_id=source_id,
            participant=r.get("participant"),
            device_id=device_id,
        )
        observation_id = (r.get("participant"), source_id)
        if explicit or observation_id not in observations:
            observations[observation_id] = observation
    return sorted(observations.values(), key=lambda x: (x["at"], x["source_id"]))


# 追加: 同じトポロジカル状態の日別歩数分布で運動不足を判定する。
def step_shortage(
    rows,
    at,
    *,
    participant=None,
    state_aliases=None,
    provisional_days=3,
    formal_days=5,
    fresh_seconds=900,
    minimum_shortage_steps=500,
    minimum_shortage_fraction=0.25,
):
    at = dt(at)
    visible = available_rows(rows, at)
    observations = step_observations(
        visible, at, measurement_fresh_seconds=fresh_seconds
    )
    state_aliases = state_aliases or {}

    def canonical_state(identity):
        while identity in state_aliases:
            identity = state_aliases[identity]
        return identity

    today = at.date()
    bindings = {}
    for row in visible:
        if row["kind"] != "state_step_snapshot":
            continue
        payload = row["payload"]
        if (
            payload.get("state_version") != TOPOLOGY_VERSION
            or not payload.get("state_id")
            or not (payload.get("place") or {}).get("id")
            or not (payload.get("action") or {}).get("id")
        ):
            continue
        bindings[(row.get("participant"), payload.get("source_session_id", row["id"]))] = dict(
            id=row["id"],
            participant=row.get("participant"),
            **payload,
        )
    current = next(
        (
            value
            for value in reversed(observations)
            if (value["participant"], value["source_id"]) in bindings
            and (participant is None or value["participant"] == participant)
            if value["at"].date() == today
            and timedelta(0) <= at - value["at"] <= timedelta(seconds=fresh_seconds)
            and timedelta(0)
            <= at - value["session_at"]
            <= timedelta(seconds=fresh_seconds)
        ),
        None,
    )
    if current is None:
        return dict(
            status="insufficient_data",
            low=False,
            reason="current_state_step_measurement_missing_or_stale",
        )
    current_binding = bindings[(current["participant"], current["source_id"])]
    state_id = canonical_state(current_binding["state_id"])
    context = dict(
        basis="topological_state",
        state_id=state_id,
        state_version=current_binding["state_version"],
        place_id=current_binding["place"]["id"],
        place_name=current_binding["place"].get("name"),
        action_id=current_binding["action"]["id"],
        action_name=current_binding["action"].get("name"),
    )
    # 追加: 日累積値の差分を、その観測時に永続化された状態へ帰属させる。
    previous = {}
    daily = {}
    progress = None
    for value in observations:
        key = (
            value["participant"],
            value.get("device_id"),
            value["at"].date(),
        )
        prior = previous.get(key)
        previous[key] = value
        # 追加: 長い観測欠落中の歩数を、直近状態へまとめて誤帰属させない。
        if (
            prior is None
            or value["steps"] < prior["steps"]
            or value["at"] - prior["at"] > timedelta(seconds=fresh_seconds)
        ):
            continue
        delta = value["steps"] - prior["steps"]
        if (
            value["participant"] == current["participant"]
            and value.get("device_id") == current.get("device_id")
            and value["at"].date() == today
            and delta > 0
        ):
            progress = value
        binding = bindings.get((value["participant"], value["source_id"]))
        if binding is None:
            continue
        binding_state = canonical_state(binding["state_id"])
        cell = daily.setdefault(
            (
                value["participant"],
                value.get("device_id"),
                value["at"].date(),
                binding["state_version"],
                binding_state,
            ),
            dict(steps=0.0, occurrences=0),
        )
        cell["steps"] += delta
        cell["occurrences"] += 1
    if progress is None:
        return dict(
            status="insufficient_data",
            low=False,
            reason="device_carry_not_observed",
            context=context,
        )
    current_key = (
        current["participant"],
        current.get("device_id"),
        today,
        current_binding["state_version"],
        state_id,
    )
    current_day = daily.get(current_key)
    if current_day is None:
        return dict(
            status="insufficient_data",
            low=False,
            reason="current_state_steps_unavailable",
            context=context,
        )
    past = [
        (date, value)
        for (owner, device, date, version, state), value in daily.items()
        if owner == current["participant"]
        and device == current.get("device_id")
        and date < today
        and version == current_binding["state_version"]
        and state == state_id
    ]
    baseline = [value["steps"] for _, value in sorted(past)]
    baseline_days = len(baseline)
    baseline_occurrences = sum(value["occurrences"] for _, value in past)
    if baseline_days < provisional_days:
        return dict(
            status="insufficient_data",
            low=False,
            reason="state_baseline_too_short",
            baseline_days=baseline_days,
            baseline_occurrences=baseline_occurrences,
            required_days=provisional_days,
            context=context,
        )
    usual = float(np.median(baseline))
    lower_quartile = float(np.quantile(baseline, 0.25))
    today_steps = current_day["steps"]
    shortage = max(0, int(round(usual - today_steps)))
    low = bool(
        today_steps < lower_quartile
        and shortage >= minimum_shortage_steps
        and shortage >= usual * minimum_shortage_fraction
    )
    result = dict(
        status="available",
        low=low,
        provisional=baseline_days < formal_days,
        baseline_stage="provisional" if baseline_days < formal_days else "formal",
        baseline_days=baseline_days,
        baseline_occurrences=baseline_occurrences,
        context=context,
        usual_steps=int(round(usual)),
        lower_quartile_steps=int(round(lower_quartile)),
        today_steps=int(round(today_steps)),
        shortage_steps=shortage,
        minimum_shortage_steps=minimum_shortage_steps,
        minimum_shortage_fraction=minimum_shortage_fraction,
        cadence_steps_per_minute=100,
        walking_equivalent_minutes=round(shortage / 100, 1),
        candidates=exercise_candidates(shortage) if low else [],
        evidence_id=current["source_id"],
        carry_evidence_id=progress["source_id"],
        note="research threshold; daily step deltas within one persisted topological state",
    )
    return result


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


def typing_change(
    windows,
    at,
    minimum=None,
    *,
    daily_min_inserts=50,
    formal_days=5,
    provisional_days=3,
    fresh_seconds=86400,
):
    if not windows:
        return dict(status="insufficient_data", unusual=False)
    if minimum is not None:
        daily_min_inserts = minimum
    current = windows[-1]
    if (at - current["end"]).total_seconds() > fresh_seconds:
        return dict(status="insufficient_data", unusual=False)

    current_date = current["at"].date()
    matching = [
        w
        for w in windows
        if w["device"] == current["device"] and w["mode"] == current["mode"]
    ]
    days = {}
    for window in matching:
        date = window["at"].date()
        pooled = days.setdefault(
            date, dict(inserts=0, bursts=0.0, windows=0, last=None)
        )
        pooled["inserts"] += window["inserts"]
        pooled["bursts"] += window["correction_rate"] * window["inserts"]
        pooled["windows"] += 1
        if pooled["last"] is None or window["end"] > pooled["last"]["end"]:
            pooled["last"] = window

    current_day = days[current_date]
    if current_day["inserts"] < daily_min_inserts:
        return dict(
            status="insufficient_data",
            unusual=False,
            inserts=current_day["inserts"],
        )

    valid_past = [
        (date, pooled)
        for date, pooled in days.items()
        if date < current_date and pooled["inserts"] >= daily_min_inserts
    ]
    valid_past.sort(key=lambda item: item[0])
    baseline_days = len(valid_past)
    baseline_windows = sum(pooled["windows"] for _, pooled in valid_past)
    if baseline_days < provisional_days:
        return dict(
            status="insufficient_data",
            unusual=False,
            baseline_windows=baseline_windows,
            baseline_days=baseline_days,
        )
    daily_rates = [
        pooled["bursts"] / pooled["inserts"] for _, pooled in valid_past
    ]
    threshold = max(
        float(np.quantile(daily_rates, 0.99)),
        0.02,
    )
    current_rate = current_day["bursts"] / current_day["inserts"]
    previous_rate = daily_rates[-1]
    result = dict(
        status="available",
        unusual=bool(current_rate > threshold and previous_rate > threshold),
        correction_rate=current_rate,
        threshold=threshold,
        inserts=current_day["inserts"],
        at=current_day["last"]["end"].isoformat(),
        evidence_id=current_day["last"]["id"],
        baseline_windows=baseline_windows,
        baseline_days=baseline_days,
        note="input corrections, not confirmed mistakes",
    )
    if baseline_days < formal_days:
        result["provisional"] = True
    return result


class InsightEngine:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.cache = {}
        self.trend_cache = {}

    def snapshot(self, p, at, prediction=None, rows=None, data_mode="real"):
        at = dt(at)
        rows = available_rows(self.store.rows(p) if rows is None else rows, at)
        rows = [r for r in rows if not (r["kind"] == "research_event" and r["payload"]["kind"] == "place_visit")]
        source = [
            r
            for r in rows
            if r["kind"]
            in (
                "session",
                "narrative",
                "correction",
                "research_event",
                "context_inference",
                "step_snapshot",
                "state_step_snapshot",
            )
        ]
        signature = tuple((r.get("seq"), r["kind"], r["id"]) for r in source)
        bucket = at.replace(minute=at.minute // 5 * 5, second=0, microsecond=0)
        cache_id = (p, data_mode)
        key = (cache_id, signature, bucket)
        if self.cache.get(cache_id, {}).get("key") == key:
            fixed = self.cache[cache_id]["data"]
        else:
            windows = typing_windows(rows, at)
            trend_key = (
                cache_id,
                tuple(
                    (r.get("seq"), r["id"])
                    for r in source
                    if r["kind"] in ("narrative", "correction")
                ),
                at.date(),
            )
            if self.trend_cache.get(cache_id, {}).get("key") != trend_key:
                self.trend_cache[cache_id] = dict(
                    key=trend_key,
                    data=weekday_trends(
                        rows, at, min_weeks=self.config.get("trend_min_weeks", 6)
                    ),
                )
            fixed = dict(
                baseline=conditional(rows, at),
                trends=self.trend_cache[cache_id]["data"],
                typing=typing_change(
                    windows,
                    at,
                    daily_min_inserts=self.config.get(
                        "typing_daily_min_inserts", 50
                    ),
                    formal_days=self.config.get("typing_baseline_days", 5),
                    provisional_days=self.config.get(
                        "typing_baseline_provisional_days", 3
                    ),
                    fresh_seconds=self.config.get("typing_fresh_seconds", 86400),
                ),
                patterns=behavior_map(
                    windows,
                    at,
                    radius=self.config.get("pattern_radius", 0.35),
                    rows=rows,
                ),
                places=dict(nodes=[],edges=[],current=None,latest=None,destination=dict(status='gps_disabled',places=[]),anomaly=None,time_anomaly=None),
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
            self.cache[cache_id] = dict(key=key, data=fixed)
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
        # 追加: モデル入力とは独立した提案用の歩数検出器。
        exercise = step_shortage(
            rows,
            at,
            participant=p,
            state_aliases=fixed["patterns"].get("aliases"),
            provisional_days=self.config.get("exercise_baseline_provisional_days", 3),
            formal_days=self.config.get("exercise_baseline_days", 5),
            fresh_seconds=self.config.get("exercise_step_fresh_seconds", 900),
            minimum_shortage_steps=self.config.get(
                "exercise_minimum_shortage_steps", 500
            ),
            minimum_shortage_fraction=self.config.get(
                "exercise_minimum_shortage_fraction", 0.25
            ),
        )
        return dict(
            version="insights-v1",
            at=at.isoformat(),
            **fixed,
            fatigue=fatigue,
            exercise=exercise,
        )


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
