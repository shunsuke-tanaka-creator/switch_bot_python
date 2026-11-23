"""Aggregate-calibrated ordinal process and causal synthetic lifelog observations."""
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .data import JST, SCALES, dt
from .sample_data import _iso, _record
from .sample_registry import record_hash, register_generation
from .tendency_profile import canonical_hash, category_probabilities, load_profile, validate_profile

VERSION = "personalized-sample-v1"
STREAMS = ("calibration", "calendar", "factors", "answer_opportunities", "completion",
           "timing", "uploads", "scores", "states", "typing", "steps", "weather", "switchbot")
SCENARIOS = ("personalized", "gate-test", "boundary", "low-steps", "shuffled-labels")


class _ReplayStore:
    """Append-only Store adapter; avoids repeatedly decoding the same typing JSON.

    The production binding functions run unmodified on normalized envelopes.
    No topology assignments or pattern IDs are synthesized by this adapter.
    """
    def __init__(self):
        self.records = {}
        self.sequence = 0

    def save(self, kind, identifier, participant, at, payload):
        from lifelog.time import stamp
        key = (participant, kind, identifier)
        if key in self.records:
            return False
        self.sequence += 1
        self.records[key] = dict(seq=self.sequence, participant=participant, kind=kind, id=identifier,
                                 at=stamp(at), payload=payload)
        return True

    def rows(self, participant):
        return sorted((r for r in self.records.values() if r["participant"] == participant),
                      key=lambda r: (r["at"], r["seq"]), reverse=True)


def rng_stream(seed, name):
    """Adding draws to one component cannot perturb another component's stream."""
    if name not in STREAMS:
        raise ValueError("unknown RNG component")
    digest = hashlib.sha256(f"{VERSION}:{int(seed)}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def generator_hash():
    # Include feature consumers: a changed binding/calibration is a new generation.
    files = ("personalized_sample.py", "tendency_profile.py", "sample_validation.py", "sample_registry.py",
             "sample_data.py", "data.py", "evaluation.py", "baseline.py", "topology.py", "insights.py", "context_data.py")
    return canonical_hash({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in files})


def _calendar(day, profile, rng):
    c = profile["measured"]["calendar"]
    work_weights = c["work_weekday_weights"]
    weekday = day.weekday()
    work = rng.random() < work_weights.get(str(weekday), 0) / max(work_weights.values())
    lab = weekday == c["lab_weekday"]
    # Adjust other scheduled days so the overall schedule rate remains ~22/29.
    work_rate = sum(work_weights.values()) / max(work_weights.values()) / 7
    union_rate = work_rate + (1 - work_weights.get(str(c["lab_weekday"]), 0) / max(work_weights.values())) / 7
    other = np.clip((c["scheduled_day_ratio"] - union_rate) / max(1 - union_rate, .01), 0, 1)
    scheduled = work or lab or rng.random() < other
    events = []
    for tag in (["work"] if work else []) + (["lab"] if lab else []) + (["other"] if scheduled and not work and not lab else []):
        hour = int(rng.choice(c["start_hours"]))
        start = day + timedelta(hours=hour, minutes=int(rng.integers(-15, 16)))
        duration = float(np.clip(rng.lognormal(math.log(c["block_median_hours"]), .25), .5, 5))
        events.append(dict(start=_iso(start), end=_iso(start + timedelta(hours=duration)),
                           title="sample:" + tag))
    return dict(date=day.date().isoformat(), is_work_day=bool(work), is_lab_day=bool(lab),
                scheduled=bool(scheduled), events=events)


def _latent(factors, hour, flags, a):
    shift = (hour - 12) / 12
    return (factors + np.array(a["work_effect"]) * flags["is_work_day"]
            + np.array(a["lab_effect"]) * flags["is_lab_day"]
            + np.array(a["intraday_effect"]) * shift)


def _calibrate(profile, rng):
    """Empirical ordinal thresholds after day effects, drift and independent noise."""
    a = profile["assumed"]
    n = 20000
    hours = rng.choice([22., 1., 10.], n, p=a["time_mixture"])
    weekdays = rng.integers(0, 7, n)
    weights = profile["measured"]["calendar"]["work_weekday_weights"]
    work = rng.random(n) < np.array([weights.get(str(int(d)), 0) / max(weights.values()) for d in weekdays])
    lab = weekdays == profile["measured"]["calendar"]["lab_weekday"]
    factors = rng.normal(size=(n, 3))
    factors += work[:, None] * a["work_effect"] + lab[:, None] * a["lab_effect"]
    # Late-night daily recall summarizes the preceding day rather than the 01h state.
    daily = rng.random(n) < .4
    drift_hours = np.where(daily, 17, hours)
    factors += ((drift_hours - 12) / 12)[:, None] * a["intraday_effect"]
    thresholds, probabilities = {}, {}
    for i, name in enumerate(SCALES):
        probabilities[name] = category_probabilities(profile["measured"]["scales"][name])
        z = factors @ np.array(a["factor_loadings"][name])
        z += rng.normal(size=n) * np.where(daily, a["recall_noise_sd"], a["noise_sd"])
        cuts = np.cumsum(probabilities[name])[:-1]
        thresholds[name] = np.quantile(z, cuts)
    return thresholds, probabilities


def _scores(factors, hour, flags, profile, thresholds, rng, daily=False):
    a = profile["assumed"]
    latent = _latent(factors, 17 if daily else hour, flags, a)
    return {name: int(1 + np.searchsorted(thresholds[name],
                latent @ np.array(a["factor_loadings"][name]) +
                rng.normal(0, a["recall_noise_sd"] if daily else a["noise_sd"])))
            for name in SCALES}


def _session(day, start, state, calendar, steps, participant, profile, rng, index):
    """Emit physical insertions/deletion bursts; consumer, not payload, computes rates."""
    ranges = profile["measured"]["topology"]["feature_ranges"]
    lower = max(22, math.ceil(math.exp(8 * ranges["input"][0]) - 1))
    upper = min(80, math.floor(math.exp(8 * ranges["input"][1]) - 1))
    inserts = int(rng.integers(lower, max(lower + 1, upper + 1)))
    pauses = int(np.clip(round(rng.uniform(*ranges["pauses"]) * (inserts - 1)), 1, inserts - 1))
    # Map corrections divides burst/insertion by 2; reproduce that exact feature.
    bursts = max(1, round(rng.uniform(*ranges["corrections"]) * 2 * inserts))
    pause_positions = set(rng.choice(np.arange(1, inserts), pauses, replace=False).tolist())
    delete_positions = set(rng.choice(np.arange(inserts), bursts, replace=False).tolist())
    events, offset = [], 2.
    for i in range(inserts):
        if i:
            offset += float(rng.uniform(3, 7) if i in pause_positions else rng.uniform(.4, 1.4))
        t = start + timedelta(seconds=offset)
        events.extend([dict(time=_iso(t - timedelta(milliseconds=50)), type="key", data={"keyID": "か"}),
                       dict(time=_iso(t), type="kana", data={"kana": "か", "input_mode": "hiragana"})])
        if i in delete_positions:
            for j in range(int(rng.integers(1, 4))):
                events.append(dict(time=_iso(t + timedelta(milliseconds=60 + 40 * j)),
                                   type="delete", data={"target": "composition", "input_mode": "hiragana"}))
    events.append(dict(time=_iso(start + timedelta(seconds=offset + 1)), type="commit",
                       data={"text": "sample observation"}))
    end = start + timedelta(minutes=10)
    measurement = end - timedelta(seconds=2)
    identifier = f"session-{day.date()}-{index}"
    place = dict(id=f"place_sample_{state}", name=f"sample place {chr(65 + state)}")
    action = dict(id=f"task_sample_{state}", name=f"sample action {chr(65 + state)}")
    context = dict(step_count=steps, walking_distance_m=round(steps * .7, 2),
                   active_energy_kcal=round(steps * .035, 2), battery_level=.75,
                   battery_state="unplugged", calendar_events=calendar["events"], place_id=place["id"],
                   updated_at=_iso(measurement),
                   measurement_updated_at={k: _iso(measurement) for k in
                       ("step_count", "walking_distance_m", "active_energy_kcal", "battery_level", "calendar_events")})
    payload = dict(participant_id=participant, session_id=identifier, device_id="sample-device",
                   started_at=_iso(start), ended_at=_iso(end), received_at=_iso(end + timedelta(seconds=1)),
                   close_reason="sample_generation", events=sorted(events, key=lambda e: e["time"]),
                   typing=dict(total_typed_characters=inserts,
                               backspace_count=sum(e["type"] == "delete" for e in events)), context=context)
    return [_record("session", identifier, participant, end, payload),
            _record("context_inference", identifier, participant, end,
                    dict(participant_id=participant, place=place, task=action, source="synthetic",
                         received_at=_iso(end + timedelta(seconds=1))))]


def _bind_steps(rows, participant, cache_windows=True):
    """Replay exactly the production ingestion and topology binding, in event time."""
    from .insights import persist_step_snapshot, persist_state_step_snapshots
    from .topology import typing_windows
    store = _ReplayStore()
    if rows:
        # Context and session upload must precede the binding call.
        relevant = [r for r in rows if r["kind"] in ("session", "context_inference")
                    or (r["kind"] == "research_event" and r["payload"]["kind"].startswith("context_"))]
        ordered = sorted(relevant, key=lambda r: (dt(r["payload"].get("received_at", r["at"])), r["kind"], r["id"]))
        # Cache only windows owned by one session. With overlapping uploads,
        # production must reconstruct partial windows at each reception cutoff.
        owners = {}
        for row in relevant:
            if row["kind"] != "session":
                continue
            payload = row["payload"]
            for event in payload["events"]:
                if event["type"] not in ("kana", "text_insert", "delete"):
                    continue
                t = dt(event["time"])
                bucket = t.replace(minute=t.minute // 10 * 10, second=0, microsecond=0)
                key = (bucket, payload.get("device_id", "legacy"), event["data"].get("input_mode", "kana"))
                owners.setdefault(key, {})[row["id"]] = dt(payload.get("received_at", payload["ended_at"]))
        cached = None
        if cache_windows and all(len(v) == 1 for v in owners.values()):
            last = max(dt(r["payload"].get("received_at", r["at"])) for r in relevant)
            cached = [(window, next(iter(owners[(window["at"], window["device"], window["mode"])].values())))
                      for window in typing_windows(relevant, last + timedelta(minutes=10))]
        groups = {}
        for row in ordered:
            groups.setdefault(dt(row["payload"].get("received_at", row["at"])), []).append(row)
        for available, group in groups.items():
            sessions = []
            for row in group:
                store.save(row["kind"], row["id"], participant, dt(row["at"]), row["payload"])
                if row["kind"] == "session":
                    persist_step_snapshot(store, participant, row["id"], row["payload"])
                    sessions.append(row)
            if sessions:
                windows = ([w for w, received in cached if w["end"] <= available and received <= available]
                           if cached is not None else None)
                persist_state_step_snapshots(store, participant, available, windows=windows)
        return [dict(kind=r["kind"], id=r["id"], participant=r["participant"], at=r["at"], payload=r["payload"])
                for r in store.rows(participant) if r["kind"] in ("step_snapshot", "state_step_snapshot")]
    return []


def build_personalized_records(profile, seed=17, weeks=8, scenario="personalized", participant=None):
    profile = validate_profile(profile)
    if type(seed) is not int:
        raise ValueError("master seed must be an integer")
    if scenario not in SCENARIOS or type(weeks) is not int or not 1 <= weeks <= 52:
        raise ValueError("invalid scenario or weeks (1..52)")
    a, m = profile["assumed"], profile["measured"]
    days = 7 if scenario == "gate-test" else weeks * 7
    identity = dict(generator_version=VERSION, generator_hash=generator_hash(), profile_hash=canonical_hash(profile),
                    seed=int(seed), weeks=weeks, days=days, scenario_id=scenario, start_date=a["start_date"])
    participant = participant or f"{profile['participant_id']}-g{canonical_hash(identity)[:20]}-sample"
    if not participant.endswith("-sample") or participant == profile["participant_id"]:
        raise ValueError("reserved sample participant required")
    rng = {name: rng_stream(seed, name) for name in STREAMS}
    thresholds, probabilities = _calibrate(profile, rng["calibration"])
    rows, opportunities, calendar, states = [], [], [], []
    factor = rng["factors"].normal(size=3)
    first = dt(a["start_date"] + "T00:00:00+09:00")
    no_answer_probability = .6 if scenario == "boundary" else a["no_answer_probability"]
    answer_days_rate = m["answers_per_day"] / (1 - no_answer_probability)
    current_ratio = m["scope_ratio"]["current"] / sum(m["scope_ratio"].values())
    daily_probability = min(1., answer_days_rate * (1 - current_ratio))
    for d in range(days):
        day = first + timedelta(days=d)
        flags = _calendar(day, profile, rng["calendar"])
        calendar.append(flags)
        factor = a["rho"] * factor + math.sqrt(1 - a["rho"] ** 2) * rng["factors"].normal(size=3)
        if rng["factors"].random() < a["anomaly_probability"]:
            factor += rng["factors"].normal(0, 1.2, 3)
        daily_factor = factor + .2 * np.sin(d / 14)  # bounded slow drift
        # Semi-Markov dwell intervals, independent of ordinal answers.
        t, stop = day + timedelta(hours=8), day + timedelta(hours=24)
        state = int(rng["states"].choice(5, p=a["state_weights"]))
        step_total, index = 0, 0
        while t < stop:
            dwell = float(np.clip(rng["states"].lognormal(math.log(a["dwell_median_minutes"]), .35), 30, 240))
            end = min(stop, t + timedelta(minutes=dwell))
            states.append(dict(start=_iso(t), end=_iso(end), state=state))
            per_hour = a["steps_per_state_hour"][state] * (.05 if scenario == "low-steps" and d >= days - 2 else 1)
            # Align every typing session to a closed, non-overlapping topology window.
            start = t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0)
            if start < t:
                start += timedelta(minutes=10)
            remainder_start = t
            for session_start in (start, start + timedelta(minutes=10)):
                session_end = session_start + timedelta(minutes=10)
                if session_end <= end:
                    step_total += int(rng["steps"].poisson(per_hour * (session_end - remainder_start).total_seconds() / 3600))
                    rows.extend(_session(day, session_start, state, flags, step_total, participant, profile, rng["typing"], index))
                    index += 1
                    remainder_start = session_end
            step_total += int(rng["steps"].poisson(per_hour * (end - remainder_start).total_seconds() / 3600))
            for axis in ("place", "task"):
                identifier = f"label-{d}-{len(states)}-{axis}"
                rows.append(_record("research_event", identifier, participant, end,
                    dict(event_id=identifier, participant_id=participant, kind="context_label",
                         occurred_at=_iso(end),
                         payload=dict(axis=axis, class_id=f"{axis}_sample_{state}",
                                      start=_iso(t), end=_iso(end)), received_at=_iso(end))))
            t = end
            state = int(rng["states"].choice(5, p=a["state_weights"]))
        for h in range(8 * 12, 24 * 12):
            at = day + timedelta(minutes=h * 5)
            identifier = f"heartbeat-{d}-{h}"
            rows.append(_record("heartbeat", identifier, participant, at,
                               dict(event_id=identifier, participant_id=participant, occurred_at=_iso(at),
                                    logging_enabled=True, questions_enabled=False, interventions_enabled=False,
                                    received_at=_iso(at), source="synthetic")))
        for axis in ("place", "task"):
            # Names explicitly mark the synthetic identities; no actual place names.
            for state_id in range(5):
                identifier = f"name-{axis}-{state_id}"
                if d == 0:
                    rows.append(_record("research_event", identifier, participant, day,
                        dict(event_id=identifier, participant_id=participant, kind="context_name", occurred_at=_iso(day),
                             payload=dict(axis=axis, class_id=f"{axis}_sample_{state_id}",
                                          name=f"sample {axis} {chr(65 + state_id)}"))))
        unanswered = scenario != "gate-test" and rng["answer_opportunities"].random() < no_answer_probability
        if scenario == "gate-test":
            scopes = ["current"] * 12 + ["daily"]
        else:
            daily_count = int(rng["answer_opportunities"].random() < daily_probability)
            expected_current = max(.1, answer_days_rate - daily_probability)
            # Mean current completions accounts for completion attrition.
            current_count = int(rng["answer_opportunities"].poisson(expected_current / a["completion_probability"]))
            if not current_count and not daily_count:
                current_count = 1
            scopes = ["current"] * current_count + ["daily"] * daily_count
        for i, scope in enumerate(scopes):
            band = int(rng["timing"].choice(3, p=a["time_mixture"]))
            hour = (22., 1., 10.)[band] + float(np.clip(rng["timing"].normal(0, a["time_jitter_minutes"] / 60), -.95, .95))
            target = day.date()
            # Daily midnight completions carry the previous day's 22h prediction cutoff.
            if scope == "daily":
                completion = day + timedelta(days=band == 1, hours=hour if band != 2 else 22)
                cutoff = day + timedelta(hours=21, minutes=45)
            else:
                completion = day + timedelta(hours=hour)
                cutoff = completion - timedelta(seconds=float(rng["timing"].uniform(25, 180)))
            if completion <= cutoff:
                completion = cutoff + timedelta(minutes=5)
            completed = not unanswered and (scenario == "gate-test" or scope == "daily" or
                                             rng["completion"].random() < a["completion_probability"])
            received = completion + timedelta(seconds=float(min(rng["uploads"].lognormal(3.5, 1.2), 7200)))
            identifier = f"answer-{d}-{i}-{scope}"
            opportunities.append(dict(id=identifier, target_day=str(target), scope=scope, cutoff=_iso(cutoff),
                                      completed=bool(completed), answered_at=_iso(completion) if completed else None,
                                      received_at=_iso(received) if completed else None,
                                      origin="observed_history_replay"))
            if not completed:
                continue
            values = _scores(daily_factor, hour, flags, profile, thresholds, rng["scores"], daily=scope == "daily")
            rows.append(_record("narrative", identifier, participant, completion,
                dict(event_id=identifier, participant_id=participant, detected_at=_iso(cutoff),
                     question_shown_at=_iso(cutoff), answered_at=_iso(completion), received_at=_iso(received),
                     target_day=str(target), question=dict(kind="currentState" if scope == "current" else "nightReflection",
                     prompt="sample state assessment"), answer=dict(label=scope,
                     free_text=json.dumps(values, sort_keys=True)), source="synthetic", event_type="state_assessment")))
        weather = str(rng["weather"].choice(list(m["weather_probabilities"]), p=list(m["weather_probabilities"].values())))
        at = day + timedelta(hours=22, minutes=30)
        rows.append(_record("research_event", f"weather-{d}", participant, at,
                           dict(participant_id=participant, event_id=f"weather-{d}", kind="weather_mood",
                                occurred_at=_iso(at), payload=dict(weather=weather))))
        if a["switchbot_enabled"]:
            for h in (8, 23):
                at = day + timedelta(hours=h)
                rows.append(_record("switchbot_observation", f"switchbot-{d}-{h}", participant, at,
                    dict(device_id="sample-door", name="sample 玄関", state="close", is_open=False,
                         battery=int(rng["switchbot"].integers(70, 95)), last_seen=_iso(at), rssi=-55)))
    if scenario == "shuffled-labels":
        answers = [r for r in rows if r["kind"] == "narrative"]
        shuffled = rng["scores"].permutation([r["payload"]["answer"]["free_text"] for r in answers])
        for row, values in zip(answers, shuffled):
            row["payload"]["answer"]["free_text"] = str(values)
    rows.extend(_bind_steps(rows, participant))
    for row in rows:
        row["payload"]["provenance"] = dict(source="synthetic", generation_id=participant, seed=int(seed),
                                           generator_version=VERSION, profile_version=profile["profile_version"],
                                           scenario_id=scenario)
    rows.sort(key=lambda r: (dt(r["at"]), r["kind"], r["id"]))
    return rows, dict(identity=identity, profile=profile, participant=participant, calendar=calendar,
                      states=states, opportunities=opportunities,
                      category_probabilities={k: v.tolist() for k, v in probabilities.items()},
                      thresholds={k: v.tolist() for k, v in thresholds.items()})


def generate_personalized_sample(tendencies, store=None, weeks=8, seed=17, scenario="personalized",
                                 require_gates=False, dry_run=False, manifest_path=None,
                                 source_participant="p001", parent_generation=None, reason="explicit generation"):
    from .sample_validation import validate_records
    profile = load_profile(tendencies) if not isinstance(tendencies, dict) else validate_profile(tendencies)
    if source_participant != profile["participant_id"]:
        raise ValueError("profile participant does not match --participant")
    initial_weeks = weeks
    while True:
        rows, details = build_personalized_records(profile, seed, weeks, scenario)
        report = validate_records(rows, profile, details)
        gate_key = "any_display_eligible" if scenario == "gate-test" else "display_eligible"
        if not require_gates or report["gates"][gate_key]:
            break
        if scenario == "gate-test" or weeks >= 52:
            raise ValueError("requested gates cannot be met in this scenario/maximum duration")
        weeks += 1
    manifest = dict(schema_version="personalized-sample-generation-v1", **details["identity"],
                    source_participant=source_participant, sample_participant=details["participant"],
                    profile=profile, rng_streams={name: f"sha256({VERSION}:seed:{name})/PCG64" for name in STREAMS},
                    period=dict(first_date=profile["assumed"]["start_date"], days=details["identity"]["days"]),
                    preview_at=max(r["payload"]["received_at"] for r in rows if r["kind"] == "session"
                        and dt(r["payload"]["received_at"]).date() ==
                            dt(r["payload"]["context"]["measurement_updated_at"]["step_count"]).date()),
                    requested_weeks=initial_weeks, parent_generation=parent_generation, regeneration_reason=reason,
                    required_gate_policy=("training start + any display item" if scenario == "gate-test" else "training start + all display items"),
                    record_hash=record_hash(rows), validation_hash=canonical_hash(report),
                    opportunities_hash=canonical_hash(details["opportunities"]), counts=report["counts"],
                    gates=report["gates"], calibration=dict(probabilities=details["category_probabilities"],
                                                         thresholds=details["thresholds"]))
    if dry_run:
        return dict(status="dry_run", inserted=0, **manifest, validation=report)
    if store is None:
        raise ValueError("store is required")
    if manifest_path is None:
        db_path = store.db.execute("PRAGMA database_list").fetchone()[2]
        if db_path:
            manifest_path = Path(db_path).parent / "sample" / details["participant"] / "manifest.json"
    result = register_generation(store, rows, manifest, report, details["opportunities"], manifest_path)
    return dict(result, manifest=str(manifest_path) if manifest_path else None, validation=report)
