"""Discovery-only, auditable break decisions; no encoder or map mutation."""
from collections import defaultdict
from datetime import timedelta
import json
import math
from statistics import mean, pstdev

from .data import SCALES, dt
from .baseline import conditional, current_answers
from .context_data import annotations, catalog, label_at, resolve
from .topology import available_rows

SCALES_SUPPORT = ("fatigue", "concentration", "sleepiness")
DIRECTIONS = {"fatigue": 1, "concentration": -1, "sleepiness": 1}
ACTIONS = {"short_break": "短い休憩", "refresh": "軽い気分転換", "switch_task": "作業の切り替え"}
PRE = "support_pre:"
BREAK = "support_discovery:"
POST = "support_post:"
BUCKET = timedelta(minutes=5)


def bucket(at):
    at = dt(at)
    return at.replace(minute=at.minute // 5 * 5, second=0, microsecond=0)


def is_support_assessment(payload):
    return str(payload.get("event_type", "")).startswith((PRE, POST))


def support_mode(config):
    value = config.get("discovery_support_mode", "shadow")
    if value not in ("off", "shadow", "live"):
        raise ValueError("discovery_support_mode must be off, shadow, or live")
    return value


def settings(config):
    result = dict(high=4.0, low_focus=2.0, delta=1.0, z=2.0,
                  trend_delta=1.0, trend_floor=3.0)
    for key, value in result.items():
        n = config.get("discovery_support_" + key, value)
        if type(n) not in (int, float) or not math.isfinite(n):
            raise ValueError("Invalid discovery support threshold: " + key)
        if key in ("high", "low_focus", "trend_floor"):
            valid = 1 <= n <= 5
        else:
            valid = 0 < n <= (10 if key == "z" else 4)
        if not valid:
            raise ValueError("Invalid discovery support threshold: " + key)
        result[key] = float(n)
    return result


def _value(item):
    v = (item or {}).get("value")
    if ((item or {}).get("status") != "estimated"
            or (item or {}).get("source") != "discovery"
            or (item or {}).get("provisional", False)
            or type(v) not in (int, float) or not math.isfinite(v)
            or not 1 <= v <= 5):
        return None
    return float(v)


def _predictions(rows, at, mode):
    result = []
    for r in rows:
        if r["kind"] != "inference":
            continue
        p = r["payload"]
        if (p.get("architecture") != "discovery" or p.get("data_mode") != mode
                or p.get("mode") != "live" or p.get("status") != "ready"
                or not p.get("model_id") or not p.get("current")
                or (mode == "real" and p.get("contains_sample", True))):
            continue
        t = dt(p["at"])
        received = max(dt(r["at"]), dt(p.get("received_at", r["at"])),
                       dt(p.get("available_at", p["at"])))
        if t <= received <= at:
            result.append((t, received, p))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _behavior(p, aliases):
    b = p.get("behavior") or {}
    if b.get("status") != "estimated" or not b.get("id"):
        return None
    key = resolve(aliases, b["id"])
    return key if key not in ("unknown", "moving") else None


def _baseline(rows, at, p, predictions):
    # Use only self-reported scales, never predicted fatigue as a reference label.
    ref = conditional(rows, at)
    labels = annotations(rows, at)
    _, aliases = catalog(rows, at)
    alias = aliases["task"]
    key = _behavior(p, alias)
    result = {s: dict(ref[s], basis="personal") for s in SCALES_SUPPORT}
    if key is None:
        return key, result
    days = {s: defaultdict(list) for s in SCALES_SUPPORT}
    for l in current_answers(rows, at):
        if l.at < at - timedelta(days=56):
            continue
        annotation = label_at(labels, "task", l.at)
        if annotation is not None:
            identity = annotation["class_id"]
        else:
            # Use only a prediction actually available at the response cutoff.
            previous = next((v for t, received, v in reversed(predictions)
                             if received <= l.at and timedelta(0) <= l.at - t <= BUCKET), None)
            identity = _behavior(previous or {}, alias)
        if identity != key:
            continue
        for s in SCALES_SUPPORT:
            v = float(l.values[SCALES.index(s)])
            if math.isfinite(v):
                days[s][l.at.date()].append(v)
    for s, by_day in days.items():
        if len(by_day) >= 3:
            values = [mean(v) for v in by_day.values()]
            result[s] = dict(status="available", basis="behavior", days=len(values),
                             mean=mean(values), sd=max(0.5, pstdev(values)))
    return key, result


def evaluate(rows, at, mode="real", config=None):
    """Evaluate the last completed 5-minute bucket, requiring actual coverage.
    All comparisons are causal, with distinct buckets and the same model ID.
    """
    at = dt(at)
    config = config or {}
    thresholds = settings(config)
    last = bucket(at) - BUCKET
    result = dict(version="discovery-break-v1", bucket=last.isoformat(),
                  data_mode=mode, eligible=False, status="insufficient_data",
                  reason="no_consecutive_observed_predictions", thresholds=thresholds,
                  scales={}, triggered=[], behavior_id=None, model_id=None)
    if support_mode(config) == "off":
        return dict(result, status="disabled", reason="disabled")
    # Combined mode has no single participant/scale history; don't mix populations.
    if mode == "combined":
        return dict(result, reason="combined_history_not_supported")
    rows = available_rows(rows, at)
    rows = [r for r in rows if not (r["kind"] == "narrative" and is_support_assessment(r["payload"]))]
    predictions = _predictions(rows, at, mode)
    coverage = defaultdict(list)
    for r in rows:
        if r["kind"] in ("session", "switchbot", "switchbot_observation"):
            t = dt(r["at"])
            coverage[bucket(t)].append(max(t, dt(r["payload"].get("received_at", r["at"]))))
    windows = {}
    for t, received, p in predictions:
        b = bucket(t)
        if b > last or received >= b + BUCKET:
            continue
        if any(known <= t for known in coverage[b]):
            windows[b] = (t, p)
    if last not in windows or last - BUCKET not in windows:
        return result
    now_t, now = windows[last]
    prior_t, prior = windows[last - BUCKET]
    result["model_id"] = now["model_id"]
    if now["model_id"] != prior["model_id"]:
        return dict(result, reason="model_changed")

    def flags(b, t, p):
        identity, refs = _baseline(rows, t, p, predictions)
        old = windows.get(b - timedelta(minutes=30))
        old_p = old[1] if old and old[1]["model_id"] == p["model_id"] else None
        values = {}
        for s in SCALES_SUPPORT:
            v = _value(p["current"].get(s))
            ref = refs[s]
            item = dict(value=v, baseline=ref, difference=None, z=None,
                        change_30min=None, flags=[])
            values[s] = item
            if v is None or ref.get("status") != "available":
                continue
            direction = DIRECTIONS[s]
            diff = direction * (v - ref["mean"])
            z = diff / max(ref["sd"], 0.5)
            item.update(difference=diff, z=z)
            if (v >= thresholds["high"] if direction == 1 else v <= thresholds["low_focus"]):
                item["flags"].append("absolute")
            if diff >= thresholds["delta"] and z >= thresholds["z"]:
                item["flags"].append("relative")
            old_v = _value(old_p["current"].get(s)) if old_p else None
            if old_v is not None:
                change = direction * (v - old_v)
                item["change_30min"] = change
                floor = thresholds["trend_floor"] if direction == 1 else 6 - thresholds["trend_floor"]
                if change >= thresholds["trend_delta"] and (v >= floor if direction == 1 else v <= floor):
                    item["flags"].append("trend")
        return identity, values

    key, current = flags(last, now_t, now)
    prev_key, previous = flags(last - BUCKET, prior_t, prior)
    triggers = []
    for s in SCALES_SUPPORT:
        shared = set(current[s]["flags"]) & set(previous[s]["flags"])
        # Avoid treating a change of Behavior-conditioned reference as persistence.
        if key != prev_key and (current[s]["baseline"]["basis"] == "behavior"
                                or previous[s]["baseline"]["basis"] == "behavior"):
            shared.discard("relative")
        triggers.extend(dict(scale=s, reason=reason) for reason in sorted(shared))
    usable = any(v["value"] is not None and v["baseline"]["status"] == "available"
                 and previous[s]["value"] is not None and previous[s]["baseline"]["status"] == "available"
                 for s, v in current.items())
    return dict(result, status="available" if usable else "insufficient_data",
                reason="persistent_condition" if triggers else
                       "no_persistent_condition" if usable else "insufficient_scale_or_reference",
                eligible=bool(triggers), behavior_id=key, scales=current,
                previous_scales=previous, triggered=triggers)


def rank_actions(rows, at, decision):
    """Singleton-consequent rules + explicit previous usefulness, not RL."""
    def clip(v):
        return max(0.0, min(1.0, v))
    items = decision.get("scales", {})
    strengths = {}
    for s in SCALES_SUPPORT:
        item = items.get(s, {})
        v = item.get("value")
        if v is None or item.get("baseline", {}).get("status") != "available":
            strengths[s] = 0.0
        else:
            absolute = (v - 2) / 2 if DIRECTIONS[s] == 1 else (4 - v) / 2
            strengths[s] = max(clip(absolute), clip((item.get("difference") or 0) / 2),
                               clip((item.get("change_30min") or 0) / 2))
    rules = [(strengths["fatigue"], (1.0, 0.6, 0.3)),
             (strengths["concentration"], (0.5, 0.6, 1.0)),
             (strengths["sleepiness"], (0.7, 1.0, 0.4))]
    total = sum(w for w, _ in rules)
    scores = {a: sum(w * v[i] for w, v in rules) / total if total else 0.0
              for i, a in enumerate(ACTIONS)}
    rows = available_rows(rows, at)
    questions = {r["id"]: r["payload"] for r in rows if r["kind"] == "question_issued"}
    outcomes = defaultdict(list)
    for r in rows:
        p = r["payload"]
        if r["kind"] != "research_event" or p.get("kind") != "intervention_outcome":
            continue
        body = p["payload"]
        q = questions.get(body.get("question_id"), {})
        quality = {"helpful": 1, "unchanged": 0, "worse": -1}.get(body.get("helpfulness"))
        if (q.get("eventType", "").startswith(BREAK) and body.get("executed") is True
                and quality is not None and body.get("action") in ACTIONS):
            outcomes[body["action"]].append(quality)
    for a, values in outcomes.items():
        scores[a] += 0.15 * mean(values)
    return [dict(action=a, label=ACTIONS[a], score=round(scores[a], 3))
            for a in sorted(scores, key=lambda a: (-scores[a], list(ACTIONS).index(a)))]


def scores_from_answer(row):
    if not row:
        return {}
    try:
        scores = json.loads(row["payload"].get("answer", {}).get("free_text") or "{}")
    except (ValueError, TypeError):
        return {}
    if not isinstance(scores, dict):
        return {}
    return {k: v for k, v in scores.items() if k in SCALES
            and type(v) in (int, float) and math.isfinite(v) and 1 <= v <= 5}


def _answers(rows):
    return {r["id"]: r for r in rows if r["kind"] == "narrative"}


def _within(q, answer, at):
    return bool(answer and dt(q["at"]) <= dt(answer["at"]) <= dt(q["payload"]["expiresAt"])
                and dt(answer["at"]) <= at)


def _execution(rows, qid, action, accepted_at):
    return next((r for r in reversed(rows) if r["kind"] == "research_event"
                 and r["payload"].get("kind") == "intervention_outcome"
                 and r["payload"]["payload"].get("question_id") == qid
                 and r["payload"]["payload"].get("action") == action
                 and dt(r["at"]) >= accepted_at), None)


def followup(rows, at):
    """Return (active episode, next question spec); no storage or side effects."""
    at = dt(at)
    rows = available_rows(rows, at)
    questions = [r for r in rows if r["kind"] == "question_issued"]
    answers = _answers(rows)
    stopped = {r["payload"]["payload"].get("question_id") for r in rows
               if r["kind"] == "research_event" and r["payload"].get("kind") == "question_state"
               and r["payload"]["payload"].get("state") in ("declined", "stop")}
    pre = next((r for r in reversed(questions)
                if r["payload"].get("eventType", "").startswith(PRE)
                and dt(r["at"]).date() == at.date()), None)
    if not pre or pre["id"] in stopped:
        return False, None
    pre_answer = answers.get(pre["id"])
    if not _within(pre, pre_answer, at) or len(scores_from_answer(pre_answer)) != len(SCALES):
        return False, None
    event = BREAK + pre["id"]
    offer = next((r for r in questions if r["payload"].get("eventType") == event), None)
    if not offer:
        if at - dt(pre_answer["at"]) > timedelta(minutes=15):
            return False, None
        ranked = pre["payload"].get("support", {}).get("actions", [])
        if not ranked:
            return False, None
        return True, dict(kind="breakCheck", event=event, ttl=15,
                          prompt="休憩前の回答を記録しました。今の状態に合わせて、少し気分を変えませんか？",
                          options=[(a["label"], "accept:" + a["action"]) for a in ranked]
                                  + [("今はしない", "declined"), ("提案を停止", "stop")])
    answer = answers.get(offer["id"])
    if offer["id"] in stopped or not _within(offer, answer, at):
        return False, None
    accepted = answer["payload"].get("answer", {}).get("label") or ""
    if not accepted.startswith("accept:") or accepted[7:] not in ACTIONS:
        return False, None
    action = accepted[7:]
    age = at - dt(answer["at"])
    if age > timedelta(hours=3):
        return False, None
    outcome = _execution(rows, offer["id"], action, dt(answer["at"]))
    outcome_question = next((r for r in questions
                             if r["payload"].get("eventType") == "outcome:" + offer["id"]), None)
    if not outcome:
        if outcome_question:
            return False, None
        if age < timedelta(minutes=15):
            return True, None
        return True, dict(kind="supportOutcome", event="outcome:" + offer["id"], ttl=60,
                          related=offer["id"], prompt="先ほどの提案を実行しましたか？",
                          options=[("実行して役立った", f"outcome:{action}:helpful"),
                                   ("実行して変わらない", f"outcome:{action}:unchanged"),
                                   ("実行して悪化した", f"outcome:{action}:worse"),
                                   ("実行していない", f"outcome:{action}:not_executed")])
    if not outcome["payload"]["payload"].get("executed"):
        return False, None
    if at - dt(outcome["at"]) > timedelta(minutes=30):
        return False, None
    post_event = POST + offer["id"]
    if any(r["payload"].get("eventType") == post_event for r in questions):
        return False, None
    return True, dict(kind="currentState", event=post_event, ttl=30,
                      prompt="休憩・気分転換後の現在の状態を教えてください。",
                      options=[("回答を始める", "start_current_state")])


def effects(rows, at):
    """Pair independently supplied answers and explicit execution; partial stays partial."""
    rows = available_rows(rows, dt(at))
    questions = {r["id"]: r for r in rows if r["kind"] == "question_issued"}
    answers = _answers(rows)
    for q in questions.values():
        event = q["payload"].get("eventType", "")
        if not event.startswith(POST):
            continue
        offer = questions.get(event[len(POST):])
        if not offer or not offer["payload"].get("eventType", "").startswith(BREAK):
            continue
        pre = questions.get(offer["payload"]["eventType"][len(BREAK):])
        if not pre or not pre["payload"].get("eventType", "").startswith(PRE):
            continue
        before, after, accept = answers.get(pre["id"]), answers.get(q["id"]), answers.get(offer["id"])
        if not all(_within(question, answer, dt(at)) for question, answer in
                   ((pre, before), (offer, accept), (q, after))):
            continue
        action = (accept["payload"].get("answer", {}).get("label") or "")
        if not action.startswith("accept:") or action[7:] not in ACTIONS:
            continue
        execution = _execution(rows, offer["id"], action[7:], dt(accept["at"]))
        if (not execution or not execution["payload"]["payload"].get("executed")
                or not dt(before["at"]) <= dt(offer["at"]) <= dt(accept["at"])
                <= dt(execution["at"]) <= dt(q["at"]) <= dt(after["at"])):
            continue
        b, a = scores_from_answer(before), scores_from_answer(after)
        delta = {s: a[s] - b[s] for s in SCALES if s in a and s in b}
        signed = [delta[s] * (1 if s in ("mood", "concentration") else -1) for s in delta]
        yield offer["id"], dict(version="discovery-break-v1", action=action[7:],
                               before_question_id=pre["id"], after_question_id=q["id"],
                               before_at=before["at"], after_at=after["at"], before=b, after=a,
                               delta=delta, improvement=mean(signed) if len(signed) == len(SCALES) else None,
                               helpfulness=execution["payload"]["payload"].get("helpfulness"),
                               note="self_reported_change_not_causal_effect")
