"""Causal provisional estimates and audited per-item serving decisions.

No training, promotion, question-budget changes, or torch dependency at import.
"""
from bisect import bisect_right
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from uuid import uuid4

import numpy as np
from .data import SCALES, Timeline, dt
from .experiment_metrics import sequential_baselines
from .io import atomic_json
from .questions import settings


SCOPES = ("current", "daily")
THRESHOLDS = dict(train_n=10, validation_n=10, validation_days=3,
                  absolute_margin=.1, relative_ratio=.9, win_days=3,
                  live_n=10, degradation_margin=.1, revert_days=2)
VERSION = "per-item-delivery-v1"


def enabled(engine, participant, at):
    # Settings always belong to the real participant, including sample previews.
    rows = [r for r in engine.rows(participant, "real")
            if dt(r["at"]) <= at and dt(r["payload"].get("received_at", r["at"])) <= at]
    return settings(rows)["baseline_estimates_enabled"]


def item_state():
    return dict(source="baseline", win_streak=0, last_win_day=None, win_answer_ids=[],
                revert_streak=0, last_revert_day=None, revert_answer_ids=[], switched_at=None)


def load_state(directory, recover=False):
    path = directory / "delivery-state.json"
    state = json.loads(path.read_text()) if path.exists() else dict(
        version=VERSION, items={}, audit_id=None)
    # Append precedes atomic state replacement. Replay a committed audit suffix
    # if the worker stopped between those writes.
    audit = directory / "delivery-audit.jsonl"
    if recover and audit.exists():
        records = [json.loads(line) for line in audit.read_text().splitlines() if line]
        start = next((i + 1 for i, r in enumerate(records) if r["id"] == state["audit_id"]), 0)
        for record in records[start:]:
            state["items"][record["item_key"]] = record["state_after"]
            state["audit_id"] = record["id"]
    return state


@contextmanager
def state_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "delivery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def answer_identity(label):
    # Include cutoff/reception times so the audit identifies the exact label.
    return json.dumps([label.id, label.scope, label.at.isoformat(), label.available.isoformat()],
                      separators=(",", ":"))


def read_checkpoint(path):
    """Weights and content ID from the same file even during atomic promotion."""
    import torch
    with path.open("rb") as stream:
        model_id = hashlib.sha256(stream.read()).hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    return saved, model_id


def answered_predictions(records, labels, at, mode="real"):
    """One saved pre-question estimate per answered item, never a reconstructed fit."""
    at = dt(at)
    predictions = sorted(
        (r for r in records if r["kind"] == "inference"
         and r["payload"].get("data_mode", "real") == mode
         and dt(r["at"]) < at and dt(r["payload"].get("at", r["at"])) < at
         and any(item.get("baselines") for scope in SCOPES
                 for item in (r["payload"].get(scope) or {}).values())),
        key=lambda r: (dt(r["payload"].get("at", r["at"])), dt(r["at"]), r.get("seq", 0)))
    times = [dt(r["payload"].get("at", r["at"])) for r in predictions]
    result = {f"{scope}.{name}": [] for scope in SCOPES for name in SCALES}
    used = {key: set() for key in result}
    for label in sorted(labels, key=lambda l: (l.available, l.at, l.id)):
        if label.scope not in SCOPES or label.available >= at:
            continue
        stop = bisect_right(times, label.at)
        for i, name in enumerate(SCALES):
            if not np.isfinite(label.values[i]):
                continue
            key = f"{label.scope}.{name}"
            for index in range(stop - 1, -1, -1):
                record = predictions[index]
                payload = record["payload"]
                # Prediction content and persistence must both precede reception.
                if dt(record["at"]) >= label.available:
                    continue
                item = (payload.get(label.scope) or {}).get(name) or {}
                baselines = item.get("baselines") or {}
                mean = (baselines.get("mean") or {}).get("value")
                last = (baselines.get("last") or {}).get("value")
                if not all(finite(v) for v in (mean, last)):
                    continue
                # A repeated answer/correction does not create another prediction.
                if record["id"] in used[key]:
                    break
                used[key].add(record["id"])
                lstm = item.get("lstm_value", item.get("value") if item.get("source") == "lstm" else None)
                result[key].append(dict(answer_id=answer_identity(label), prediction_id=record["id"],
                                        at=payload.get("at", record["at"]), saved_at=record["at"],
                                        available=label.available.isoformat(), target=float(label.values[i]),
                                        mean=float(mean), last=float(last), lstm=lstm,
                                        source=item.get("source"), model_id=item.get("model_id")))
                break
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def maes(points, methods=("mean", "last", "lstm")):
    return {method: float(np.mean([abs(p[method] - p["target"]) for p in points]))
            if points and all(finite(p.get(method)) for p in points) else None
            for method in methods}


def apply_delivery(engine, participant, at, mode, rows, response):
    """Overlay serving flags; keep original LSTM gates and shadow predictions."""
    # Every new item estimate has explicit provenance even with the setting off.
    for scope in SCOPES:
        for item in (response.get(scope) or {}).values():
            item.update(source="lstm", provisional=False, baseline_method=None)
    if not enabled(engine, participant, at):
        return response
    labels = Timeline(rows).labels
    state = load_state(engine.directory(participant, mode))
    answered = answered_predictions(engine.store.rows(participant), labels, at, mode)
    model_id = response.get("model_id")
    for scope in SCOPES:
        values = sequential_baselines(labels, at, scope)
        if response.get(scope) is None and not any(v["n"] for v in values.values()):
            continue
        response[scope] = response.get(scope) or {}
        for name in SCALES:
            key = f"{scope}.{name}"
            baseline = values[name]
            history = answered[key]
            scores = maes(history, ("mean", "last"))
            method = "mean" if (scores["mean"] is not None and scores["last"] is not None
                                and scores["mean"] < scores["last"]) else "last"
            item = response[scope].setdefault(name, dict(value=None, status="insufficient_labels"))
            lstm = item["value"]
            source = state["items"].get(key, item_state())["source"]
            item.update(source=source, provisional=source == "baseline",
                        baseline_method=method if source == "baseline" and baseline["n"] else None,
                        baselines={m: dict(value=baseline[m], mae=scores[m], n=len(history))
                                   for m in ("mean", "last")},
                        lstm_value=lstm, model_id=model_id, prior_answers=baseline["n"])
            if source == "baseline":
                item["value"] = baseline[method]
                item["status"] = "estimated" if baseline["n"] else "insufficient_labels"
            elif not baseline["n"]:
                item["value"] = None
                item["status"] = "insufficient_labels"
    if any(item["value"] is not None for scope in SCOPES for item in (response.get(scope) or {}).values()):
        response["lstm_status"] = response["status"]
        response["status"] = "ready"
    return response


def validation_evidence(engine, participant, mode, rows, at):
    """Score the active weights and both sequential baselines on exact split IDs."""
    import torch
    from .data import VERSION as FEATURE_VERSION, FEATURES
    from .model import HierarchicalLSTM
    from .train import batch

    checkpoint = engine.read_path(participant, mode, "active.pt")
    # Read a single checkpoint snapshot, including its content identity.
    saved, model_id = read_checkpoint(checkpoint)
    active_report = saved.get("report") or {}
    latest = engine.status(participant, mode)
    report = latest if latest.get("run", {}).get("splits") else active_report
    splits = report.get("run", {}).get("splits") or {}
    active_splits = active_report.get("run", {}).get("splits") or {}
    split = dict(train_before=active_report.get("train_before"), validation_from=report.get("train_before"),
                 test_from=report.get("test_from"), train=active_splits.get("train", []),
                 validation=splits.get("validation", []))
    split_id = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    evidence = dict(model_id=model_id, split_id=split_id, split=split, items={})
    for scope in SCOPES:
        for i, name in enumerate(SCALES):
            evidence["items"][f"{scope}.{name}"] = dict(
                train_n=active_report.get("support", {}).get(scope, [0] * 5)[i],
                validation_n=0, validation_days=0, answer_ids=[], maes=maes([]), points=[])
    if saved.get("version") != FEATURE_VERSION or saved.get("features") != FEATURES:
        evidence["reason"] = "incompatible_checkpoint"
        return evidence
    timeline = Timeline(rows, text_model=engine.config.get("text_model") or active_report.get("text_model"))
    identity = timeline.encoder.identity if timeline.encoder else "character-ngram-v1"
    if identity != active_report.get("text_identity", identity):
        evidence["reason"] = "text_encoder_changed"
        return evidence
    specs = {(s["id"], dt(s["at"]), dt(s["available"])) for s in split["validation"]}
    trained_ids = {s["id"] for s in active_splits.get("train", [])}
    examples = [(l, sample) for l, sample in timeline.examples()
                if l.scope in SCOPES and (l.id, l.at, l.available) in specs
                and l.available < at and l.id not in trained_ids]
    model = HierarchicalLSTM(**saved["settings"])
    model.load_state_dict(saved["weights"])
    model.eval()
    mean, std = torch.tensor(saved["mean"]), torch.tensor(saved["std"])
    with torch.no_grad():
        for label, sample in examples:
            prediction = model(batch([sample], "cpu", mean, std))[label.scope][0].numpy()
            baselines = sequential_baselines(timeline.labels, label.at, label.scope)
            for i, name in enumerate(SCALES):
                baseline = baselines[name]
                if not np.isfinite(label.values[i]) or not baseline["n"]:
                    continue
                item = evidence["items"][f"{label.scope}.{name}"]
                item["points"].append(dict(answer_id=answer_identity(label), at=label.at.isoformat(),
                                           available=label.available.isoformat(), target=float(label.values[i]),
                                           mean=baseline["mean"], last=baseline["last"], lstm=float(prediction[i])))
    for item in evidence["items"].values():
        item["validation_n"] = len(item["points"])
        item["validation_days"] = len({dt(p["at"]).date() for p in item["points"]})
        item["answer_ids"] = [p["answer_id"] for p in item["points"]]
        item["maes"] = maes(item["points"])
    return evidence


def decide(item, validation, live, at):
    """Pure per-item state transition; repeated data/days never advance a streak."""
    item = copy.deepcopy(item)
    day = at.date().isoformat()
    if item["source"] == "baseline":
        scores = validation["maes"]
        quantity = (validation["train_n"] >= THRESHOLDS["train_n"]
                    and validation["validation_n"] >= THRESHOLDS["validation_n"]
                    and validation["validation_days"] >= THRESHOLDS["validation_days"])
        better = min(scores["mean"], scores["last"]) if all(finite(scores[m]) for m in ("mean", "last")) else None
        quality = (finite(scores["lstm"]) and better is not None
                   and scores["lstm"] <= better - THRESHOLDS["absolute_margin"] + 1e-12
                   and scores["lstm"] <= THRESHOLDS["relative_ratio"] * better + 1e-12)
        new = set(validation["answer_ids"]) - set(item["win_answer_ids"])
        new_day = item["last_win_day"] is None or day > item["last_win_day"]
        reason = "quantity_failed" if not quantity else "quality_failed" if not quality else "awaiting_new_validation"
        if not quantity or not quality:
            item["win_streak"] = 0
        elif new and new_day:
            item["win_streak"] += 1
            item["last_win_day"] = day
            item["win_answer_ids"] = sorted(set(item["win_answer_ids"]) | set(validation["answer_ids"]))
            reason = "qualifying_day"
            if item["win_streak"] >= THRESHOLDS["win_days"]:
                item.update(source="lstm", switched_at=at.isoformat(), revert_streak=0,
                            last_revert_day=None, revert_answer_ids=[])
                return item, "switch", "three_qualifying_days", dict(quantity=quantity, quality=quality)
        return item, "hold", reason, dict(quantity=quantity, quality=quality)
    live = [p for p in live if p["source"] == "lstm" and finite(p["lstm"])
            and (not item["switched_at"] or dt(p["saved_at"]) >= dt(item["switched_at"]))][-THRESHOLDS["live_n"]:]
    scores = maes(live)
    enough = len(live) == THRESHOLDS["live_n"]
    degraded = (enough and all(finite(v) for v in scores.values())
                and scores["lstm"] >= min(scores["mean"], scores["last"]) + THRESHOLDS["degradation_margin"] - 1e-12)
    new = {p["answer_id"] for p in live} - set(item["revert_answer_ids"])
    new_day = item["last_revert_day"] is None or day > item["last_revert_day"]
    checks = dict(live_n=len(live), live_maes=scores, degraded=degraded,
                  live_answer_ids=[p["answer_id"] for p in live], live_points=live,
                  model_ids=sorted({p["model_id"] for p in live if p.get("model_id")}))
    reason = "insufficient_live_predictions" if not enough else "live_quality_ok" if not degraded else "awaiting_new_live_answers"
    if not degraded:
        item["revert_streak"] = 0
    elif new and new_day:
        item["revert_streak"] += 1
        item["last_revert_day"] = day
        item["revert_answer_ids"] = sorted(set(item["revert_answer_ids"]) | {p["answer_id"] for p in live})
        reason = "degraded_day"
        if item["revert_streak"] >= THRESHOLDS["revert_days"]:
            item.update(source="baseline", win_streak=0, last_win_day=day,
                        win_answer_ids=sorted(set(item["win_answer_ids"]) | set(validation["answer_ids"])))
            return item, "revert", "two_degraded_days", checks
    return item, "hold", reason, checks


def evaluate_delivery(engine, participant, mode="real", at=None):
    """Called after hourly training/promotion, including no-new-label cycles."""
    from datetime import datetime, timezone
    at = dt(at or datetime.now(timezone.utc))
    if not enabled(engine, participant, at) or not engine.read_path(participant, mode, "active.pt").exists():
        return None
    rows = engine.rows(participant, mode)
    evidence = validation_evidence(engine, participant, mode, rows, at)
    live = answered_predictions(engine.store.rows(participant), Timeline(rows).labels, at, mode)
    directory = engine.directory(participant, mode)
    evaluation_id = str(uuid4())
    decisions = []
    with state_lock(directory):
        state = load_state(directory, recover=True)
        with (directory / "delivery-audit.jsonl").open("a", encoding="utf-8") as stream:
            for key, validation in evidence["items"].items():
                before = state["items"].get(key, item_state())
                after, decision, reason, checks = decide(before, validation, live[key], at)
                state["items"][key] = after
                record = dict(id=str(uuid4()), evaluation_id=evaluation_id, version=VERSION,
                              participant=participant, data_mode=mode, at=at.isoformat(), item_key=key,
                              model_id=evidence["model_id"], split_id=evidence["split_id"], split=evidence["split"],
                              train_n=validation["train_n"], validation_n=validation["validation_n"],
                              validation_days=validation["validation_days"], validation_maes=validation["maes"],
                              validation_answer_ids=validation["answer_ids"], thresholds=THRESHOLDS,
                              win_streak=after["win_streak"], revert_streak=after["revert_streak"],
                              source_before=before["source"], source_after=after["source"],
                              decision=decision, reason=reason, validation_status=evidence.get("reason", "evaluated"),
                              checks=checks, state_after=after)
                if before["source"] == "lstm":
                    record["prediction_model_ids"] = checks["model_ids"]
                stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
                state["audit_id"] = record["id"]
                decisions.append(record)
            stream.flush()
            os.fsync(stream.fileno())
        state.update(participant=participant, data_mode=mode, updated_at=at.isoformat())
        atomic_json(directory / "delivery-state.json", state)
    return decisions
