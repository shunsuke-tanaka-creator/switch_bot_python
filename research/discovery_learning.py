"""Causal batch training of a shared open-set model. Legacy checkpoints are untouched."""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import random
import shutil
import numpy as np
import torch
from torch.nn import functional as fn
from .data import FEATURES, SCALES, ACTIVITIES, Timeline, LazySample, dt
from .context_data import observations, cutoffs, annotations, label_at, catalog, fresh
from .topology import available_rows
from .discovery_model import (DiscoveryModel, VERSION, observation_mask, masked_input,
                              masked_mse, identity_loss, variance_loss)
from .discovery_map import DiscoveryMap
from .train import normalization, batch
from .io import atomic_json


def source_rows(rows, at=None):
    """Fixed-category labels cannot affect inputs, sampling, losses or fingerprint."""
    result = []
    for original in (available_rows(rows, dt(at)) if at is not None else rows):
        row = copy.deepcopy(original)
        kind, payload = row["kind"], row["payload"]
        if kind == "narrative" and payload.get("question", {}).get("kind") not in ("currentState", "nightReflection"):
            continue
        if kind == "correction":
            if not payload.get("scores"):
                continue
            payload.pop("activity", None)
        if kind == "research_event" and payload.get("kind") not in ("context_label", "context_name", "context_merge"):
            continue
        if kind not in ("session", "heartbeat", "switchbot", "switchbot_observation", "narrative", "correction", "research_event", "context_candidate"):
            continue
        result.append(row)
    return result


def annotation_digest(rows):
    relevant = [(r["id"], r["payload"]) for r in rows
                if r["kind"] == "research_event" and r["payload"].get("kind") in ("context_label", "context_merge")]
    return hashlib.sha256(json.dumps(sorted(relevant, key=lambda v: v[0]), sort_keys=True).encode()).hexdigest()


class SourceTimelines:
    """Discovery examples retain their own Timeline, including next-bin targets."""
    def __init__(self, timelines):
        self.timelines = timelines
        self.encoder = next(iter(timelines.values())).encoder

    def sample(self, at, scope="current", source=None):
        if source is None:
            from .data import inference_rows
            _, source = inference_rows([r for t in self.timelines.values() for r in t.rows], at)
        return self.timelines[source].sample(at, scope)


def examples_for(rows, at, text_model=None):
    groups = {}
    for row in rows:
        groups.setdefault(row.get("participant", ""), []).append(row)
    if len(groups) <= 1:
        return _source_examples(rows, at, text_model)
    times = cutoffs(observations(source_rows(rows, dt(at))), dt(at))
    dates = sorted({t.date() for t in times})
    if len(dates) < 3:
        return _source_examples(rows, at, text_model)
    boundaries = (dt(str(dates[max(1, int(len(dates)*.65))]) + "T00:00:00+09:00"),
                  dt(str(dates[max(2, int(len(dates)*.85))]) + "T00:00:00+09:00"))
    timelines, examples = {}, []
    for source, group in groups.items():
        timeline, subset, _ = _source_examples(group, at, text_model, boundaries)
        timelines[source] = timeline
        examples.extend(dict(e, source=source) for e in subset)
    split = dict(train_before=boundaries[0].isoformat(), test_from=boundaries[1].isoformat(),
                 windows=len(times), days=len(dates))
    return SourceTimelines(timelines), sorted(examples, key=lambda e: (e["at"], e["source"])), split


def _source_examples(rows, at, text_model=None, boundaries=None):
    """Observation-based day split; questionnaires never change input boundaries."""
    at = dt(at)
    rows = source_rows(rows, at)
    observed = observations(rows)
    timeline = Timeline(observed, text_model=text_model)
    times = cutoffs(observed, at)
    dates = sorted({t.date() for t in times})
    if boundaries is None and (len(times) < 12 or len(dates) < 3):
        return timeline, [], dict(status="insufficient_observations", windows=len(times), days=len(dates))
    validation_at, test_at = boundaries or (
        dt(str(dates[max(1, int(len(dates)*0.65))]) + "T00:00:00+09:00"),
        dt(str(dates[max(2, int(len(dates)*0.85))]) + "T00:00:00+09:00"))
    ends = dict(train=validation_at, validation=test_at, test=at + timedelta(microseconds=1))
    def split(t):
        return "train" if t < validation_at else "validation" if t < test_at else "test"
    labels = [l for l in Timeline(rows).labels if l.scope in ("current", "daily") and l.available <= at]
    axis_labels = {s: annotations(rows, end-timedelta(microseconds=1)) for s,end in ends.items()}
    candidates = set(times)
    for l in labels:
        if at-timedelta(days=28) < l.at <= at and l.available < ends[split(l.at)]:
            candidates.add(l.at)
    for s, anns in axis_labels.items():
        candidates.update(l["end"] for l in anns
                          if at-timedelta(days=28) < l["end"] <= at and split(l["end"]) == s)
    examples = []
    if len(candidates)>480:
        raise ValueError("Too many discovery examples; split the export rather than silently discard labels")
    for t in sorted(candidates):
        s = split(t)
        target_end = t+timedelta(hours=1)
        if target_end < ends[s] and target_end <= at:
            target = timeline.features(t, target_end, target_end)
            mask = observation_mask(torch.from_numpy(target)).numpy()
        else:
            target = np.zeros(len(FEATURES), np.float32)
            mask = np.zeros_like(target)  # Never read a target across a split boundary.
        scores = {scope: np.full(5, np.nan, np.float32) for scope in ("current", "daily")}
        for label in sorted(labels, key=lambda l: l.available):
            if label.at == t and label.available < ends[s]:
                valid = np.isfinite(label.values)
                scores[label.scope][valid] = label.values[valid]
        ids = {}
        for axis, wire in (("place", "place"), ("behavior", "task")):
            annotation = label_at(axis_labels[s], wire, t)
            key = annotation["class_id"] if annotation else None
            ids[axis] = key if key not in ("unknown", "moving") else None
        if not fresh(observed, t):
            continue
        examples.append(dict(at=t, split=s, sample=LazySample(timeline, t),
                             target=target, mask=mask, scores=scores, labels=ids))
    return timeline, examples, dict(train_before=validation_at.isoformat(), test_from=test_at.isoformat(),
                                   train_received_before=validation_at.isoformat(),
                                   validation_received_before=test_at.isoformat(), observed_through=at.isoformat())


def evaluate(model, examples, device, mean, std, reference_std=None):
    """Compare old/new normalizers in the same target error scale."""
    sums = {s: np.zeros(5) for s in ("current", "daily")}
    counts = {s: np.zeros(5, dtype=int) for s in sums}
    huber = {s: 0.0 for s in sums}
    future_error = future_n = 0.0
    model.eval()
    with torch.no_grad():
        for e in examples:
            out = model(batch([e["sample"]], device, mean, std))
            target = torch.tensor(e["target"], device=device)
            mask = torch.tensor(e["mask"], device=device)
            error = (out["next"][0]*std+mean-target)/(std if reference_std is None else reference_std)
            future_error += float((error.square()*mask).sum())
            future_n += float(mask.sum())
            for s in sums:
                y = torch.tensor(e["scores"][s], device=device)
                valid = torch.isfinite(y)
                if valid.any():
                    d = (out[s][0]-torch.nan_to_num(y)).abs().cpu().numpy()
                    valid_np = valid.cpu().numpy()
                    sums[s][valid_np] += d[valid_np]
                    counts[s] += valid_np
                    huber[s] += float(fn.smooth_l1_loss(out[s][0][valid], y[valid], reduction="sum"))
    terms = ([future_error/future_n] if future_n else [])
    terms += [huber[s]/counts[s].sum() for s in sums if counts[s].sum()]
    result = {s: {k: dict(n=int(counts[s][i]), mae=float(sums[s][i]/counts[s][i]) if counts[s][i] else None)
                  for i,k in enumerate(SCALES)} for s in sums}
    result.update(objective=float(sum(terms)) if terms else None, future_n=int(future_n),
                  future_mse=future_error/future_n if future_n else None)
    return result


def build_maps(model, examples, timeline, rows, before, device, mean, std):
    maps = {a: DiscoveryMap(a, dimension=model.settings["embedding_dim"]) for a in ("place", "behavior")}
    known, aliases = catalog(rows, before-timedelta(microseconds=1))
    model.eval()
    with torch.no_grad():
        for axis, wire in (("place", "place"), ("behavior", "task")):
            maps[axis].aliases = dict(aliases[wire])
            for key, node in sorted(known[wire].items(), key=lambda kv: (not kv[1]["confirmed"],kv[0])):
                t = dt(node["anchor"])
                if t >= before or not fresh(observations(rows), t):
                    continue
                if isinstance(timeline, SourceTimelines):
                    source = next((r.get("participant", "") for r in rows
                                   if r["kind"] in ("research_event", "context_candidate")
                                   and (r["payload"].get("class_id") == key
                                        or r["payload"].get("payload", {}).get("class_id") == key)), None)
                    sample = timeline.sample(t, source=source)
                else:
                    sample = timeline.sample(t)
                out = model(batch([sample], device, mean, std))
                maps[axis].seed(out[axis][0].cpu().numpy(), key, t.isoformat(), node["confirmed"])
        for e in sorted(examples, key=lambda e: e["at"]):
            out = model(batch([e["sample"]], device, mean, std))
            for a, mapping in maps.items():
                mapping.observe(out[a][0].cpu().numpy(), e["at"].isoformat(), e["labels"][a])
    return maps


def map_metrics(model, examples, maps, device, mean, std):
    result = {a: dict(n=0, correct=0, unknown=0) for a in maps}
    model.eval()
    with torch.no_grad():
        for e in examples:
            out = model(batch([e["sample"]], device, mean, std))
            for a, mapping in maps.items():
                predicted = mapping.predict(out[a][0].cpu().numpy())
                result[a]["unknown"] += int(predicted["candidate_id"] is None)
                if e["labels"][a] is not None:
                    result[a]["n"] += 1
                    result[a]["correct"] += int(predicted["candidate_id"] == e["labels"][a])
    for r in result.values():
        r["accuracy"] = r["correct"]/r["n"] if r["n"] else None
    return result


def train_discovery(rows, directory, at, epochs=12, hidden=48, text_model=None,
                    device="cpu", force=False, evaluate_test=False,
                    data_mode="real", contains_sample=False, sample_manifest_hash=None):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    from .context_learning import device_for
    from .text import encoder_identity
    at, directory = dt(at), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sanitized = source_rows(rows,at)
    signature = dict(version=VERSION, rows=sanitized, epochs=epochs, hidden=hidden, data_mode=data_mode,
                     text=encoder_identity(text_model) if text_model else "character-ngram-v1",
                     rolling_hour=at.replace(minute=0, second=0, microsecond=0).isoformat())
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
    with (directory/"training.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="busy", version=VERSION)
        active, status_path = directory/"active.pt", directory/"status.json"
        if not force and not evaluate_test and active.exists() and status_path.exists():
            old_report = json.loads(status_path.read_text())
            if old_report.get("experiment_id") == run_id:
                return dict(old_report, unchanged=True)
        random.seed(37); np.random.seed(37); torch.manual_seed(37)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        timeline, examples, split = examples_for(sanitized, at, text_model)
        report = dict(version=VERSION, architecture="discovery", experiment_id=run_id, fingerprint=run_id,
                      data_mode=data_mode, contains_sample=contains_sample,
                      sample_manifest_hash=sample_manifest_hash,
                      trained_through=at.isoformat(), activity_role="posthoc_only", seed=37,
                      text_model=str(text_model) if text_model else None,
                      text_identity=signature["text"], annotation_digest=annotation_digest(sanitized), **split)
        def stop(status):
            report["status"] = status
            atomic_json(status_path, report)
            return report
        if split.get("status"):
            return stop(split["status"])
        tr, va, te = ([e for e in examples if e["split"]==s] for s in ("train", "validation", "test"))
        if len(tr)<4 or not va:
            return stop("insufficient_split")
        dev = device_for(device)
        mean_np, std_np = normalization([(None,e["sample"]) for e in tr])
        mean, std = (torch.tensor(x, device=dev) for x in (mean_np,std_np))
        model = DiscoveryModel(hidden=hidden).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        best, best_score = None, float("inf")
        for _ in range(epochs):
            model.train()
            order = list(tr); random.shuffle(order)
            for offset in range(0,len(order),8):
                chunk = order[offset:offset+8]
                b = batch([e["sample"] for e in chunk],dev,mean,std)
                out = model(b)
                masked, target_now, selected = masked_input(b)
                reconstruction = model(masked)["reconstruction"]
                future = (torch.tensor(np.stack([e["target"] for e in chunk]),device=dev)-mean)/std
                future_mask = torch.tensor(np.stack([e["mask"] for e in chunk]),device=dev)
                loss = masked_mse(out["next"],future,future_mask) + masked_mse(reconstruction,target_now,selected)
                for s in ("current","daily"):
                    y = torch.tensor(np.stack([e["scores"][s] for e in chunk]),device=dev)
                    valid = torch.isfinite(y)
                    if valid.any():
                        loss = loss + fn.smooth_l1_loss(out[s][valid],y[valid])
                for a in ("place","behavior"):
                    loss = loss + 0.1*identity_loss(out[a],[e["labels"][a] for e in chunk]) + 0.1*variance_loss(out[a])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite discovery loss")
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                optimizer.step()
            score = evaluate(model,va,dev,mean,std)["objective"]
            if score is not None and np.isfinite(score) and score < best_score:
                best_score = score
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best is None:
            return stop("insufficient_validation")
        model.load_state_dict(best)
        maps = build_maps(model,tr,timeline,sanitized,dt(split["train_before"]),dev,mean,std)
        # Descriptions are fitted ONLY after geometry and weights have been fixed.
        model.eval()
        with torch.no_grad():
            for label in Timeline(available_rows(rows,at)).labels:
                if label.scope != "activity" or max(label.at,label.available) >= dt(split["train_before"]):
                    continue
                sample = timeline.sample(label.at, source=label.source) if isinstance(timeline, SourceTimelines) else timeline.sample(label.at)
                out = model(batch([sample],dev,mean,std))
                key = maps["behavior"].predict(out["behavior"][0].cpu().numpy())["candidate_id"]
                if key:
                    maps["behavior"].annotate(key, ACTIVITIES[label.activity])
        report.update(status="trained", settings=model.settings, train_n=len(tr), validation_n=len(va), test_n=len(te),
                      support={s:[int(sum(np.isfinite(e["scores"][s][i]) for e in tr)) for i in range(5)] for s in ("current","daily")},
                      validation=evaluate(model,va,dev,mean,std), validation_axes=map_metrics(model,va,maps,dev,mean,std),
                      test=evaluate(model,te,dev,mean,std) if evaluate_test else None,
                      test_axes=map_metrics(model,te,maps,dev,mean,std) if evaluate_test else None,
                      example_times={s:[e["at"].isoformat() for e in es] for s,es in (("train",tr),("validation",va),("test",te))},
                      limitations=["Axis separation is not guaranteed", "Cosine similarity is not calibrated",
                                   "Maps are rebuilt on training data; live inference is frozen", "No fixed activity classifier"])
        report["support_by_source"] = {
            source: {scope: {name: dict(
                train=int(sum(e["sample"].timeline.rows[0].get("participant", "") == source
                              and np.isfinite(e["scores"][scope][i]) for e in tr)),
                validation=int(sum(e["sample"].timeline.rows[0].get("participant", "") == source
                                   and np.isfinite(e["scores"][scope][i]) for e in va)))
                for i,name in enumerate(SCALES)} for scope in ("current","daily")}
            for source in {r.get("participant", "") for r in sanitized}}
        report["validation_by_source"] = {
            source: evaluate(model,[e for e in va if e["sample"].timeline.rows[0].get("participant", "") == source],dev,mean,std)
            for source in report["support_by_source"]}
        promoted = True
        if active.exists():
            old = torch.load(active,map_location=dev,weights_only=True)
            if old.get("version")==VERSION and old.get("features")==FEATURES and old["report"].get("text_identity")==report["text_identity"]:
                previous = DiscoveryModel(**old["settings"]).to(dev)
                previous.load_state_dict(old["weights"])
                old_score = evaluate(previous,va,dev,torch.tensor(old["mean"],device=dev),torch.tensor(old["std"],device=dev),std)["objective"]
                promoted = old_score is None or (np.isfinite(old_score) and best_score <= old_score)
        report["promoted"] = promoted
        state = dict(version=VERSION,features=FEATURES,weights=best,settings=model.settings,
                     mean=mean_np.tolist(),std=std_np.tolist(),maps={a:m.to_dict() for a,m in maps.items()},report=report)
        torch.save(state,directory/"candidate.tmp"); os.replace(directory/"candidate.tmp",directory/"candidate.pt")
        if promoted:
            if active.exists(): shutil.copy2(active,directory/"previous.pt")
            torch.save(state,directory/"active.tmp"); os.replace(directory/"active.tmp",active)
        atomic_json(status_path,report)
        return report
