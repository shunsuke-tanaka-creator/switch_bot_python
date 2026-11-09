"""Independent GPS-free training lifecycle. Portable CPU checkpoint; causal day split."""

from datetime import timedelta
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import random
import numpy as np
import torch
from .data import BASE, F, FEATURES, TEXT_DIM, Timeline, dt
from .train import atomic_json, normalization, batch
from .context_data import (
    AXES,
    VERSION,
    annotations,
    label_at,
    observations,
    cutoffs,
    ident,
    catalog,
)
from .context_model import ContextLSTM, relation_loss


def fingerprint(rows):
    source = [
        (r["kind"], r["id"], r["at"], r["payload"])
        for r in rows
        if r["kind"] in ("session", "switchbot", "switchbot_observation", "heartbeat")
        or (
            r["kind"] == "research_event"
            and r["payload"]["kind"].startswith("context_")
        )
    ]
    return hashlib.sha256(
        json.dumps(
            sorted(source, key=lambda r: (r[2], r[0], r[1])), sort_keys=True
        ).encode()
    ).hexdigest()


def annotation_fingerprint(rows):
    source = [
        (r["id"], r["at"], r["payload"])
        for r in rows
        if r["kind"] == "research_event"
        and r["payload"]["kind"] in ("context_label", "context_merge")
    ]
    return hashlib.sha256(
        json.dumps(sorted(source, key=lambda r: (r[1], r[0])), sort_keys=True).encode()
    ).hexdigest()


def device_for(requested="auto"):
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("context_device: auto/cpu/cuda/mps required")
    # CPU is the portable Mac default; CUDA is used on a capable Ubuntu host.
    chosen = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else requested
    )
    if chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if chosen == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return torch.device(chosen)


def prototypes(model, examples, device, mean, std):
    values = {a: {} for a in AXES}
    days = {a: {} for a in AXES}
    model.eval()
    with torch.no_grad():
        for e in examples:
            out = model(batch([e["sample"]], device, mean, std))
            for a in AXES:
                key = e["labels"][a]
                if key:
                    values[a].setdefault(key, []).append(out[a][0].cpu().numpy())
                    days[a].setdefault(key, set()).add(e["at"].date())
    result = {a: [] for a in AXES}
    for a in AXES:
        for key, z in values[a].items():
            center = np.mean(z, axis=0)
            center /= max(np.linalg.norm(center), 1e-8)
            result[a].append(
                dict(
                    id=key,
                    vector=center.tolist(),
                    count=len(z),
                    days=len(days[a][key]),
                    confirmed=True,
                )
            )
    return result


def classify(vector, nodes):
    if not nodes:
        return []
    return sorted(
        [
            dict(
                id=n["id"],
                score=float(np.dot(vector, n["vector"])),
                days=n["days"],
                confirmed=n["confirmed"],
            )
            for n in nodes
        ],
        key=lambda n: -n["score"],
    )


def evaluate(model, examples, proto, device, mean, std):
    result = {a: dict(n=0, correct=0, cross_context_n=0) for a in AXES}
    model.eval()
    with torch.no_grad():
        for e in examples:
            out = model(batch([e["sample"]], device, mean, std))
            for a in AXES:
                key = e["labels"][a]
                if not key:
                    continue
                ranked = classify(out[a][0].cpu().numpy(), proto[a])
                result[a]["n"] += 1
                result[a]["correct"] += int(bool(ranked) and ranked[0]["id"] == key)
                if all(e["labels"].values()):
                    result[a]["cross_context_n"] += 1
    for a in AXES:
        r = result[a]
        r["accuracy"] = r["correct"] / r["n"] if r["n"] else None
    return result


def train_context(
    rows,
    directory,
    at,
    epochs=12,
    hidden=24,
    text_model=None,
    device="auto",
    evaluate_test=False,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    at = dt(at)
    fp = fingerprint(rows)
    with (directory / "training.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="busy")
        status_path = directory / "status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if (
                previous.get("fingerprint") == fp
                and previous.get("text_model") == text_model
                and not evaluate_test
            ):
                return dict(previous, unchanged=True)
        random.seed(37)
        np.random.seed(37)
        torch.manual_seed(37)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        timeline = Timeline(observations(rows), text_model=text_model)
        labels = annotations(rows, at)
        times = set(cutoffs(rows, at))
        times.update(
            l["end"] for l in labels if at - timedelta(days=28) < l["end"] <= at
        )
        times = sorted(times)[-240:]
        dates = sorted({t.date() for t in times})
        report = dict(
            version=VERSION,
            annotation_fingerprint=annotation_fingerprint(rows),
            fingerprint=fp,
            text_model=text_model,
            windows=len(times),
            days=len(dates),
            trained_through=at.isoformat(),
            gps_used=False,
        )
        if len(times) < 12 or len(dates) < 3:
            report.update(
                status="insufficient_observations", required_windows=12, required_days=3
            )
            atomic_json(status_path, report)
            return report
        va_day = dates[max(1, int(len(dates) * 0.65))]
        test_day = dates[max(2, int(len(dates) * 0.85))]
        labels_by_split = {
            "train": annotations(
                rows, dt(str(va_day) + "T00:00:00+09:00") - timedelta(microseconds=1)
            ),
            "validation": annotations(
                rows, dt(str(test_day) + "T00:00:00+09:00") - timedelta(microseconds=1)
            ),
            "test": labels,
        }
        examples = []
        for t in times:
            target_end = t + timedelta(hours=1)
            if target_end > at:
                continue
            target = timeline.features(t, target_end)
            # Presence/clock bits alone are not evidence for a learnable behavior target.
            mask = np.zeros(F, np.float32)
            mask[: len(BASE)] = target[len(BASE) : 2 * len(BASE)]
            for j, k in enumerate(BASE):
                if k.endswith(("_sin", "_cos")) or k in (
                    "elapsed",
                    "heartbeat_age",
                    "collection_enabled",
                ):
                    mask[j] = 0
            if target[BASE.index("text_chars") + len(BASE)]:
                mask[-TEXT_DIM:] = 1
            split = (
                "train"
                if target_end.date() < va_day
                else "validation"
                if t.date() >= va_day and target_end.date() < test_day
                else "test"
                if t.date() >= test_day
                else None
            )
            if split is None:
                continue
            # Replay only labels/merges known at that split. A late correction must not
            # erase an earlier valid label or relabel training classes from the future.
            labs = {a: label_at(labels_by_split[split], a, t) for a in AXES}
            ids = {
                a: (
                    l["class_id"]
                    if l and l["class_id"] not in ("unknown", "moving")
                    else None
                )
                for a, l in labs.items()
            }
            if mask.sum() == 0 and not any(ids.values()):
                continue
            examples.append(
                dict(
                    at=t,
                    sample=timeline.sample(t),
                    labels=ids,
                    target=target,
                    mask=mask,
                    split=split,
                )
            )
        tr = [e for e in examples if e["split"] == "train"]
        va = [e for e in examples if e["split"] == "validation"]
        te = [e for e in examples if e["split"] == "test"]
        if len(tr) < 4 or not va:
            report.update(status="insufficient_split")
            atomic_json(status_path, report)
            return report
        mean_np, std_np = normalization([(None, e["sample"]) for e in tr])
        dev = device_for(device)
        mean = torch.tensor(mean_np, device=dev)
        std = torch.tensor(std_np, device=dev)
        model = ContextLSTM(hidden).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        best = None
        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            random.shuffle(tr)
            for offset in range(0, len(tr), 8):
                chunk = tr[offset : offset + 8]
                out = model(batch([e["sample"] for e in chunk], dev, mean, std))
                target = (
                    torch.tensor(np.stack([e["target"] for e in chunk]), device=dev)
                    - mean
                ) / std
                mask = torch.tensor(np.stack([e["mask"] for e in chunk]), device=dev)
                loss = (
                    (out["next"] - target).square() * mask
                ).sum() / mask.sum().clamp_min(1)
                for a in AXES:
                    loss = loss + relation_loss(out[a], [e["labels"][a] for e in chunk])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            model.eval()
            total = denom = 0
            with torch.no_grad():
                for e in va:
                    out = model(batch([e["sample"]], dev, mean, std))
                    target = (torch.tensor(e["target"], device=dev) - mean) / std
                    mask = torch.tensor(e["mask"], device=dev)
                    total += float(((out["next"][0] - target).square() * mask).sum())
                    denom += float(mask.sum())
            score = total / max(denom, 1)
            if score < best_loss:
                best_loss = score
                best = copy.deepcopy(model.state_dict())
        if best is None:
            raise ValueError("No finite context validation loss")
        model.load_state_dict(best)
        proto = prototypes(model, tr, dev, mean, std)
        report.update(
            status="trained",
            hidden=hidden,
            device=str(dev),
            train_before=str(va_day),
            test_from=str(test_day),
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            validation_loss=best_loss,
            validation=evaluate(model, va, proto, dev, mean, std),
            test=evaluate(model, te, proto, dev, mean, std) if evaluate_test else None,
            text_identity=timeline.encoder.identity
            if timeline.encoder
            else "character-ngram-v1",
            note="Separate metric heads; self-supervised candidates are unconfirmed, similarity is not a probability",
        )
        # Stable identities are anchored in source observations, NOT old latent coordinates.
        known, aliases = catalog(rows, at)
        anchors = []
        model.eval()
        with torch.no_grad():
            for a in AXES:
                for key, n in known[a].items():
                    if any(x["id"] == key for x in proto[a]):
                        continue
                    t = dt(n["anchor"])
                    if t.date() >= va_day:
                        continue
                    z = (
                        model(batch([timeline.sample(t)], dev, mean, std))[a][0]
                        .cpu()
                        .numpy()
                    )
                    proto[a].append(
                        dict(
                            id=key,
                            vector=z.tolist(),
                            count=0,
                            days=0,
                            confirmed=False,
                            anchor=t.isoformat(),
                        )
                    )
            encoded = [
                (
                    e,
                    {
                        a: z[0].cpu().numpy()
                        for a, z in model(batch([e["sample"]], dev, mean, std)).items()
                        if a in AXES
                    },
                )
                for e in sorted(tr, key=lambda e: e["at"])
            ]
            for e, out in encoded:
                for a in AXES:
                    z = out[a]
                    rank = classify(z, proto[a])
                    if not rank or rank[0]["score"] < 0.88:
                        if len(proto[a]) >= 64:
                            continue
                        key = ident(a, e["at"].isoformat())
                        if key in aliases[a]:
                            continue
                        if any(n["id"] == key for n in proto[a]):
                            continue
                        node = dict(
                            id=key,
                            vector=z.tolist(),
                            count=1,
                            days=1,
                            confirmed=False,
                            anchor=e["at"].isoformat(),
                        )
                        proto[a].append(node)
                        anchors.append(
                            dict(axis=a, class_id=key, anchor=node["anchor"])
                        )
            # Count candidate support without ever treating the cluster assignment as a label.
            for a in AXES:
                for node in proto[a]:
                    if node["confirmed"]:
                        continue
                    matches = []
                    for e, out in encoded:
                        z = out[a]
                        if float(np.dot(z, node["vector"])) >= 0.88:
                            matches.append(e["at"])
                    node["count"] = len(matches)
                    node["days"] = len({t.date() for t in matches})
        checkpoint = dict(
            version=VERSION,
            features=FEATURES,
            weights={k: v.cpu() for k, v in best.items()},
            hidden=hidden,
            mean=mean_np.tolist(),
            std=std_np.tolist(),
            report=report,
            prototypes=proto,
            anchors=anchors,
        )
        temp = directory / "active.tmp"
        torch.save(checkpoint, temp)
        os.replace(temp, directory / "active.pt")
        atomic_json(status_path, report)
        return report
