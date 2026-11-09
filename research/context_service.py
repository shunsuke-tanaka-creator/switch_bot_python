"""Portable lifecycle and append-only live/review records for behavioral context."""

from datetime import timedelta
from pathlib import Path
import json
from uuid import uuid4
from .data import dt, Timeline, FEATURES
from .context_data import (
    AXES,
    VERSION,
    observations,
    annotations,
    label_at,
    catalog,
    resolve,
    fresh,
    cutoffs,
    summary,
    transitions,
)


class ContextEngine:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.cache = {}

    def directory(self, p, mode="real"):
        return (
            Path(self.config.get("research_models", "data/models"))
            / p
            / mode
            / "context"
        )

    def load(self, p, mode="real"):
        path = self.directory(p, mode) / "active.pt"
        if not path.exists():
            return None
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.cache:
            return self.cache[key]
        import torch
        from .context_model import ContextLSTM

        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != VERSION or state.get("features") != FEATURES:
            return None
        model = ContextLSTM(state["hidden"])
        model.load_state_dict(state["weights"])
        model.eval()
        self.cache = {key: (state, model)}
        return state, model

    def train(self, p, rows, at, mode="real"):
        from .context_learning import train_context

        result = train_context(
            rows,
            self.directory(p, mode),
            at,
            epochs=self.config.get("context_epochs", 12),
            text_model=self.config.get("text_model"),
            device=self.config.get("context_device", "auto"),
        )
        loaded = self.load(p, mode)
        if mode == "real" and loaded:
            for a in loaded[0].get("anchors", []):
                self.store.save(
                    "context_candidate",
                    a["class_id"],
                    p,
                    dt(a["anchor"]),
                    dict(a, received_at=dt(at).isoformat()),
                )
        return result

    def infer(
        self, p, at, rows=None, mode="real", retrospective=False, available_at=None
    ):
        at = dt(at)
        known_at = dt(available_at or at)
        rows = self.store.rows(p) if rows is None else rows
        result = dict(
            at=at.isoformat(),
            available_at=known_at.isoformat(),
            mode="review" if retrospective else "live",
            status="untrained",
            place=None,
            task=None,
            version=VERSION,
            gps_used=False,
        )
        nodes, aliases = catalog(rows, known_at)
        labels = annotations(rows, known_at)
        loaded = self.load(p, mode)
        if loaded:
            state, model = loaded
            report = state["report"]
            result["model"] = report
            if not retrospective and dt(report["trained_through"]) > at:
                result["status"] = "checkpoint_from_future"
            elif not fresh(rows, at):
                result["status"] = "observation_unknown"
            else:
                import torch
                from .train import batch
                from .context_learning import classify, annotation_fingerprint

                # Text path may change on migration, but weights/encoder identity must match.
                text_model = self.config.get("text_model") or report.get("text_model")
                timeline = Timeline(observations(rows), text_model=text_model)
                identity = (
                    timeline.encoder.identity
                    if timeline.encoder
                    else "character-ngram-v1"
                )
                if identity != report["text_identity"]:
                    result["status"] = "text_encoder_changed"
                else:
                    with torch.no_grad():
                        out = model(
                            batch(
                                [timeline.sample(at)],
                                "cpu",
                                torch.tensor(state["mean"]),
                                torch.tensor(state["std"]),
                            )
                        )
                    result["status"] = "ready"
                    for a in AXES:
                        ranked = classify(out[a][0].numpy(), state["prototypes"][a])
                        if not ranked:
                            continue
                        best = ranked[0]
                        key = resolve(aliases[a], best["id"])
                        validation = report["validation"][a]
                        margin = (
                            best["score"] - ranked[1]["score"] if len(ranked) > 1 else 0
                        )
                        accepted = (
                            report.get("annotation_fingerprint")
                            == annotation_fingerprint(rows)
                            and best["confirmed"]
                            and best["days"] >= 2
                            and validation["n"] >= 2
                            and (validation["accuracy"] or 0) >= 0.5
                            and best["score"] >= 0.8
                            and margin >= 0.08
                        )
                        result[a] = dict(
                            id=key if accepted else None,
                            candidate_id=key if best["score"] >= 0.65 else None,
                            status="estimated" if accepted else "unconfirmed",
                            name=nodes[a].get(key, {}).get("name"),
                            similarity=best["score"],
                            margin=margin,
                            source="model",
                            note="similarity is not a calibrated probability",
                        )
        # Fresh explicit labels take precedence, independently on each axis. Never propagate over gaps.
        for a in AXES:
            label = label_at(labels, a, at)
            if label:
                key = label["class_id"]
                result[a] = dict(
                    id=None if key in ("unknown", "moving") else key,
                    status=key if key in ("unknown", "moving") else "confirmed",
                    source="self_report",
                    name=nodes[a].get(key, {}).get("name"),
                )
        return result

    def review(self, p, day, at):
        at = dt(at)
        start = dt(day + "T00:00:00+09:00")
        end = start + timedelta(days=1)
        if end > at:
            raise ValueError("Review requires a completed JST day")
        rows = self.store.rows(p)
        times = [t for t in cutoffs(rows, end) if start <= t < end]
        times.extend(l["end"] for l in annotations(rows, at) if start <= l["end"] < end)
        episodes = [
            self.infer(p, t, rows, retrospective=True, available_at=at)
            for t in sorted(set(times))
        ]
        live = [
            r["payload"]
            for r in reversed(rows)
            if r["kind"] == "context_inference" and start <= dt(r["at"]) < end
        ]
        result = dict(
            day=day,
            created_at=at.isoformat(),
            episodes=episodes,
            original_live=live,
            edges=transitions(episodes),
            note="Retrospective reanalysis; never an online accuracy evaluation",
        )
        self.store.save("context_review", str(uuid4()), p, at, result)
        return result

    def snapshot(self, p, at, rows=None):
        rows = self.store.rows(p) if rows is None else rows
        result = summary(rows, dt(at))
        result["reviews"] = [
            r["payload"] for r in rows if r["kind"] == "context_review"
        ][:7]
        result["history"] = [
            r["payload"] for r in reversed(rows) if r["kind"] == "context_inference"
        ][-288:]
        result["edges"] = transitions(result["history"])
        path = self.directory(p) / "status.json"
        result["training"] = (
            json.loads(path.read_text()) if path.exists() else dict(status="untrained")
        )
        return result
