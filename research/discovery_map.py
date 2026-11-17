"""Open-set prototypes, optional human IDs and directed temporal edges.

Only observe() changes geometry. predict() is read-only; annotations cannot change
assignment. Rebuild from source anchors after every encoder update.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import math
import numpy as np


def _time(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("Timezone required")
    return t


class DiscoveryMap:
    def __init__(self, axis, dimension=32, threshold=0.88, max_nodes=256,
                 max_gap_seconds=3600):
        if axis not in ("place", "behavior"):
            raise ValueError("Unknown discovery axis")
        if dimension < 2 or max_nodes < 1 or not math.isfinite(threshold) or not -1 < threshold <= 1:
            raise ValueError("Invalid discovery map settings")
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("Invalid transition gap")
        self.axis, self.dimension = axis, dimension
        self.threshold, self.max_nodes = threshold, max_nodes
        self.max_gap_seconds = max_gap_seconds
        self.nodes, self.edges, self.aliases = {}, {}, {}
        self.last = None
        self.overflow = 0

    def canonical(self, identity):
        seen = set()
        while identity in self.aliases:
            if identity in seen:
                raise ValueError("Cyclic identity alias")
            seen.add(identity)
            identity = self.aliases[identity]
        return identity

    def vector(self, value):
        v = np.asarray(value, dtype=np.float64)
        if v.shape != (self.dimension,) or not np.isfinite(v).all():
            raise ValueError("Embedding has wrong shape or non-finite values")
        norm = float(np.linalg.norm(v))
        if norm <= 1e-8:
            raise ValueError("Zero embedding")
        return v / norm

    def rank(self, value):
        v = self.vector(value)
        return sorted((dict(id=k, similarity=float(np.clip(np.dot(v, n["vector"]), -1, 1)))
                       for k, n in self.nodes.items()),
                      key=lambda item: (-item["similarity"], item["id"]))

    def predict(self, value):
        ranked = self.rank(value)
        if not ranked:
            return dict(id=None, candidate_id=None, status="unknown", similarity=None, margin=None)
        best = ranked[0]
        accepted = best["similarity"] >= self.threshold
        return dict(id=None, candidate_id=best["id"] if accepted else None,
                    status="candidate" if accepted else "unknown",
                    similarity=best["similarity"],
                    margin=best["similarity"]-ranked[1]["similarity"] if len(ranked)>1 else None)

    def seed(self, value, identity, anchor, confirmed=False):
        """Re-embed stable source anchors; never reuse a previous latent center."""
        identity = self.canonical(identity)
        if identity in self.nodes:
            self.nodes[identity]["confirmed"] |= bool(confirmed)
            return
        _time(anchor)
        # Bound unconfirmed candidates, never silently discard a human identity.
        if not confirmed and sum(not n["confirmed"] for n in self.nodes.values()) >= self.max_nodes:
            self.overflow += 1
            return
        self.nodes[identity] = dict(id=identity, vector=self.vector(value).tolist(),
                                   anchor=anchor, count=0, days=[], confirmed_days=[],
                                   confirmed=bool(confirmed), annotation_counts={})

    def observe(self, value, at, identity=None):
        timestamp = _time(at)
        if self.last and timestamp <= _time(self.last[1]):
            raise ValueError("Map observations must be strictly chronological")
        v = self.vector(value)
        ranked = self.rank(v)
        key = self.canonical(identity) if identity is not None else None
        if key is None and ranked and ranked[0]["similarity"] >= self.threshold:
            key = ranked[0]["id"]
        if key is None:
            prefix = "place" if self.axis == "place" else "task"  # legacy wire contract
            key = self.canonical(prefix + "_" + sha256(at.encode()).hexdigest()[:16])
        if key not in self.nodes:
            self.seed(v, key, at, confirmed=identity is not None)
        if key not in self.nodes:
            self.last = None  # Never bridge a capacity/unknown gap.
            return None
        node = self.nodes[key]
        center = np.asarray(node["vector"])*node["count"] + v
        norm = np.linalg.norm(center)
        node["vector"] = (center/norm if norm > 1e-8 else v).tolist()
        node["count"] += 1
        day = timestamp.date().isoformat()
        node["days"] = sorted(set(node["days"]) | {day})
        if identity is not None:
            node["confirmed"] = True
            node["confirmed_days"] = sorted(set(node["confirmed_days"]) | {day})
        if self.last and 0 < (timestamp-_time(self.last[1])).total_seconds() <= self.max_gap_seconds:
            if self.last[0] != key:
                edge = (self.last[0], key)
                self.edges[edge] = self.edges.get(edge, 0) + 1
        self.last = (key, at)
        return key

    def annotate(self, identity, label):
        """Post-hoc descriptions only: no effect on geometry or confirmation."""
        if identity not in self.nodes:
            raise ValueError("Unknown map identity")
        counts = self.nodes[identity]["annotation_counts"]
        counts[label] = counts.get(label, 0) + 1

    def to_dict(self):
        return dict(axis=self.axis, dimension=self.dimension, threshold=self.threshold,
                    max_nodes=self.max_nodes, max_gap_seconds=self.max_gap_seconds,
                    nodes=deepcopy(list(self.nodes.values())), overflow=self.overflow,
                    aliases=deepcopy(self.aliases),
                    edges=[dict(source=a, target=b, count=n) for (a,b),n in sorted(self.edges.items())])

    @classmethod
    def from_dict(cls, state):
        obj = cls(**{k: state[k] for k in ("axis", "dimension", "threshold", "max_nodes", "max_gap_seconds")})
        obj.aliases = dict(state.get("aliases", {}))
        for key in obj.aliases:
            obj.canonical(key)
        for node in state["nodes"]:
            obj.vector(node["vector"])
            _time(node["anchor"])
            if node["id"] in obj.nodes or obj.canonical(node["id"]) != node["id"]:
                raise ValueError("Duplicate or non-canonical map identity")
            obj.nodes[node["id"]] = deepcopy(node)
        if sum(not n["confirmed"] for n in obj.nodes.values()) > obj.max_nodes:
            raise ValueError("Map exceeds candidate capacity")
        obj.edges = {(e["source"],e["target"]):int(e["count"]) for e in state["edges"]}
        if any(a not in obj.nodes or b not in obj.nodes or n < 1 for (a,b),n in obj.edges.items()):
            raise ValueError("Invalid map edge")
        obj.overflow = int(state.get("overflow", 0))
        return obj
