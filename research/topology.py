"""Deterministic growing prototype topology over stable observations, not LSTM weights.

This is a masked incremental prototype map, not a reproduction of original GWR.
Replaying append-only source records reconstructs nodes/edges after restart or replication.
"""

from collections import Counter
from datetime import timedelta
import hashlib
import math
from .data import dt

VERSION = "observation-topology-v2-state-context"


def stable_id(prefix, value):
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:16]


def available_rows(rows, at):
    return sorted(
        (
            r
            for r in rows
            if dt(r["at"]) <= at and dt(r["payload"].get("received_at", r["at"])) <= at
        ),
        key=lambda r: (dt(r["at"]), r.get("seq", 0), r["id"]),
    )


def typing_windows(rows, at, minutes=10):
    """Non-overlapping fixed JST windows; physical inserts and user deletions only."""
    events = {}
    meta = {}
    for r in available_rows(rows, at):
        if r["kind"] != "session":
            continue
        p = r["payload"]
        device = p.get("device_id", "legacy")
        for i, e in enumerate(p.get("events", [])):
            data = e.get("data", {})
            if data.get("purpose") == "assessment" or data.get("synthetic"):
                continue
            if e.get("type") not in ("kana", "text_insert", "delete"):
                continue
            # `key` includes navigation/modifiers. Kana is exactly one insertion path.
            t = dt(e["time"])
            if t > at:
                continue
            bucket = t.replace(
                minute=(t.minute // minutes) * minutes, second=0, microsecond=0
            )
            if bucket + timedelta(minutes=minutes) > at:
                continue
            mode = data.get("input_mode", "kana")
            key = (bucket, device, mode)
            eid = (p.get("session_id", r["id"]), i)
            events.setdefault(key, {})[eid] = (t, e["type"], data)
            meta[key] = dict(device=device, mode=mode)
    output = []
    for key, by_id in sorted(events.items()):
        start = key[0]
        ordered = sorted(by_id.values(), key=lambda x: x[0])
        inserts = deletes = bursts = 0
        last_delete = None
        times = []
        for t, kind, data in ordered:
            if kind == "delete":
                deletes += 1
                if last_delete is None or (t - last_delete).total_seconds() > 2:
                    bursts += 1
                last_delete = t
            else:
                inserts += 1
                times.append(t)
                last_delete = None
        intervals = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        active = sum(min(x, 2) for x in intervals)
        output.append(
            dict(
                id=stable_id("typing:", start.isoformat() + key[1] + key[2]),
                at=start,
                end=start + timedelta(minutes=minutes),
                inserts=inserts,
                deletes=deletes,
                correction_rate=bursts / max(inserts, 1),
                delete_rate=deletes / max(inserts, 1),
                speed=inserts * 60 / active if active > 0 else None,
                pause_rate=sum(x > 2 for x in intervals) / max(len(intervals), 1),
                **meta[key],
            )
        )
    return output


class GrowingMap:
    def __init__(self, radius=0.35, min_shared=3, max_nodes=256):
        self.radius = radius
        self.min_shared = min_shared
        self.max_nodes = max_nodes
        self.nodes = []
        self.edges = Counter()
        self.last = None
        self.last_at = None
        self.aliases = {}

    def canonical(self, identity):
        while identity in self.aliases:
            identity = self.aliases[identity]
        return identity

    def label_key(self, labels):
        return tuple((labels.get(axis) or {}).get("id") for axis in ("place", "action"))

    def merge_nearby(self, node):
        # Centroids may approach one another as evidence grows. Preserve the oldest stable ID.
        for other in list(self.nodes):
            if other is node:
                continue
            # 追加: 場所・アクションが異なる生活状態を同じノードへ統合しない。
            if self.label_key(other["labels"]) != self.label_key(node["labels"]):
                continue
            common = other["center"].keys() & node["center"].keys()
            if len(common) < self.min_shared:
                continue
            d = math.sqrt(
                sum((other["center"][k] - node["center"][k]) ** 2 for k in common)
                / len(common)
            )
            if d > self.radius / 2:
                continue
            keep, drop = sorted([node, other], key=lambda n: (n["first_at"], n["id"]))
            for k in drop["center"]:
                a = keep["counts"].get(k, 0)
                b = drop["counts"][k]
                keep["center"][k] = (
                    keep["center"].get(k, 0) * a + drop["center"][k] * b
                ) / (a + b)
                keep["counts"][k] = a + b
            keep["count"] += drop["count"]
            keep["dates"] |= drop["dates"]
            keep["last_at"] = max(keep["last_at"], drop["last_at"])
            self.aliases[drop["id"]] = keep["id"]
            self.nodes.remove(drop)
            node = keep
            edges = Counter()
            for (a, b), n in self.edges.items():
                a = self.canonical(a)
                b = self.canonical(b)
                if a != b:
                    edges[a, b] += n
            self.edges = edges
            if self.last:
                self.last = self.canonical(self.last)
        return node

    def add(self, identity, at, features, labels=None):
        if len(features) < self.min_shared:
            return None
        labels = dict(labels or {})

        def distance(n):
            # 追加: トポロジカル状態は場所・アクションを条件にして分離する。
            if self.label_key(n["labels"]) != self.label_key(labels):
                return float("inf")
            common = features.keys() & n["center"].keys()
            if len(common) < self.min_shared:
                return float("inf")
            return math.sqrt(
                sum((features[k] - n["center"][k]) ** 2 for k in common) / len(common)
            )

        closest = min(self.nodes, key=distance, default=None)
        novel = closest is None or distance(closest) > self.radius
        if novel and len(self.nodes) < self.max_nodes:
            node = dict(
                id=stable_id("pattern:", identity),
                center=dict(features),
                counts={k: 1 for k in features},
                labels=labels,
                count=1,
                dates={at.date().isoformat()},
                first_at=at.isoformat(),
                last_at=at.isoformat(),
            )
            self.nodes.append(node)
        elif closest is not None and not novel:
            node = closest
            # 追加: IDを維持したまま、後から得た表示名だけを更新する。
            for axis in ("place", "action"):
                if (labels.get(axis) or {}).get("name"):
                    node["labels"][axis] = labels[axis]
            node["count"] += 1
            node["dates"].add(at.date().isoformat())
            node["last_at"] = at.isoformat()
            for k, v in features.items():
                count = node["counts"].get(k, 0) + 1
                node["center"][k] = (
                    node["center"].get(k, v) + (v - node["center"].get(k, v)) / count
                )
                node["counts"][k] = count
        else:
            return None  # capacity is explicit; never merge a truly novel observation into a distant class
        node = self.merge_nearby(node)
        if (
            self.last
            and self.last != node["id"]
            and (at - self.last_at).total_seconds() <= 12 * 3600
        ):
            self.edges[(self.last, node["id"])] += 1
        self.last = node["id"]
        self.last_at = at
        return node["id"]

    def export(self, at):
        return dict(
            version=VERSION,
            nodes=[
                dict(
                    id=n["id"],
                    center=n["center"],
                    place=n["labels"].get("place"),
                    action=n["labels"].get("action"),
                    count=n["count"],
                    days=len(n["dates"]),
                    first_at=n["first_at"],
                    last_at=n["last_at"],
                    status="established" if len(n["dates"]) >= 3 else "emerging",
                    activation=n["count"]
                    * math.exp(-(at - dt(n["last_at"])).total_seconds() / (30 * 86400)),
                )
                for n in self.nodes
            ],
            edges=[
                dict(source=a, target=b, count=n)
                for (a, b), n in sorted(self.edges.items())
            ],
            capacity=self.max_nodes,
            at_capacity=len(self.nodes) >= self.max_nodes,
            aliases=dict(self.aliases),
        )


def behavior_map(windows, at, radius=0.35, rows=()):
    """Behavior prototypes use typing features within exact place/action states."""
    # 追加: 永続化済みContext推定から、各窓を特徴づける場所・アクションを解決する。
    visible = available_rows(rows, at)
    context_rows = [r for r in visible if r["kind"] == "context_inference"]
    from .context_data import catalog, resolve

    classes, aliases = catalog(visible, at)

    def labels_for(window):
        candidates = [
            r
            for r in context_rows
            if abs((dt(r["at"]) - window["end"]).total_seconds()) <= 10 * 60
            and (r["payload"].get("place") or {}).get("id")
            and (r["payload"].get("task") or {}).get("id")
        ]
        if not candidates:
            return {}
        payload = min(
            candidates,
            key=lambda r: abs((dt(r["at"]) - window["end"]).total_seconds()),
        )["payload"]
        place = payload["place"]
        action = payload["task"]
        place_id = resolve(aliases["place"], place["id"])
        action_id = resolve(aliases["task"], action["id"])
        return dict(
            place=dict(
                id=place_id,
                name=place.get("name") or classes["place"].get(place_id, {}).get("name"),
            ),
            action=dict(
                id=action_id,
                name=action.get("name") or classes["task"].get(action_id, {}).get("name"),
            ),
        )

    model = GrowingMap(radius=radius)
    assignments = []
    for window in windows:
        if window['inserts'] < 20:
            continue
        hour = window['at'].hour + window['at'].minute / 60
        labels = labels_for(window)
        state_id = model.add(window['id'],window['at'],dict(
            input=math.log1p(window['inserts'])/8,
            corrections=min(window['correction_rate'],2)/2,
            pauses=window['pause_rate'],hour_sin=math.sin(hour*math.pi/12),hour_cos=math.cos(hour*math.pi/12)),
            labels=labels)
        if state_id:
            # 追加: 歩数スナップショットと結びつけるため、再計算可能な状態割当てを公開する。
            assignments.append(dict(
                state_id=state_id,
                at=window["at"].isoformat(),
                end=window["end"].isoformat(),
                place=labels.get("place"),
                action=labels.get("action"),
                source_id=window["id"],
            ))
    result = model.export(at)
    for assignment in assignments:
        assignment["state_id"] = model.canonical(assignment["state_id"])
    result["assignments"] = assignments
    return result
