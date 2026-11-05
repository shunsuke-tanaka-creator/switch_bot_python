"""Deterministic growing prototype topology over stable observations, not LSTM weights.

This is a masked incremental prototype map, not a reproduction of original GWR.
Replaying append-only source records reconstructs nodes/edges after restart or replication.
"""

from collections import Counter
from datetime import timedelta
import hashlib
import math
from .data import dt

VERSION = "observation-topology-v1"


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

    def merge_nearby(self, node):
        # Centroids may approach one another as evidence grows. Preserve the oldest stable ID.
        for other in list(self.nodes):
            if other is node:
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

    def add(self, identity, at, features):
        if len(features) < self.min_shared:
            return None

        def distance(n):
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
                count=1,
                dates={at.date().isoformat()},
                first_at=at.isoformat(),
                last_at=at.isoformat(),
            )
            self.nodes.append(node)
        elif closest is not None and not novel:
            node = closest
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
        )


def behavior_map(windows, at, radius=0.35, rows=()):
    model = GrowingMap(radius=radius)
    links = Counter()
    visits = sorted(
        (
            r
            for r in available_rows(rows, at)
            if r["kind"] == "research_event"
            and r["payload"]["kind"] == "place_visit"
            and r["payload"]["payload"]["accuracy_m"] <= 150
        ),
        key=lambda r: max(dt(r["at"]), dt(r["payload"].get("received_at", r["at"]))),
    )
    cursor = 0
    visit = None
    last_visit_at = None
    for w in windows:
        while (
            cursor < len(visits)
            and max(
                dt(visits[cursor]["at"]),
                dt(visits[cursor]["payload"].get("received_at", visits[cursor]["at"])),
            )
            <= w["at"]
        ):
            r = visits[cursor]
            if last_visit_at is None or dt(r["at"]) >= last_visit_at:
                visit = r["payload"]["payload"]
                last_visit_at = dt(r["at"])
            cursor += 1
        if w["inserts"] < 20:
            continue
        hour = w["at"].hour + w["at"].minute / 60
        f = dict(
            input=math.log1p(w["inserts"]) / 8,
            corrections=min(w["correction_rate"], 2) / 2,
            pauses=w["pause_rate"],
            hour_sin=math.sin(hour * math.pi / 12),
            hour_cos=math.cos(hour * math.pi / 12),
        )
        identity = model.add(w["id"], w["at"], f)
        if identity:
            # Context links carry provenance; place IDs are never arbitrary Euclidean coordinates.
            if visit and visit["phase"] == "arrival":
                links[(identity, visit["place_id"])] += 1
    result = model.export(at)
    normalized = Counter()
    for (a, b), n in links.items():
        normalized[model.canonical(a), b] += n
    result["place_links"] = [
        dict(pattern=a, place=b, count=n) for (a, b), n in sorted(normalized.items())
    ]
    return result


def place_map(rows, at, min_days=5):
    visits = [
        r
        for r in available_rows(rows, at)
        if r["kind"] == "research_event" and r["payload"]["kind"] == "place_visit"
    ]
    nodes = {}
    edges = Counter()
    trips = []
    last = None
    current = None
    latest = None
    novel = False
    for r in visits:
        p = r["payload"]["payload"]
        place = p["place_id"]
        t = dt(r["at"])
        phase = p["phase"]
        if p.get("accuracy_m", 1e9) > 150:
            continue
        if phase == "arrival":
            novel = place not in nodes
            n = nodes.setdefault(
                place,
                dict(
                    id=place,
                    count=0,
                    dates=set(),
                    hours=Counter(),
                    first_at=t.isoformat(),
                ),
            )
            n["count"] += 1
            n["dates"].add(t.date().isoformat())
            n["hours"][(t.weekday(), t.hour)] += 1
            n["last_at"] = t.isoformat()
            if last and last != place:
                edges[(last, place)] += 1
                trips.append(dict(source=last, target=place, at=t))
            last = place
            current = place
        else:
            # Departure defines the origin even if its arrival was not recorded.
            last = place
            current = None
        latest = dict(at=t.isoformat(), phase=phase, place_id=place)

    def distribution(t, origin=None, before=None):
        candidates = [
            x
            for x in trips
            if (origin is None or x["source"] == origin)
            and (before is None or x["at"] < before)
        ]
        exact = [
            x
            for x in candidates
            if x["at"].weekday() == t.weekday() and abs(x["at"].hour - t.hour) <= 2
        ]
        group = (
            exact
            if len({x["at"].date() for x in exact}) >= min_days
            else [x for x in candidates if abs(x["at"].hour - t.hour) <= 2]
        )
        days = len({x["at"].date() for x in group})
        counts = Counter(x["target"] for x in group)
        if days < min_days:
            return dict(status="insufficient_data", days=days, places=[])
        total = sum(counts.values()) + len(nodes) + 1
        return dict(
            status="predicted",
            days=days,
            places=[
                dict(place_id=k, probability=(v + 1) / total)
                for k, v in counts.most_common()
            ],
            note="smoothed empirical frequency; prediction is not an observed arrival",
        )

    fresh = bool(latest and at - dt(latest["at"]) <= timedelta(hours=24))
    if not fresh:
        current = None
    expected = distribution(at, (current or latest["place_id"]) if fresh else None)
    expected["context"] = "last_observed_origin" if fresh else "time_only"
    anomaly = None
    if (
        latest
        and latest["phase"] == "arrival"
        and (at - dt(latest["at"])).total_seconds() <= 3600
    ):
        trip = next((x for x in reversed(trips) if x["at"] == dt(latest["at"])), None)
        prior = distribution(
            dt(latest["at"]), trip["source"] if trip else None, before=dt(latest["at"])
        )
        target = next(
            (
                x["probability"]
                for x in prior["places"]
                if x["place_id"] == latest["place_id"]
            ),
            0,
        )
        if prior["status"] == "predicted" and (novel or target < 0.15):
            anomaly = dict(
                kind="unusual_place",
                at=latest["at"],
                place_id=latest["place_id"],
                expected=prior,
                observed=True,
            )
    # Departure-time novelty uses only previous departure dates, including those with no known arrival.
    departures = [r for r in visits if r["payload"]["payload"]["phase"] == "departure"]
    time_anomaly = None
    if (
        latest
        and latest["phase"] == "departure"
        and (at - dt(latest["at"])).total_seconds() <= 3600
    ):
        t = dt(latest["at"])
        past = [
            dt(r["at"])
            for r in departures
            if dt(r["at"]) < t and dt(r["at"]).weekday() == t.weekday()
        ]
        hours = [x.hour + x.minute / 60 for x in past]
        if len({x.date() for x in past}) >= min_days:
            # Circular distance avoids treating 23:55 and 00:05 as far apart.
            distance = min(
                min(
                    abs(t.hour + t.minute / 60 - h),
                    24 - abs(t.hour + t.minute / 60 - h),
                )
                for h in hours
            )
            if distance > 2:
                time_anomaly = dict(
                    kind="unusual_departure",
                    at=latest["at"],
                    distance_hours=distance,
                    days=len({x.date() for x in past}),
                )
    return dict(
        nodes=[
            dict(
                id=k,
                count=n["count"],
                days=len(n["dates"]),
                first_at=n["first_at"],
                last_at=n["last_at"],
                schedule=[
                    dict(weekday=d, hour=h, count=c)
                    for (d, h), c in sorted(n["hours"].items())
                ],
            )
            for k, n in nodes.items()
        ],
        edges=[dict(source=a, target=b, count=c) for (a, b), c in edges.items()],
        current=current,
        latest=latest,
        destination=expected,
        anomaly=anomaly,
        time_anomaly=time_anomaly,
    )
