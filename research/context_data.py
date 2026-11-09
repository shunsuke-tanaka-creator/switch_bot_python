"""GPS-free, independently annotated place/task identities and causal observations."""

from collections import Counter
from datetime import timedelta
import hashlib
from .data import dt
from .topology import available_rows

AXES = ("place", "task")
VERSION = "behavior-context-v1"


def ident(axis, anchor):
    return axis + "_" + hashlib.sha256(anchor.encode()).hexdigest()[:16]


def observations(rows):
    # Whitelist; never use place_visit, RSSI, Wi-Fi IDs, questionnaires or predictions.
    return [
        r
        for r in rows
        if r["kind"] in ("session", "switchbot", "switchbot_observation", "heartbeat")
    ]


def catalog(rows, at):
    """Replay append-only names/merges. Split = label a selected interval with a new ID."""
    nodes = {a: {} for a in AXES}
    aliases = {a: {} for a in AXES}
    for r in available_rows(rows, at):
        p = r["payload"]
        kind = r["kind"]
        if kind == "context_candidate":
            a, key = p["axis"], p["class_id"]
            nodes[a].setdefault(
                key, dict(id=key, name=None, anchor=p["anchor"], confirmed=False)
            )
        if kind != "research_event" or p["kind"] not in (
            "context_label",
            "context_name",
            "context_merge",
        ):
            continue
        body = p["payload"]
        a = body["axis"]
        if p["kind"] == "context_merge":
            src, dst = body["source_id"], body["target_id"]
            # Flatten through current aliases, preventing cycles even when replayed after sync.
            src = resolve(aliases[a], src)
            dst = resolve(aliases[a], dst)
            if src != dst:
                aliases[a][src] = dst
        else:
            key = body["class_id"]
            if key in ("unknown", "moving"):
                continue
            node = nodes[a].setdefault(
                key,
                dict(
                    id=key, name=None, anchor=body.get("end", r["at"]), confirmed=False
                ),
            )
            if p["kind"] == "context_label":
                node["confirmed"] = True
            else:
                node["name"] = body["name"]
    for a in AXES:
        for key in list(nodes[a]):
            target = resolve(aliases[a], key)
            if key != target:
                old = nodes[a].pop(key)
                node = nodes[a].setdefault(target, dict(old, id=target))
                node["confirmed"] = node["confirmed"] or old["confirmed"]
    return nodes, aliases


def resolve(aliases, key):
    seen = set()
    while key in aliases and key not in seen:
        seen.add(key)
        key = aliases[key]
    return key


def annotations(rows, at):
    _, aliases = catalog(rows, at)
    out = []
    for r in available_rows(rows, at):
        p = r["payload"]
        if r["kind"] == "research_event" and p["kind"] == "context_label":
            b = p["payload"]
            axis = b["axis"]
            out.append(
                dict(
                    axis=axis,
                    class_id=resolve(aliases[axis], b["class_id"]),
                    start=dt(b["start"]),
                    end=dt(b["end"]),
                    available=max(dt(r["at"]), dt(p.get("received_at", r["at"]))),
                )
            )
    return out


def label_at(labels, axis, at):
    # Last correction wins for this axis only; labels never become input features.
    return next(
        (
            l
            for l in reversed(labels)
            if l["axis"] == axis and l["start"] < at <= l["end"]
        ),
        None,
    )


def cutoffs(rows, at, limit=240):
    """At most one sample/hour plus labelled endpoints; retained recent 28 days."""
    rows = available_rows(observations(rows), at)
    times = {
        dt(r["at"]).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        for r in rows
        if dt(r["at"]) > at - timedelta(days=28)
        and r["kind"] in ("session", "switchbot")
    }
    return sorted(t for t in times if t <= at)[-limit:]


def fresh(rows, at):
    # A stationary door advertising every minute cannot establish the user's presence.
    return any(
        r["kind"] == "session"
        and timedelta(0) <= at - dt(r["at"]) <= timedelta(hours=1)
        for r in available_rows(rows, at)
    )


def summary(rows, at):
    nodes, aliases = catalog(rows, at)
    latest = next(
        (
            r["payload"]
            for r in reversed(available_rows(rows, at))
            if r["kind"] == "context_inference"
            and (at - dt(r["at"])).total_seconds() <= 180
        ),
        None,
    )
    return dict(
        version=VERSION,
        nodes={a: list(nodes[a].values()) for a in AXES},
        current=latest
        or dict(status="worker_not_running_or_stale", place=None, task=None),
        gps_used=False,
    )


def transitions(episodes):
    edges = {a: Counter() for a in AXES}
    for left, right in zip(episodes, episodes[1:]):
        if dt(right["at"]) - dt(left["at"]) > timedelta(hours=1):
            continue
        for a in AXES:
            x = (left.get(a) or {}).get("id")
            y = (right.get(a) or {}).get("id")
            if x and y and x != y:
                edges[a][x, y] += 1
    return {
        a: [dict(source=x, target=y, count=n) for (x, y), n in edges[a].items()]
        for a in AXES
    }
