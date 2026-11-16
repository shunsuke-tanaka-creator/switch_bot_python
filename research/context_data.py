"""GPS-free, independently annotated place/task identities and causal observations."""

from collections import Counter, defaultdict
from datetime import timedelta
from heapq import heappop, heappush
import hashlib
from .data import dt
from .baseline import period
from .topology import available_rows

AXES = ("place", "task")
VERSION = "behavior-context-v2-received-cutoff"


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


def node_profiles(rows, at):
    """Self-confirmed intervals only, with the same precedence as label_at().

    A partially corrected interval counts once, using its last surviving endpoint.
    Fully superseded intervals (including duplicate reports) contribute no vote.
    Dates and answer bands refer to these endpoints in JST, never model visits.
    """
    at = dt(at)
    nodes, _ = catalog(rows, at)
    labels = annotations(rows, at)
    profiles = {
        axis: {
            key: dict(
                confirmed_count=0,
                confirmed_days=0,
                last_confirmed_at=None,
                dominant_band=None,
            )
            for key in nodes[axis]
        }
        for axis in AXES
    }
    for axis in AXES:
        intervals = [l for l in labels if l["axis"] == axis]
        starts = defaultdict(list)
        bounds = set()
        for index, interval in enumerate(intervals):
            starts[interval["start"]].append(index)
            bounds.update((interval["start"], interval["end"]))
        bounds = sorted(bounds)
        active = []
        surviving = {}
        # A sweep selects the latest annotation on every (start, end] segment.
        for start, end in zip(bounds, bounds[1:]):
            for index in starts[start]:
                heappush(active, (-index, intervals[index]["end"]))
            while active and active[0][1] <= start:
                heappop(active)
            if active:
                surviving[-active[0][0]] = end
        days = defaultdict(set)
        bands = defaultdict(Counter)
        for index, end in surviving.items():
            key = intervals[index]["class_id"]
            if key in ("unknown", "moving"):
                continue
            profile = profiles[axis][key]
            profile["confirmed_count"] += 1
            days[key].add(end.date())
            bands[key][period(end)] += 1
            last = profile["last_confirmed_at"]
            profile["last_confirmed_at"] = max(last, end) if last else end
        for key, profile in profiles[axis].items():
            profile["confirmed_days"] = len(days[key])
            count = profile["confirmed_count"]
            if count >= 3:
                band, votes = bands[key].most_common(1)[0]
                if votes * 5 >= count * 3:
                    profile["dominant_band"] = band
    return profiles


def node_descriptions(axis, nodes, profiles):
    """Shared question/Web labels; identities are used only for deterministic ordering."""
    noun = "拠点" if axis == "place" else "作業"
    band_names = dict(morning="午前", afternoon="午後", evening="夕方", night="夜")
    descriptions = {}

    def hour_text(at):
        at = dt(at)
        return f"{at.month}月{at.day}日の{at.hour}時"

    def precise_text(at):
        at = dt(at)
        return f"{at.year}年{at.month}月{at.day}日{at.hour}時{at.minute}分"

    for key, node in nodes.items():
        profile = profiles.get(key, {})
        last = profile.get("last_confirmed_at")
        if node.get("name"):
            label, source = node["name"], "name"
        elif last:
            label, source = f"前回{hour_text(last)}に確認した{noun}", "last_confirmed"
            if profile.get("dominant_band"):
                label += f"（{band_names[profile['dominant_band']]}の回答で多い）"
        else:
            anchor = node.get("anchor")
            label = (
                f"{hour_text(anchor)}ごろの記録に似た{noun}候補（未確認）"
                if anchor
                else f"未確認の{noun}候補"
            )
            source = "anchor"
        descriptions[key] = dict(label=label, source=source)

    # Reserve the existing special option names as well as the ordinary labels.
    reserved = {"わからない", "新しい拠点" if axis == "place" else "新しいタスク"}
    if axis == "place":
        reserved.add("移動中")
    counts = Counter(d["label"] for d in descriptions.values())
    for key, description in descriptions.items():
        if counts[description["label"]] < 2 and description["label"] not in reserved:
            continue
        last = profiles.get(key, {}).get("last_confirmed_at")
        anchor = nodes[key].get("anchor")
        detail = (
            "最終確認：" + precise_text(last)
            if last
            else "本人確認なし・出典：" + precise_text(anchor)
            if anchor
            else "本人確認なし"
        )
        description["label"] += f"（{detail}）"

    groups = defaultdict(list)
    for key, description in descriptions.items():
        groups[description["label"]].append(key)
    used = set(groups) | reserved

    def recency(key):
        last = profiles.get(key, {}).get("last_confirmed_at")
        anchor = nodes[key].get("anchor")
        return (
            last is None,
            -dt(last).timestamp() if last else 0,
            -dt(anchor).timestamp() if anchor else 0,
            key,
        )

    for label, keys in sorted(groups.items()):
        if len(keys) < 2 and label not in reserved:
            continue
        number = 1
        for key in sorted(keys, key=recency):
            unique = f"{label}（同じ説明の候補{number}）"
            while unique in used:
                number += 1
                unique = f"{label}（同じ説明の候補{number}）"
            descriptions[key]["label"] = unique
            used.add(unique)
            number += 1
    return descriptions


def question_presentation(rows, question_id, at):
    """Look up server-owned presentation metadata without changing annotation adoption."""
    return next(
        (
            r["payload"]
            for r in reversed(available_rows(rows, dt(at)))
            if r["kind"] == "question_presentation" and r["id"] == question_id
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
