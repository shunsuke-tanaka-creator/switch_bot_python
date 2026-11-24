"""Bounded, content-keyed research computations over read-only record snapshots."""

from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
from datetime import timedelta
from heapq import merge
import hashlib
import json
import threading

from .data import dt


def row_order(row):
    return dt(row["at"]), row["seq"], row["participant"], row["id"]


def content_fingerprint(rows):
    """Fallback for callers supplying fixture rows rather than cached views."""
    return hashlib.sha256(json.dumps(
        [(r.get("seq"), r.get("participant"), r["id"], r["kind"], r["at"], r["payload"])
         for r in rows], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class RecordView:
    def __init__(self, rows, fingerprint):
        self.rows = rows
        self.fingerprint = fingerprint
        self.boundaries = None

    def time_key(self, at):
        """Replay changes at reception boundaries and strictly after expiry."""
        boundaries = self.boundaries
        if boundaries is None:
            available = sorted(
                max(dt(r["at"]), dt(r["payload"].get("received_at", r["at"])))
                for r in self.rows
            )
            expires = sorted(
                dt(r["at"]) + timedelta(seconds=180)
                for r in self.rows if r["kind"] == "context_inference"
            )
            # Publish both arrays together for concurrent endpoint readers.
            boundaries = self.boundaries = available, expires
        available, expires = boundaries
        return (bisect_left(available, dt(at)), bisect_right(available, dt(at)),
                bisect_left(expires, dt(at)))


class _Source:
    def __init__(self, raw, stamp, maximum, previous=None):
        self.stamp, self.maximum = stamp, maximum
        self.by_kind = dict(previous.by_kind) if previous else {}
        self.digests = {k: v.copy() for k, v in previous.digests.items()} if previous else {}
        added = defaultdict(list)
        for row in raw:
            kind = row["kind"]
            # Mirror training's content identity, including sequence/order metadata.
            digest = self.digests.setdefault(kind, hashlib.sha256())
            digest.update(json.dumps(
                [row["seq"], row["participant"], kind, row["id"], row["at"], row["payload"]],
                ensure_ascii=False, separators=(",", ":"),
            ).encode())
            added[kind].append(dict(row, payload=json.loads(row["payload"])))
        for kind, rows in added.items():
            rows.sort(key=row_order, reverse=True)
            self.by_kind[kind] = list(merge(
                self.by_kind.get(kind, ()), rows, key=row_order, reverse=True,
            ))
        # A heartbeat append keeps every heartbeat-independent view intact.
        self.views = {
            kinds: view for kinds, view in previous.views.items()
            if not set(added).intersection(kinds)
        } if previous else {}

    def view(self, kinds=None):
        kinds = tuple(sorted(self.by_kind if kinds is None else kinds))
        if kinds not in self.views:
            fingerprints = [(k, self.digests[k].hexdigest()) for k in kinds if k in self.digests]
            fingerprint = hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
            rows = list(merge(
                *(self.by_kind.get(k, ()) for k in kinds), key=row_order, reverse=True,
            ))
            self.views[kinds] = RecordView(rows, fingerprint)
        return self.views[kinds]


class ResearchCache:
    """Share parsing and expensive artifacts without caching wall-clock responses.

    Store writes are append-only. Verify that fast path against SQLite counters;
    updates, deletes, registry writes and external connections take a content
    rescan. No database schema or persisted record changes are required.
    """

    def __init__(self, store, limit=16):
        self.store = store
        self.limit = limit
        self.sources = OrderedDict()
        self.computations = OrderedDict()
        self.lock = threading.RLock()
        self.enabled = True

    def clear(self):
        with self.lock:
            self.sources.clear()
            self.computations.clear()

    def _source(self, participant):
        if not hasattr(self.store, "db"):
            # Portable/in-memory adapters have no SQLite mutation counters.
            # Read their current content rather than assuming append-only writes.
            raw = [dict(r, participant=r.get("participant", participant), seq=r.get("seq", 0),
                        payload=json.dumps(r["payload"], sort_keys=True))
                   for r in self.store.rows(participant)]
            raw.sort(key=lambda r: r["seq"])
            return _Source(raw, None, 0)
        with self.lock, self.store.lock:
            stamp = (
                self.store.db.execute("PRAGMA data_version").fetchone()[0],
                self.store.db.total_changes,
            )
            previous = self.sources.get(participant) if self.enabled else None
            if previous is None or previous.stamp != stamp:
                maximum = self.store.db.execute("SELECT COALESCE(MAX(seq),0) FROM records").fetchone()[0]
                incremental = False
                if previous and previous.stamp[0] == stamp[0]:
                    appended = self.store.db.execute(
                        "SELECT COUNT(*) FROM records WHERE seq>?", (previous.maximum,),
                    ).fetchone()[0]
                    incremental = stamp[1] - previous.stamp[1] == appended
                # With both predicates SQLite otherwise chooses records_day
                # and scans the participant's entire history on every append.
                table = "records NOT INDEXED" if incremental else "records"
                query = f"SELECT * FROM {table} WHERE participant=?"
                params = [participant]
                if incremental:
                    query += " AND seq>?"
                    params.append(previous.maximum)
                raw = self.store.db.execute(query + " ORDER BY seq", params).fetchall()
                if incremental and not raw:
                    previous.stamp, previous.maximum = stamp, maximum
                else:
                    previous = _Source(raw, stamp, maximum, previous if incremental else None)
                if self.enabled:
                    self.sources[participant] = previous
            if self.enabled:
                self.sources.move_to_end(participant)
                while len(self.sources) > self.limit:
                    self.sources.popitem(last=False)
            return previous

    def records(self, participant, kinds=None):
        return self.record_views(participant, (kinds,))[0]

    # 追加: 履歴とキー用のviewを同じsnapshotからまとめて取得する。
    def record_views(self, participant, kind_sets):
        with self.lock:
            source = self._source(participant)
            return tuple(source.view(kinds) for kinds in kind_sets)

    def newest(self, participant):
        """Group ordering may depend on a heartbeat even if graph data does not."""
        with self.lock:
            source = self._source(participant)
            return max((row_order(rows[0]) for rows in source.by_kind.values() if rows), default=None)

    def compute(self, slot, fingerprint, build):
        with self.lock:
            cached = self.computations.get(slot) if self.enabled else None
            if cached is not None and cached[0] == fingerprint:
                self.computations.move_to_end(slot)
                return cached[1]
            result = build()
            if self.enabled:
                self.computations[slot] = (fingerprint, result)
                self.computations.move_to_end(slot)
                while len(self.computations) > self.limit:
                    self.computations.popitem(last=False)
            return result
