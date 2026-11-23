"""Immutable synthetic generations and an atomically published active selection."""
from contextlib import contextmanager
from datetime import timezone
import json
from pathlib import Path
import re
import sqlite3

from .data import dt
from .io import atomic_json
from .tendency_profile import canonical_hash


def is_sample_participant(participant):
    return isinstance(participant, str) and participant.endswith("-sample")


def real_participants(store, sample_participant="p001-sample"):
    with store.lock:
        return [r[0] for r in store.db.execute("SELECT DISTINCT participant FROM records")
                if r[0] != sample_participant and not is_sample_participant(r[0])]


def record_hash(rows):
    # Store normalizes envelope timestamps to UTC; seq is local DB metadata.
    records = [dict(kind=r["kind"], id=r["id"], participant=r["participant"],
                    at=dt(r["at"]).astimezone(timezone.utc).isoformat(timespec="milliseconds"),
                    payload=r["payload"]) for r in rows]
    return canonical_hash(sorted(records, key=lambda r: (r["at"], r["kind"], r["id"])))


def _schema(store):
    store.db.execute("""CREATE TABLE IF NOT EXISTS sample_generations (
        sample_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, manifest TEXT NOT NULL,
        report TEXT NOT NULL, opportunities TEXT NOT NULL)""")
    store.db.execute("""CREATE TABLE IF NOT EXISTS sample_active (
        source_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, manifest_hash TEXT NOT NULL)""")


@contextmanager
def registry_lock(store):
    # RLock handles threads; BEGIN IMMEDIATE serializes other processes/connections.
    with store.lock:
        store.db.execute("BEGIN IMMEDIATE")
        try:
            _schema(store)
            yield
            store.db.commit()
        except BaseException:
            store.db.rollback()
            raise


def generation(store, sample_id):
    with store.lock:
        try:
            row = store.db.execute("SELECT * FROM sample_generations WHERE sample_id=?", (sample_id,)).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return None
    if row is None:
        return None
    return dict(sample_participant=row["sample_id"], source_participant=row["source_id"],
                manifest=json.loads(row["manifest"]), report=json.loads(row["report"]),
                opportunities=json.loads(row["opportunities"]))


def active_generation(store, source, fallback=None):
    with store.lock:
        try:
            row = store.db.execute("SELECT sample_id FROM sample_active WHERE source_id=?", (source,)).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return fallback
    return row[0] if row else fallback


def active_manifest(store, source):
    sample_id = active_generation(store, source)
    info = generation(store, sample_id) if sample_id else None
    return info["manifest"] if info else None


def _retain_legacy(store, source):
    legacy = source + "-sample"
    if generation(store, legacy) or not store.rows(legacy):
        return
    manifest = dict(schema_version="legacy-sample-generation-v1", source_participant=source,
                    sample_participant=legacy, legacy=True, record_hash=record_hash(store.rows(legacy)))
    store.db.execute("INSERT INTO sample_generations VALUES(?,?,?,?,?)",
                     (legacy, source, json.dumps(manifest), "{}", "[]"))


def register_generation(store, rows, manifest, report, opportunities, manifest_path=None):
    sample_id, source = manifest["sample_participant"], manifest["source_participant"]
    if not re.fullmatch(re.escape(source) + r"-g[0-9a-f]{20}-sample", sample_id):
        raise ValueError("personalized generation requires a reserved generation ID")
    if any(r["participant"] != sample_id or r["payload"].get("participant_id", sample_id) != sample_id for r in rows):
        raise ValueError("synthetic rows must belong exclusively to their generation")
    if record_hash(rows) != manifest["record_hash"]:
        raise ValueError("generation record hash mismatch")
    with registry_lock(store):
        _retain_legacy(store, source)
        old = generation(store, sample_id)
        existing = store.rows(sample_id)
        if old:
            if old["manifest"] != manifest or record_hash(existing) != manifest["record_hash"]:
                raise RuntimeError("immutable generation differs or its records were modified")
            return dict(status="generated", inserted=0, existing=len(existing), **manifest)
        if existing:
            raise RuntimeError("unregistered generation already has records")
        for r in rows:
            store.db.execute("INSERT INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                             (sample_id, r["kind"], r["id"],
                              dt(r["at"]).astimezone(timezone.utc).isoformat(timespec="milliseconds"),
                              json.dumps(r["payload"], ensure_ascii=False)))
        store.db.execute("INSERT INTO sample_generations VALUES(?,?,?,?,?)",
                         (sample_id, source, json.dumps(manifest, ensure_ascii=False),
                          json.dumps(report, ensure_ascii=False), json.dumps(opportunities, ensure_ascii=False)))
        if manifest_path:
            path = Path(manifest_path)
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise RuntimeError("refusing to replace an immutable manifest")
            atomic_json(path.with_name("validation.json"), report)
            atomic_json(path.with_name("opportunities.json"), opportunities)
            atomic_json(path, manifest)
    return dict(status="generated", inserted=len(rows), existing=0, **manifest)


def activate_sample(store, source, sample_id):
    """Idempotent publication of generation ID and its exact manifest under one lock."""
    if not is_sample_participant(sample_id) or sample_id == source:
        raise ValueError("activation requires a reserved sample participant")
    with registry_lock(store):
        _retain_legacy(store, source)
        info = generation(store, sample_id)
        if not info or info["source_participant"] != source:
            raise ValueError("generation is not registered for this participant")
        manifest = info["manifest"]
        if record_hash(store.rows(sample_id)) != manifest["record_hash"]:
            raise RuntimeError("generation integrity check failed")
        digest = canonical_hash(manifest)
        store.db.execute("INSERT INTO sample_active VALUES(?,?,?) ON CONFLICT(source_id) DO UPDATE SET "
                         "sample_id=excluded.sample_id,manifest_hash=excluded.manifest_hash",
                         (source, sample_id, digest))
    return dict(status="active", source_participant=source, sample_participant=sample_id, manifest_hash=digest)
