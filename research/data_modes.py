"""Data-mode selection and source identities shared by training and displays."""
from pathlib import Path
import hashlib
import json

from .data import dt


DATA_MODES = ("real", "sample", "combined")
DEFAULT_SAMPLE_PARTICIPANT = "p001-sample"
TRAINING_KINDS = {
    "session",
    "heartbeat",
    "sensor",
    "switchbot",
    "switchbot_observation",
    "correction",
}
TRAINING_QUESTIONS = {"currentState", "nightReflection", "activityCheck"}


def data_mode(value):
    if value not in DATA_MODES:
        raise ValueError("data_mode must be real, sample, or combined")
    return value


def participants_for_mode(participant, mode, sample_participant=DEFAULT_SAMPLE_PARTICIPANT):
    mode = data_mode(mode)
    if mode == "real":
        return (participant,)
    if mode == "sample":
        return (sample_participant,)
    return tuple(dict.fromkeys((participant, sample_participant)))


def resolved_sample_participant(store, participant, fallback=DEFAULT_SAMPLE_PARTICIPANT):
    from .sample_registry import active_generation
    return active_generation(store, participant, fallback)


def rows_for_mode(store, participant, mode="real", sample_participant=DEFAULT_SAMPLE_PARTICIPANT):
    if mode != "real":
        sample_participant = resolved_sample_participant(store, participant, sample_participant)
    rows = []
    for source in participants_for_mode(participant, mode, sample_participant):
        rows.extend(store.rows(source))
    return sorted(
        rows,
        key=lambda r: (dt(r["at"]), r.get("seq", 0), r["participant"], r["id"]),
        reverse=True,
    )


def is_training_row(row):
    if row["kind"] in TRAINING_KINDS:
        return True
    if row["kind"] == "narrative":
        return row["payload"].get("question", {}).get("kind") in TRAINING_QUESTIONS
    return False


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def source_identity(
    store,
    participant,
    mode="real",
    sample_participant=DEFAULT_SAMPLE_PARTICIPANT,
    sample_manifest=None,
    resolve_active=True,
):
    mode = data_mode(mode)
    if mode != "real" and resolve_active:
        sample_participant = resolved_sample_participant(store, participant, sample_participant)
    sources = participants_for_mode(participant, mode, sample_participant)
    maximum = {}
    content = {}
    for source in sources:
        rows = [r for r in store.rows(source) if is_training_row(r)]
        content[source] = hashlib.sha256(json.dumps([(r['id'],r['kind'],r['at'],r['payload']) for r in rows],sort_keys=True).encode()).hexdigest()
        maximum[source] = max((int(r.get("seq", 0)) for r in rows), default=0)
    manifest_hash = file_hash(sample_manifest) if sample_participant in sources and sample_manifest else None
    if mode != "real":
        from .sample_registry import generation
        from .tendency_profile import canonical_hash
        info = generation(store, sample_participant)
        if info and not info["manifest"].get("legacy"):
            manifest_hash = canonical_hash(info["manifest"])
    value = {
        "data_mode": mode,
        "source_max_seq": maximum,
        "source_content_hash": content,
        "sample_manifest_hash": manifest_hash,
    }
    value["fingerprint"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def mode_snapshot(store, participant, mode="real", sample_participant=DEFAULT_SAMPLE_PARTICIPANT,
                  sample_manifest=None):
    """Pin one immutable selection for training rows and checkpoint provenance."""
    mode = data_mode(mode)
    with store.lock:
        selected = (resolved_sample_participant(store, participant, sample_participant)
                    if mode != "real" else sample_participant)
        rows = [r for source in participants_for_mode(participant, mode, selected) for r in store.rows(source)]
        rows.sort(key=lambda r: (dt(r["at"]), r.get("seq", 0), r["participant"], r["id"]), reverse=True)
        identity = source_identity(store, participant, mode, selected, sample_manifest, resolve_active=False)
        return rows, identity
