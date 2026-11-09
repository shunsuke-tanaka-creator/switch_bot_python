"""Validated model+record transfer between Mac and Ubuntu, without OS-specific state."""

from contextlib import ExitStack
from pathlib import Path, PurePosixPath
import fcntl
import hashlib
import json
import os
import zipfile
from .service import export_page, ingest_page, identifier
from lifelog_server import Store
import tempfile
import shutil


def export_bundle(store, participant, models, path):
    participant = identifier(participant)
    root = Path(models) / participant
    with ExitStack() as stack:
        # The two trainers atomically replace checkpoints; hold their locks for matching metadata.
        if root.exists():
            for lockpath in sorted(root.rglob("training.lock")):
                lock = stack.enter_context(lockpath.open("r"))
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        records = export_page(store, participant, limit=2**31 - 1)
        files = {"records.json": json.dumps(records, ensure_ascii=False).encode()}
        if root.exists():
            for file in sorted(root.rglob("*")):
                if file.is_file() and file.name in (
                    "active.pt",
                    "status.json",
                    "comparison.json",
                    "personalization.json",
                ):
                    files["models/" + file.relative_to(root).as_posix()] = (
                        file.read_bytes()
                    )
        manifest = dict(
            format="behavior-transfer-v1",
            participant=participant,
            sha256={
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
            note="Text encoder weights are separate: copy identical weights and configure text_model on destination.",
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
            z.writestr("manifest.json", json.dumps(manifest))
        os.replace(tmp, path)
    return dict(
        status="exported",
        participant=participant,
        records=len(records["records"]),
        files=len(files),
        path=str(path),
    )


def import_bundle(store, participant, models, path):
    """Import into an empty destination only. Validate entire bundle before any DB write."""
    participant = identifier(participant)
    root = Path(models) / participant
    if store.rows(participant) or root.exists():
        raise ValueError(
            "Destination participant/models must be empty; use a new destination or back it up first"
        )
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate archive entries")
        manifest = json.loads(z.read("manifest.json"))
        if (
            manifest.get("format") != "behavior-transfer-v1"
            or manifest.get("participant") != participant
        ):
            raise ValueError("Bundle participant/format mismatch")
        names = manifest["sha256"]
        if set(z.namelist()) != set(names) | {"manifest.json"}:
            raise ValueError("Unexpected archive entries")
        files = {}
        for name, digest in names.items():
            rel = PurePosixPath(name)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or "\\" in name
                or (
                    name != "records.json"
                    and (not name.startswith("models/") or len(rel.parts) < 2)
                )
            ):
                raise ValueError("Unsafe archive path")
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Bundle integrity failure")
            files[name] = data
        records = json.loads(files["records.json"])
        if records.get("participant") != participant or any(
            r["participant"] != participant for r in records["records"]
        ):
            raise ValueError("Records participant mismatch")
        if any(
            r["participant"] != participant
            for key in ("devices", "device_names")
            for r in records.get(key, [])
        ):
            raise ValueError("Metadata participant mismatch")
    # Validate record format/timestamps/metadata with the same importer before touching disk.
    staged = Store(":memory:")
    try:
        count = ingest_page(staged, records)
        Path(models).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".context-import-", dir=models
        ) as temporary:
            staging_root = Path(temporary) / participant
            staging_root.mkdir()
            for name, data in files.items():
                if name.startswith("models/"):
                    target = staging_root / name[len("models/") :]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            installed = False
            try:
                with store.lock, store.db:
                    if store.rows(participant) or root.exists():
                        raise ValueError("Destination changed during import")
                    for row in staged.db.execute(
                        "SELECT participant,kind,id,at,payload FROM records ORDER BY seq"
                    ):
                        store.db.execute(
                            "INSERT INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                            tuple(row),
                        )
                    for table in ("devices", "device_names"):
                        for row in staged.db.execute("SELECT * FROM " + table):
                            store.db.execute(
                                "INSERT INTO " + table + " VALUES(?,?,?)", tuple(row)
                            )
                    os.rename(staging_root, root)
                    installed = True
            except Exception:
                if installed:
                    shutil.rmtree(root)
                raise
    finally:
        staged.close()
    return dict(
        status="imported", participant=participant, records=count, models=str(root)
    )
