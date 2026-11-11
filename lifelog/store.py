"""SQLite persistence shared by collection, replication and research."""
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4
from .time import stamp, parsed


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, participant TEXT NOT NULL,
                kind TEXT NOT NULL, id TEXT NOT NULL, at TEXT NOT NULL, payload TEXT NOT NULL,
                UNIQUE(participant, kind, id));
            CREATE INDEX IF NOT EXISTS records_day ON records(participant, at);
            CREATE TABLE IF NOT EXISTS device_names (
                participant TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL,
                PRIMARY KEY(participant, id));
            CREATE TABLE IF NOT EXISTS devices (
                participant TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(participant, id));
        """)
        self.db.commit()

    def save(self, kind, identifier, participant, at, payload):
        with self.lock, self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO records(participant,kind,id,at,payload) VALUES(?,?,?,?,?)",
                                  (participant, kind, identifier, stamp(at), json.dumps(payload, ensure_ascii=False)))
            return cur.rowcount == 1

    def contact(self, participant, identifier, name, state, received_at, rssi):
        with self.lock, self.db:
            row = self.db.execute("SELECT payload FROM devices WHERE participant=? AND id=?", (participant, identifier)).fetchone()
            previous = json.loads(row[0]) if row else None
            alias = self.db.execute("SELECT name FROM device_names WHERE participant=? AND id=?", (participant, identifier)).fetchone()
            payload = dict(state, device_id=identifier, name=alias[0] if alias else name, last_seen=stamp(received_at), rssi=rssi)
            self.db.execute("INSERT OR REPLACE INTO devices VALUES(?,?,?)", (participant, identifier, json.dumps(payload)))
            observation_id = identifier + ":" + received_at.strftime("%Y%m%d%H%M")
            self.save("switchbot_observation", observation_id, participant, received_at, payload)
            if previous is None or previous["state"] != state["state"]:
                payload["initial_observation"] = previous is None
                self.save("switchbot", str(uuid4()), participant, received_at, payload)

    def rows(self, participant, start=None, end=None, snapshot=None):
        where, params = ["participant=?"], [participant]
        for clause, value in [("at>=?", stamp(start) if start else None), ("at<?", stamp(end) if end else None), ("seq<=?", snapshot)]:
            if value is not None:
                where.append(clause)
                params.append(value)
        with self.lock:
            rows = self.db.execute("SELECT * FROM records WHERE " + " AND ".join(where) + " ORDER BY at DESC,seq DESC", params).fetchall()
        return [dict(row, payload=json.loads(row["payload"])) for row in rows]

    def devices(self, participant, config, at):
        with self.lock:
            rows = self.db.execute("SELECT payload FROM devices WHERE participant=?", (participant,)).fetchall()
        result = []
        for row in rows:
            p = json.loads(row[0])
            p["stale"] = (at - parsed(p["last_seen"])).total_seconds() > config["stale_seconds"]
            result.append(p)
        for identifier, spec in config["devices"].items():
            if spec.get("participant_id", config["participant_id"]) == participant and not any(p["device_id"].lower() == identifier.lower() for p in result):
                result.append(dict(device_id=identifier, name=spec.get("name", identifier), state="unknown", battery=None, last_seen=None, rssi=None, stale=True))
        return sorted(result, key=lambda d: d["name"])

    def close(self):
        self.db.close()

