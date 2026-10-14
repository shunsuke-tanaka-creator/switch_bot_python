#!/usr/bin/env python3
"""MacBook lifelog API and SwitchBot BLE collector. Run with python lifelog_server.py."""
import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent
JST = ZoneInfo("Asia/Tokyo")
DEFAULTS = {
    "host": "100.106.185.3", "port": 8000, "participant_id": "p001",
    "allowed_clients": ["100.124.206.1", "100.106.185.3", "127.0.0.1", "::1"],
    "database": "data/lifelog.sqlite3", "ble_enabled": True,
    "stale_seconds": 120, "devices": {},
}
ACTIVITIES = dict(zip(
    "desk_work meeting study cooking cleaning laundry shopping meal bath toilet dressing relax nap sleep_prep video game reading sns walk commute errand family friends call".split(),
    "デスクワーク 会議・打合せ 勉強・研究 料理 掃除 洗濯 買い物 食事 入浴 トイレ 身支度 くつろぎ 仮眠 就寝準備 テレビ・動画 ゲーム 読書 スマホ・SNS 散歩 通勤・通学 用事 家族と 友人と 通話".split()))
EVENT_LABELS = {"wake_up": "起床", "outing": "外出（記録）", "return_home": "帰宅（記録）", "sleep": "就寝（記録）", "door_open": "ドアが開いた", "door_close": "ドアが閉じた"}


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def parsed(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    participant_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class TimedEvent(Event):
    @field_validator("*", mode="after")
    @classmethod
    def valid_time(cls, value):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("timezone required")
            if value > now() + timedelta(minutes=5):
                raise ValueError("timestamp is in the future")
        return value


class TypingValues(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_typed_characters: int | None = Field(None, ge=0)


class QuestionFields(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    prompt: str = "本人の回答"


class AnswerFields(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: str | None = None
    free_text: str | None = None


class Session(TimedEvent):
    session_id: str = Field(min_length=1, max_length=200)
    device_id: str
    started_at: datetime
    ended_at: datetime
    close_reason: str
    events: list[dict] = Field(max_length=50000)
    typing: TypingValues
    context: dict | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.ended_at < self.started_at:
            raise ValueError("ended_at precedes started_at")
        for event in self.events:
            try:
                t = parsed(event["time"])
            except (KeyError, TypeError, AttributeError, ValueError) as exc:
                raise ValueError("invalid session event time") from exc
            if t.tzinfo is None or not self.started_at - timedelta(seconds=1) <= t <= self.ended_at + timedelta(seconds=1):
                raise ValueError("event time outside session")
            if not isinstance(event.get("type"), str) or not isinstance(event.get("data"), dict):
                raise ValueError("invalid session event")
        return self


class Narrative(TimedEvent):
    event_id: str = Field(min_length=1, max_length=200)
    detected_at: datetime
    answered_at: datetime | None = None
    question: QuestionFields
    answer: AnswerFields


class Context(TimedEvent):
    event_id: str = Field(min_length=1, max_length=200)
    detected_at: datetime
    event_type: str


class Sensor(TimedEvent):
    device_id: str
    type: str
    value: str | None = None
    occurred_at: datetime


class DeviceName(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=100)


class Activity(TimedEvent):
    event_id: str = Field(min_length=1, max_length=200)
    occurred_at: datetime
    label: str = Field(min_length=1, max_length=500)


def activity_label(payload):
    answer = payload.get("answer", {})
    label = answer.get("free_text") or answer.get("label") or "回答なし"
    return ACTIVITIES.get(label, label)


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


def timeline_item(row):
    p, kind = row["payload"], row["kind"]
    detail = ""
    if kind == "session":
        title = "キーボード入力"
        detail = f'{len(p["events"])}イベント・{round((parsed(p["ended_at"])-parsed(p["started_at"])).total_seconds())}秒'
    elif kind == "narrative":
        title = activity_label(p)
        detail = p.get("question", {}).get("prompt", "本人の回答")
    elif kind == "switchbot":
        state = {"close": "閉じている", "open": "開いている", "timeOutNotClose": "開いたまま"}.get(p["state"], "不明")
        title = f'{p["name"]}：{state}'
        detail = "初回観測" if p.get("initial_observation") else "状態変化を受信"
    else:
        title = EVENT_LABELS.get(p.get("event_type", p.get("type")), p.get("event_type", p.get("type", "記録")))
        detail = p.get("value") or ""
    return dict(id=f'{kind}:{row["id"]}', record_id=row["id"], kind=kind, occurred_at=row["at"], title=title, detail=detail)


def current_activity(rows, at):
    candidates = []
    for row in rows:
        p = row["payload"]
        if row["kind"] == "narrative" and p.get("question", {}).get("kind") == "activityCheck":
            t = parsed(row["at"])
            if 0 <= (at-t).total_seconds() <= 1800:
                candidates.append((t, activity_label(p), "本人の回答", "30分以内の活動回答です。継続しているとは限りません。"))
        elif row["kind"] == "session":
            times = [parsed(e["time"]) for e in p["events"] if e.get("type") in {"key", "kana", "commit", "delete", "space", "return"}]
            if times:
                t = max(times)
                if 0 <= (at-t).total_seconds() <= 120:
                    candidates.append((t, "スマホで入力していた可能性", "推定", "2分以内の打鍵記録。使っているアプリや目的は判定できません。"))
    if not candidates:
        return dict(label="現在の活動は不明", source="情報不足", evidence="最近の活動回答・打鍵記録がありません。ドアの開閉だけでは在宅・外出を断定できません。", observed_at=None)
    t, label, source, evidence = max(candidates, key=lambda x: x[0])
    return dict(label=label, source=source, evidence=evidence, observed_at=stamp(t))


async def collect_ble(store, config, status):
    from bleak import BleakScanner
    from switchbot_contact_ble import SERVICE_UUIDS, parse_contact
    last_saved = {}

    def callback(device, advertisement):
        configured = {k.lower(): v for k, v in config["devices"].items()}
        if configured and device.address.lower() not in configured:
            return
        for uuid in SERVICE_UUIDS:
            state = parse_contact(advertisement.service_data.get(uuid, b""))
            if state is None:
                continue
            t = now()
            previous = last_saved.get(device.address)
            if previous and previous[1] == state and (t - previous[0]).total_seconds() < 10:
                return
            spec = configured.get(device.address.lower(), {})
            try:
                store.contact(spec.get("participant_id", config["participant_id"]), device.address,
                              spec.get("name", device.name or "SwitchBot 開閉センサー"), state, t, advertisement.rssi)
                last_saved[device.address] = (t, state)
                status.update(state="scanning", error=None, last_received_at=stamp(t))
            except Exception:
                logging.exception("Could not persist BLE observation")
                status.update(state="error", error="センサーデータの保存に失敗しました")
            return

    while True:
        try:
            status.update(state="starting", error=None)
            async with BleakScanner(callback, scanning_mode="active"):
                status.update(state="scanning", error=None)
                while True:
                    await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.warning("BLE scanner: %s", exc)
            status.update(state="error", error=str(exc))
            await asyncio.sleep(10)


def create_app(config=None, store=None):
    config = dict(DEFAULTS, **(config or {}))
    store = store or Store(ROOT / config["database"])
    status = dict(state="starting" if config["ble_enabled"] else "disabled", error=None, last_received_at=None)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(collect_ble(store, config, status)) if config["ble_enabled"] else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            store.close()

    app = FastAPI(title="MacBook Life Log", lifespan=lifespan)
    app.state.store, app.state.scanner = store, status

    @app.middleware("http")
    async def private_network(request: Request, call_next):
        # Use socket peer, never trust X-Forwarded-For (uvicorn proxy_headers=False).
        if not request.client or request.client.host not in config["allowed_clients"]:
            return JSONResponse({"detail": "This device is not allowed"}, status_code=403)
        # Bound JSON before parsing, including chunked transfer bodies.
        if request.method in {"POST", "PUT"}:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 8 * 1024 * 1024:
                    return JSONResponse({"detail": "Payload too large"}, status_code=413)
            request._body = bytes(data)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return dict(status="ok", server_time=stamp(now()), scanner=status)

    @app.post("/api/sessions")
    def sessions(payload: Session):
        return dict(inserted=store.save("session", payload.session_id, payload.participant_id, payload.ended_at, payload.model_dump(mode="json")))

    @app.post("/api/narrative-events")
    def narratives(payload: Narrative | list[Narrative]):
        items = payload if isinstance(payload, list) else [payload]
        return dict(inserted=sum(store.save("narrative", p.event_id, p.participant_id, p.answered_at or p.detected_at, p.model_dump(mode="json")) for p in items))

    @app.post("/api/context-events")
    def contexts(payload: Context):
        return dict(inserted=store.save("context", payload.event_id, payload.participant_id, payload.detected_at, payload.model_dump(mode="json")))

    @app.post("/api/sensor-events")
    def sensors(payload: Sensor):
        # Compatibility with the old in-app simulator. Never impersonate BLE devices.
        p = payload.model_dump(mode="json")
        identifier = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        return dict(inserted=store.save("sensor", identifier, payload.participant_id, payload.occurred_at, p))

    @app.post("/api/activities")
    def activities(payload: Activity):
        p = dict(event_id=payload.event_id, participant_id=payload.participant_id, detected_at=stamp(payload.occurred_at), answered_at=stamp(payload.occurred_at),
                 question={"kind": "activityCheck", "prompt": "今、何をしていますか？"}, answer={"free_text": payload.label}, source="parent_app")
        return dict(inserted=store.save("narrative", payload.event_id, payload.participant_id, payload.occurred_at, p))

    @app.get("/api/questions/pending/{participant_id}", status_code=204)
    def pending(participant_id: str):
        # Keyboard continues using its existing local question policy.
        return Response(status_code=204)

    @app.get("/api/dashboard/{participant_id}")
    def dashboard(participant_id: str, day: date | None = None, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500), snapshot_seq: int | None = Query(None, ge=0)):
        at = now()
        day = day or at.astimezone(JST).date()
        start = datetime.combine(day, datetime.min.time(), JST)
        end = start + timedelta(days=1)
        with store.lock:
            snapshot = snapshot_seq if snapshot_seq is not None else store.db.execute("SELECT COALESCE(MAX(seq),0) FROM records").fetchone()[0]
        rows = store.rows(participant_id, start, end, snapshot)
        recent = store.rows(participant_id, at-timedelta(hours=24), at+timedelta(microseconds=1))
        sessions_ = [r["payload"] for r in rows if r["kind"] == "session"]
        narratives_ = [r for r in rows if r["kind"] == "narrative"]
        sensor_rows = [r for r in rows if r["kind"] == "switchbot"]
        changes = [r for r in sensor_rows if not r["payload"].get("initial_observation")]
        typed = sum(p.get("typing", {}).get("total_typed_characters") or 0 for p in sessions_)
        summaries = []
        for row in reversed(narratives_):
            if row["payload"].get("question", {}).get("kind") == "activityCheck":
                summaries.append(parsed(row["at"]).astimezone(JST).strftime("%H:%M") + " " + activity_label(row["payload"]))
        summary = f"入力 {len(sessions_)}セッション・{typed}文字、回答 {len(narratives_)}件、開閉の状態変化 {len(changes)}件。"
        if summaries:
            summary += " 活動記録：" + " → ".join(summaries[-12:])
        elif not rows:
            summary = "この日のログはまだありません。"
        else:
            summary += " 活動の回答はまだありません。"
        context = next((r["payload"].get("context") for r in recent if r["kind"] == "session" and r["payload"].get("context")), None)
        devices = store.devices(participant_id, config, at)
        names = {d["device_id"]: d["name"] for d in devices}
        for row in rows:
            if row["kind"] == "switchbot":
                row["payload"]["name"] = names.get(row["payload"]["device_id"], row["payload"]["name"])
        return dict(day=str(day), timezone="Asia/Tokyo", generated_at=stamp(at), scanner=status,
                    current_activity=current_activity(recent, at), devices=devices,
                    summary=dict(text=summary, session_count=len(sessions_), typed_characters=typed, answer_count=len(narratives_), door_changes=len(changes)),
                    device_context=context, timeline=[timeline_item(r) for r in rows[offset:offset+limit]],
                    total_events=len(rows), next_offset=offset+limit if offset+limit < len(rows) else None, snapshot_seq=snapshot)

    @app.put("/api/devices/{identifier}/name")
    def device_name(identifier: str, payload: DeviceName):
        with store.lock, store.db:
            row = store.db.execute("SELECT payload FROM devices WHERE participant=? AND id=?", (payload.participant_id, identifier)).fetchone()
            if row is None:
                raise HTTPException(404, "Device not yet observed")
            store.db.execute("INSERT OR REPLACE INTO device_names VALUES(?,?,?)", (payload.participant_id, identifier, payload.name))
            p = json.loads(row[0])
            p["name"] = payload.name
            store.db.execute("UPDATE devices SET payload=? WHERE participant=? AND id=?", (json.dumps(p), payload.participant_id, identifier))
        return {"status": "ok"}

    @app.get("/api/records/{participant_id}/{kind}/{identifier}")
    def record(participant_id: str, kind: str, identifier: str):
        with store.lock:
            row = store.db.execute("SELECT payload FROM records WHERE participant=? AND kind=? AND id=?", (participant_id, kind, identifier)).fetchone()
        if row is None:
            raise HTTPException(404, "Record not found")
        return json.loads(row[0])

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "server-config.json"))
    parser.add_argument("--no-ble", action="store_true", help="API only, for development")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = dict(DEFAULTS)
    if Path(args.config).exists():
        config.update(json.loads(Path(args.config).read_text()))
    if args.no_ble:
        config["ble_enabled"] = False
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port
    uvicorn.run(create_app(config), host=config["host"], port=config["port"], proxy_headers=False)


if __name__ == "__main__":
    main()
