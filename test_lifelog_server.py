from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app, stamp, current_activity, JST
from switchbot_contact_ble import parse_contact


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path / "log.sqlite3")
    app = create_app({"ble_enabled": False, "allowed_clients": ["testclient"], "stale_seconds": 120}, store)
    with TestClient(app) as client:
        yield client, store


def session(at, identifier="session-1", participant="p001", event_type="key"):
    return dict(session_id=identifier, participant_id=participant, device_id="iphone", started_at=stamp(at-timedelta(seconds=30)), ended_at=stamp(at), close_reason="keyboard_hidden",
                events=[{"time": stamp(at-timedelta(seconds=1)), "type": event_type, "data": {"text": "test"}}], typing={"total_typed_characters": 4})


def test_upload_retry_is_idempotent_and_detail_isolated(setup):
    client, _ = setup
    payload = session(datetime.now(timezone.utc))
    assert client.post("/api/sessions", json=payload).json()["inserted"] is True
    assert client.post("/api/sessions", json=payload).json()["inserted"] is False
    data = client.get("/api/dashboard/p001").json()
    assert data["summary"]["session_count"] == 1
    assert data["summary"]["typed_characters"] == 4
    assert data["current_activity"]["source"] == "推定"
    assert client.get("/api/records/p001/session/session-1").json()["events"] == payload["events"]
    assert client.get("/api/records/p002/session/session-1").status_code == 404
    assert client.get("/api/dashboard/p002").json()["timeline"] == []


def test_japan_day_boundary_and_pagination_snapshot(setup):
    client, _ = setup
    midnight = datetime(2026, 1, 2, tzinfo=JST)
    for i, delta in enumerate([-1, 0, 1, 2]):
        assert client.post("/api/sessions", json=session(midnight+timedelta(seconds=delta), str(i))).status_code == 200
    assert client.get("/api/dashboard/p001?day=2026-01-01").json()["summary"]["session_count"] == 1
    page = client.get("/api/dashboard/p001?day=2026-01-02&limit=2").json()
    assert page["summary"]["session_count"] == 3
    assert page["next_offset"] == 2
    client.post("/api/sessions", json=session(midnight+timedelta(seconds=3), "new"))
    following = client.get(f'/api/dashboard/p001?day=2026-01-02&limit=2&offset=2&snapshot_seq={page["snapshot_seq"]}').json()
    assert len(following["timeline"]) == 1
    assert following["next_offset"] is None
    assert len({x["id"] for x in page["timeline"] + following["timeline"]}) == 3


def test_contact_freshness_changes_and_name_survives_new_advertisements(setup):
    client, store = setup
    t = datetime.now(timezone.utc)
    state = dict(state="close", is_open=False, battery=80)
    store.contact("p001", "door", "SwitchBot", state, t-timedelta(seconds=180), -60)
    first = client.get("/api/dashboard/p001").json()
    assert first["devices"][0]["stale"] is True
    assert first["summary"]["door_changes"] == 0  # Initial observation is not an opening.
    store.contact("p001", "door", "SwitchBot", state, t, -55)
    second = client.get("/api/dashboard/p001").json()
    assert second["devices"][0]["stale"] is False
    assert second["total_events"] == 1  # heartbeat is not another transition
    assert client.put("/api/devices/door/name", json={"participant_id": "p001", "name": "玄関"}).status_code == 200
    store.contact("p001", "door", "SwitchBot", dict(state="open", is_open=True, battery=79), t, -60)
    third = client.get("/api/dashboard/p001").json()
    assert third["devices"][0]["name"] == "玄関"
    assert third["summary"]["door_changes"] == 1
    assert third["current_activity"]["source"] == "情報不足"
    assert client.put("/api/devices/door/name", json={"participant_id": "p002", "name": "wrong"}).status_code == 404


def test_activity_freshness_and_meta_is_not_typing(setup):
    client, store = setup
    t = datetime.now(timezone.utc)
    client.post("/api/sessions", json=session(t, event_type="meta"))
    assert client.get("/api/dashboard/p001").json()["current_activity"]["source"] == "情報不足"
    body = dict(event_id="activity-1", participant_id="p001", occurred_at=stamp(t), label="研究中")
    assert client.post("/api/activities", json=body).status_code == 200
    client.post("/api/activities", json=body)
    data = client.get("/api/dashboard/p001").json()
    assert data["current_activity"]["label"] == "研究中"
    assert data["summary"]["answer_count"] == 1
    assert current_activity(store.rows("p001"), t+timedelta(minutes=31))["source"] == "情報不足"


def test_legacy_narrative_batch_and_context_retry(setup):
    client, _ = setup
    t = stamp(datetime.now(timezone.utc))
    body = dict(event_id="n1", participant_id="p001", detected_at=t, answered_at=t, question={"kind": "activityCheck"}, answer={"label": "study"})
    assert client.post("/api/narrative-events", json=body).json()["inserted"] == 1
    assert client.post("/api/narrative-events", json=[body, dict(body, event_id="n2")]).json()["inserted"] == 1
    context = dict(event_id="c1", participant_id="p001", detected_at=t, event_type="outing")
    assert client.post("/api/context-events", json=context).json()["inserted"] is True
    assert client.post("/api/context-events", json=context).json()["inserted"] is False
    assert client.get("/api/dashboard/p001").json()["current_activity"]["label"] == "勉強・研究"
    assert client.get("/api/questions/pending/p001").status_code == 204


def test_validation_and_peer_allowlist(setup):
    client, _ = setup
    t = datetime.now(timezone.utc)
    for payload in [session(t+timedelta(days=1)), dict(session(t), ended_at="2026-01-01T00:00:00"), dict(session(t), events=[{}]), dict(session(t), participant_id="../bad")]:
        assert client.post("/api/sessions", json=payload).status_code == 422
    assert client.get("/api/dashboard/p001?day=not-a-date").status_code == 422
    assert client.get("/api/dashboard/p001?offset=-1").status_code == 422
    assert client.post("/api/sessions", content=b"x" * (8*1024*1024 + 1)).status_code == 413
    with TestClient(create_app({"ble_enabled": False, "allowed_clients": ["100.124.206.1"]}, Store(":memory:"))) as forbidden:
        assert forbidden.get("/health", headers={"X-Forwarded-For": "100.124.206.1"}).status_code == 403


def test_sqlite_persists_across_restart(tmp_path):
    path = tmp_path / "persist.sqlite3"
    store = Store(path)
    t = datetime.now(timezone.utc)
    store.save("session", "id", "p001", t, session(t))
    store.close()
    reopened = Store(path)
    assert len(reopened.rows("p001")) == 1
    reopened.close()


def test_contact_parser_rejects_unknown_encrypted_and_invalid_hall():
    packet = bytearray([ord("d"), 0, 85, 2, 0, 0, 0, 0, 0])
    assert parse_contact(packet) == {"state": "open", "is_open": True, "battery": 85}
    packet[3] = 4
    assert parse_contact(packet)["state"] == "timeOutNotClose"
    packet[3] = 6
    assert parse_contact(packet) is None
    packet[0] |= 0x80
    assert parse_contact(packet) is None
    assert parse_contact(b"short") is None
