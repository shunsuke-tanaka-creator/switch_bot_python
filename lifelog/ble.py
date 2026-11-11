"""SwitchBot acquisition. BLE is imported only when collection starts."""
import asyncio
import logging
from .time import now, stamp


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

