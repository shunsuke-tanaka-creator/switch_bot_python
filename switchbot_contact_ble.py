#!/usr/bin/env python3
"""SwitchBot WoContact: official unencrypted scan-response format.

Requires: pip install bleak
Examples: python3 switchbot_contact_ble.py
          python3 switchbot_contact_ble.py --address UUID_OR_MAC --seconds 30
No Hub, cloud token, pairing or GATT connection is used.
This is a diagnostic sample; hardware/firmware compatibility is unverified.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime

from bleak import BleakScanner

SERVICE_UUIDS = (
    "0000fd3d-0000-1000-8000-00805f9b34fb",
    "00000d00-0000-1000-8000-00805f9b34fb",
)


def parse_contact(data):
    if len(data) < 9 or data[0] & 0x80:
        return None
    if chr(data[0] & 0x7F) not in ("d", "D"):
        return None
    hall = (data[3] >> 1) & 3
    if hall not in (0, 1, 2):
        return None
    return {
        "state": ("close", "open", "timeOutNotClose")[hall],
        "is_open": hall != 0,
        "battery": data[2] & 0x7F,
    }


async def scan(args):
    results = {}

    def callback(device, advertisement):
        if args.address and device.address.lower() != args.address.lower():
            return
        for service_uuid in SERVICE_UUIDS:
            data = advertisement.service_data.get(service_uuid, b"")
            state = parse_contact(data)
            if state is None:
                continue
            record = {
                "address": device.address,
                **state,
                "received_at": datetime.now().astimezone().isoformat(),
            }
            previous = results.get(device.address)
            results[device.address] = record
            if previous is None or previous["state"] != state["state"]:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            break

    print(f"BLEスキャン中（{args.seconds:g}秒）...", file=sys.stderr)
    async with BleakScanner(callback, scanning_mode="active"):
        await asyncio.sleep(args.seconds)
    if not results:
        print("状態不明：対応する開閉データを受信できませんでした。", file=sys.stderr)
        return 2
    print("最終受信結果：", file=sys.stderr)
    for record in results.values():
        print(json.dumps(record, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Windows/Linux: MAC; macOS: BLE UUID")
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds は正の値にしてください")
    try:
        return asyncio.run(scan(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"BLEエラー: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
