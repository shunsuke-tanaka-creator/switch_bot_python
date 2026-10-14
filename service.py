#!/usr/bin/env python3
"""Install/start/status/stop/remove the current user's MacBook lifelog LaunchAgent."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess

ROOT = Path(__file__).resolve().parent
LABEL = "jp.kubotalab.lifelog"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
DOMAIN = f"gui/{os.getuid()}"
TARGET = f"{DOMAIN}/{LABEL}"


def run(*args, check=True):
    return subprocess.run(["launchctl", *args], check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "start", "status", "stop", "remove"])
    args = parser.parse_args()
    if args.command == "install":
        if not (ROOT / ".venv/bin/python").exists():
            parser.error("先に python3 -m venv .venv と .venv/bin/pip install -r requirements.txt を実行してください")
        if PLIST.exists():
            existing = plistlib.loads(PLIST.read_bytes())
            if existing.get("WorkingDirectory") != str(ROOT):
                parser.error("別のディレクトリのサービスが存在するため、上書きしません")
            run("bootout", TARGET, check=False)
        (ROOT / "data").mkdir(exist_ok=True)
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_bytes(plistlib.dumps({
            "Label": LABEL,
            "ProgramArguments": ["/usr/bin/caffeinate", "-i", "-s", str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "lifelog_server.py")],
            "WorkingDirectory": str(ROOT), "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 15,
            "StandardOutPath": str(ROOT / "data/server.stdout.log"),
            "StandardErrorPath": str(ROOT / "data/server.stderr.log"),
        }))
        run("bootstrap", DOMAIN, str(PLIST))
        print(f"Installed: {PLIST}")
    elif args.command == "start":
        result = subprocess.run(["launchctl", "print", TARGET], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            run("bootstrap", DOMAIN, str(PLIST))
        else:
            run("kickstart", TARGET)
    elif args.command == "status":
        run("print", TARGET)
    elif args.command == "stop":
        run("bootout", TARGET)
    else:
        run("bootout", TARGET, check=False)
        PLIST.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
