"""Bundled Rockpload integration.

Rockpload (https://github.com/LEX0RE/rockpload) signs in to your Epic account,
fetches replays from Rocket League's match history and uploads them to websites.
We ship its Windows build in tools/rockpload/ and, before launching it, make sure
its storage settings contain an enabled upload target pointing at this dashboard
(2v2 only). Only our own entry is added or updated; other settings are untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools" / "rockpload"
ENTRY_NAME = "RL xG dashboard"
RANKED_DOUBLES = 11


def settings_file() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return base / "rockpload" / "settings" / "storage.json"


def find_exe() -> Path | None:
    if not TOOLS_DIR.exists():
        return None
    exes = sorted(TOOLS_DIR.glob("rockpload*.exe"))
    return exes[0] if exes else None


def is_running() -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return any(line.lower().startswith('"rockpload') for line in out.splitlines())


def our_entry(port: int) -> dict:
    return {
        "name": ENTRY_NAME,
        "help_text": "Local xG dashboard (rocket_league_xg). Uploads ranked 2v2 replays for xG analysis.",
        "help_url": "",
        "upload_style": 1,                 # Multipart Form POST
        "replay_path": "/upload",
        "template_file": "{YEAR}-{MONTH}-{DAY}.{HOUR}.{MIN} {PLAYER} {MODE} {WINLOSS}",
        "url": f"http://localhost:{port}/api",
        "enabled": True,
        "uri_params": {},
        "token_style": 0,                  # no token
        "ping_style": 0, "ping_path": "/", "ping_probe_id": "",
        "live_style": 0, "live_path": "",
        "open_replay_style": 0, "open_replay_origin": "", "open_replay_path": "",
        "rename_style": 0, "rename_path": "",
        "playlist_filter_style": 1,        # whitelist
        "filtered_playlists": [{"id": RANKED_DOUBLES, "name": "Ranked Doubles"}],
    }


def ensure_config(port: int) -> str:
    """Add/refresh our upload target in Rockpload's storage.json. Returns a status message."""
    path = settings_file()
    entries: list = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8") or "[]") or []
        except json.JSONDecodeError:
            return f"left {path} alone: it isn't valid JSON"
        if not isinstance(entries, list):
            return f"left {path} alone: unexpected format"
    want = our_entry(port)
    existing = next((e for e in entries if isinstance(e, dict) and e.get("name") == ENTRY_NAME), None)
    if existing is not None:
        # keep the user's tweaks (e.g. playlist filter) but make sure it points at us and is on
        changed = existing.get("url") != want["url"] or existing.get("replay_path") != "/upload" \
            or not existing.get("enabled") or existing.get("upload_style") != 1
        if not changed:
            return "upload target already configured"
        existing.update(url=want["url"], replay_path="/upload", enabled=True, upload_style=1)
    else:
        entries.append(want)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return "upload target added" if existing is None else "upload target updated"


def ensure_upload_on_launch() -> None:
    """Turn on Rockpload's 'Upload Replays on launch' so starting the dashboard catches up on
    games played since last time. Only touches that one flag, and only if the file exists."""
    path = settings_file().with_name("behavior.json")
    if not path.exists():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(cfg, dict) and cfg.get("upload_on_launch") is False:
        cfg["upload_on_launch"] = True
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def launch(port: int) -> str:
    """Configure and start Rockpload if it's bundled and not already running."""
    exe = find_exe()
    if exe is None:
        return f"not bundled (put the Windows build in {TOOLS_DIR})"
    if is_running():
        # Rockpload rewrites its settings on exit, so only edit them while it's closed.
        return "already running (if uploads don't arrive, check it has the 'RL xG dashboard' website enabled)"
    msg = ensure_config(port)
    ensure_upload_on_launch()
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(exe)], cwd=exe.parent, creationflags=flags, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"started ({msg})"


def status() -> dict:
    exe = find_exe()
    return {"bundled": exe is not None, "running": is_running() if exe else False,
            "settings": str(settings_file())}
