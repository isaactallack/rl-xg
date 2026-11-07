"""SQLite storage for analysed matches and seen replay files."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id          TEXT PRIMARY KEY,      -- replay header id
    sha1        TEXT,
    path        TEXT,
    source      TEXT,                  -- watch | upload | rockpload
    played_at   TEXT,                  -- replay date
    added_at    REAL,
    team_size   INTEGER,
    data        TEXT                   -- MatchXG.to_dict() as JSON
);
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    size        INTEGER,
    mtime       REAL,
    sha1        TEXT,
    status      TEXT,                  -- done | skipped | error | queued
    reason      TEXT,
    match_id    TEXT,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    started_at  REAL,
    ended_at    REAL                   -- NULL while active
);
CREATE TABLE IF NOT EXISTS clips (
    match_id    TEXT,
    idx         INTEGER,               -- shot index within the match's shots list
    data        BLOB,                  -- zlib-compressed JSON (rlxg.outcome.shot_clip)
    PRIMARY KEY (match_id, idx)
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


SAME_GAME_WINDOW = 30 * 60   # s: copies of one game are saved seconds apart; a rematch is minutes later


def game_key(data: dict) -> str:
    """Identifies a game independently of which replay file it came from.

    The Demos folder copy and the Rockpload upload of the same game have different replay ids and
    slightly different frames, so match on who played, the score and who scored (plus the play time,
    checked separately).
    """
    return json.dumps([
        sorted(p["pid"] for p in data.get("players", [])),
        data.get("score", {}).get("blue"), data.get("score", {}).get("orange"),
        sorted([g.get("team") or "", g.get("scorer_pid") or ""] for g in data.get("goals", [])),
    ])


def played_ts(date: str | None) -> float:
    """Replay header date ('2026-11-06 17-07-20', local time) -> epoch seconds; 0 if unknown."""
    import datetime as dt
    import re
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2})[-:](\d{2})[-:](\d{2})", date or "")
    if not m:
        return 0.0
    return dt.datetime(*map(int, m.groups())).timestamp()


def _same_time(a: str | None, b: str | None) -> bool:
    ta, tb = played_ts(a), played_ts(b)
    return bool(ta and tb) and abs(ta - tb) < SAME_GAME_WINDOW


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(matches)")}
            if "session_id" not in cols:
                self._db.execute("ALTER TABLE matches ADD COLUMN session_id INTEGER")
            if "model_version" not in cols:
                self._db.execute("ALTER TABLE matches ADD COLUMN model_version TEXT")
            if "game_key" not in cols:
                self._db.execute("ALTER TABLE matches ADD COLUMN game_key TEXT")
                for r in self._db.execute("SELECT id, data FROM matches").fetchall():
                    self._db.execute("UPDATE matches SET game_key=? WHERE id=?", (game_key(json.loads(r[1])), r[0]))
            self._db.commit()

    # ---- files --------------------------------------------------------
    def file_state(self, path: str):
        with self._lock:
            return self._db.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def set_file(self, path: str, size: int, mtime: float, sha1: str | None, status: str,
                 reason: str | None = None, match_id: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                (path, size, mtime, sha1, status, reason, match_id, time.time()))
            self._db.commit()

    def has_sha1(self, sha1: str) -> bool:
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM matches WHERE sha1=? UNION SELECT 1 FROM files WHERE sha1=? AND status IN ('done','skipped')",
                (sha1, sha1)).fetchone() is not None

    def queued_files(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT path FROM files WHERE status='queued' ORDER BY updated_at").fetchall()
        return [r["path"] for r in rows]

    def recent_files(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT path, status, reason, match_id, updated_at FROM files ORDER BY updated_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- matches ------------------------------------------------------
    def has_match(self, match_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM matches WHERE id=?", (match_id,)).fetchone() is not None

    def add_match(self, data: dict, sha1: str, path: str, source: str,
                  session_id: int | None, model_version: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO matches (id, sha1, path, source, played_at, added_at, team_size, data,"
                " session_id, model_version, game_key) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (data["replay_id"], sha1, path, source, data.get("date") or "", time.time(),
                 data.get("team_size"), json.dumps(data), session_id, model_version, game_key(data)))
            self._db.commit()

    def update_match_data(self, match_id: str, data: dict, model_version: str) -> None:
        """Replace analysis results (re-score) while keeping source/session/added_at."""
        with self._lock:
            self._db.execute("UPDATE matches SET data=?, model_version=?, game_key=? WHERE id=?",
                             (json.dumps(data), model_version, game_key(data), match_id))
            self._db.commit()

    def find_same_game(self, data: dict) -> str | None:
        """Id of an already stored copy of this game from a different replay file, if any."""
        with self._lock:
            rows = self._db.execute("SELECT id, played_at FROM matches WHERE game_key=? AND id!=?",
                                    (game_key(data), data.get("replay_id"))).fetchall()
        return next((r["id"] for r in rows if _same_time(r["played_at"], data.get("date"))), None)

    def merge_duplicate_games(self) -> int:
        """Keep one copy of each game stored more than once (the first one added); returns copies removed.

        The removed copy's files are pointed at the kept match, so re-scoring can use either file.
        """
        with self._lock:
            rows = self._db.execute("SELECT id, game_key, played_at, added_at, session_id FROM matches"
                                    " WHERE game_key IS NOT NULL ORDER BY added_at").fetchall()
            kept: dict[str, list] = {}
            removed = 0
            for r in rows:
                twin = next((k for k in kept.get(r["game_key"], []) if _same_time(k["played_at"], r["played_at"])), None)
                if twin is None:
                    kept.setdefault(r["game_key"], []).append(dict(r))
                    continue
                if twin["session_id"] is None and r["session_id"] is not None:
                    self._db.execute("UPDATE matches SET session_id=? WHERE id=?", (r["session_id"], twin["id"]))
                    twin["session_id"] = r["session_id"]
                self._db.execute("UPDATE files SET match_id=?, status='skipped', reason=? WHERE match_id=?",
                                 (twin["id"], f"same game as {twin['id']}", r["id"]))
                self._db.execute("DELETE FROM matches WHERE id=?", (r["id"],))
                self._db.execute("DELETE FROM clips WHERE match_id=?", (r["id"],))
                removed += 1
            self._db.commit()
        return removed

    def stale_matches(self, model_version: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, path, sha1 FROM matches WHERE model_version IS NULL OR model_version != ?"
                " ORDER BY added_at DESC", (model_version,)).fetchall()
        return [dict(r) for r in rows]

    def files_for_match(self, match_id: str) -> list[str]:
        """Every replay file that was analysed as this match (e.g. a Demos copy and a Rockpload upload)."""
        with self._lock:
            rows = self._db.execute("SELECT path FROM files WHERE match_id = ? ORDER BY updated_at DESC",
                                    (match_id,)).fetchall()
        return [r[0] for r in rows]

    def matches(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, source, played_at, added_at, session_id, model_version, data FROM matches"
                " ORDER BY played_at DESC, added_at DESC").fetchall()
        return [dict(r) | {"data": json.loads(r["data"])} for r in rows]

    # ---- sessions -----------------------------------------------------
    def active_session(self) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def start_session(self, name: str | None) -> dict:
        self.end_session()
        with self._lock:
            cur = self._db.execute("INSERT INTO sessions (name, started_at) VALUES (?, ?)",
                                   (name or time.strftime("Session %d %b %H:%M"), time.time()))
            self._db.commit()
            sid = cur.lastrowid
        return self.session(sid)

    def end_session(self) -> None:
        with self._lock:
            self._db.execute("UPDATE sessions SET ended_at=? WHERE ended_at IS NULL", (time.time(),))
            self._db.commit()

    def session(self, sid: int) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def sessions(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.*, COUNT(m.id) AS games FROM sessions s LEFT JOIN matches m ON m.session_id = s.id"
                " GROUP BY s.id ORDER BY s.id DESC").fetchall()
        return [dict(r) for r in rows]

    def rename_session(self, sid: int, name: str) -> None:
        with self._lock:
            self._db.execute("UPDATE sessions SET name=? WHERE id=?", (name, sid))
            self._db.commit()

    def delete_session(self, sid: int) -> None:
        """Forget the session; its games stay (untagged)."""
        with self._lock:
            self._db.execute("UPDATE matches SET session_id=NULL WHERE session_id=?", (sid,))
            self._db.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self._db.commit()

    def match(self, match_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
        return json.loads(r["data"]) | {"source": r["source"]} if r else None

    def delete_match(self, match_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM matches WHERE id=?", (match_id,))
            self._db.execute("DELETE FROM clips WHERE match_id=?", (match_id,))
            self._db.commit()

    # ---- shot clips (3D replay) ---------------------------------------
    def set_clips(self, match_id: str, clips: list[dict]) -> None:
        """Replace every clip of a match (shot indices change when a game is re-analysed)."""
        rows = [(match_id, i, zlib.compress(json.dumps(c, separators=(",", ":")).encode(), 6))
                for i, c in enumerate(clips)]
        with self._lock:
            self._db.execute("DELETE FROM clips WHERE match_id=?", (match_id,))
            self._db.executemany("INSERT INTO clips VALUES (?,?,?)", rows)
            self._db.commit()

    def clip(self, match_id: str, idx: int) -> bytes | None:
        """The clip as JSON bytes, or None."""
        with self._lock:
            r = self._db.execute("SELECT data FROM clips WHERE match_id=? AND idx=?", (match_id, idx)).fetchone()
        return zlib.decompress(r[0]) if r else None

    # ---- settings -----------------------------------------------------
    def get_setting(self, key: str, default=None):
        with self._lock:
            r = self._db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))
            self._db.commit()
