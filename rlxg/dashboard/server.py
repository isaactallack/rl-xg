"""xG dashboard server.

    python -m rlxg.dashboard            # http://localhost:4200

Replays arrive three ways, all into one analysis queue:
  * folder watcher (default: Rocket League's Demos folder, plus any extra dirs)
  * POST /api/upload (multipart field "file") - what Rockpload's "Localhost"
    multipart storage sends; returns 201 / 409 like ballchasing
  * drag-and-drop in the web UI (same endpoint)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import rockpload
from .store import Store, played_ts

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_DEMOS = Path.home() / "Documents" / "My Games" / "Rocket League" / "TAGame" / "Demos"


class Hub:
    """Fan-out of server-sent events to connected browsers."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subs: set[asyncio.Queue] = set()

    def publish(self, event: str, data: dict) -> None:
        if not self.loop:
            return
        for q in list(self.subs):
            self.loop.call_soon_threadsafe(q.put_nowait, (event, data))


class Engine:
    def __init__(self, data_dir: Path, watch_dirs: list[Path], team_size: int, backfill_hours: float):
        self.data_dir = data_dir
        self.uploads = data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.store = Store(data_dir / "xg.sqlite")
        if (n := self.store.merge_duplicate_games()):
            print(f"merged {n} games that were stored twice (Demos copy + upload)")
        self.watch_dirs = [d for d in watch_dirs]
        self.team_size = team_size
        self.backfill_hours = backfill_hours
        self.skill: str | None = None
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self.pending: set[str] = set()
        self.current: str | None = None
        self.hub = Hub()
        self.started = time.time()

    # ---- intake -------------------------------------------------------
    def enqueue(self, path: Path, source: str, priority: int = 0, match_id: str | None = None) -> None:
        """priority 0 = new game, 1 = backfill of older replays, 2 = re-score with a newer model."""
        key = str(path)
        if key in self.pending:
            return
        self.pending.add(key)
        if source != "rescore":
            st = path.stat()
            self.store.set_file(key, st.st_size, st.st_mtime, None, "queued")
        self._seq += 1
        self.q.put((priority, self._seq, path, source, match_id))
        self.hub.publish("status", self.status())

    def scan(self, first: bool = False) -> None:
        # Only games saved after start-up are tracked, unless a backfill window was asked for.
        cutoff = self.started - (self.backfill_hours * 3600 if (first and self.backfill_hours) else 60)
        found = []
        for d in self.watch_dirs:
            if not d.exists():
                continue
            for p in d.glob("*.replay"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                if time.time() - st.st_mtime < 3:      # still being written
                    continue
                prev = self.store.file_state(str(p))
                if prev and prev["size"] == st.st_size and abs(prev["mtime"] - st.st_mtime) < 1 \
                        and prev["status"] != "queued":
                    continue
                if prev is None and st.st_mtime < cutoff:
                    continue
                found.append((st.st_mtime, p))
        for mtime, p in sorted(found, reverse=True):  # newest first
            self.enqueue(p, "watch", priority=1 if mtime < self.started else 0)

    def queue_rescore(self) -> int:
        """Re-analyse stored games that were scored by a different model version."""
        n = 0
        for m in self.store.stale_matches(model_version()):
            # the original file may be gone (e.g. RL prunes Demos); any other copy of the same match will do
            cands = [Path(m["path"]), self.uploads / f"{m['sha1']}.replay",
                     *map(Path, self.store.files_for_match(m["id"]))]
            for cand in cands:
                if cand.exists():
                    self.enqueue(cand, "rescore", priority=2, match_id=m["id"])
                    n += 1
                    break
            else:
                print(f"can't re-score {m['id']}: no replay file left for it")
        return n

    def resume_queued(self) -> int:
        """Re-queue files that were waiting when the dashboard last stopped."""
        n = 0
        for p in map(Path, self.store.queued_files()):
            if p.exists():
                self.enqueue(p, "upload" if p.parent == self.uploads else "watch")
                n += 1
        return n

    def watch_loop(self) -> None:
        if (n := self.resume_queued()):
            print(f"resuming {n} replays that were queued before the last shutdown")
        n = self.queue_rescore()
        if n:
            print(f"model changed: re-scoring {n} stored games in the background")
        self.scan(first=True)
        while True:
            time.sleep(4)
            try:
                self.scan()
            except Exception as e:  # never let the watcher die
                print("watcher error:", e, file=sys.stderr)

    # ---- analysis -----------------------------------------------------
    def work_loop(self) -> None:
        while True:
            _, _, path, source, match_id = self.q.get()
            self.current = str(path)
            self.hub.publish("status", self.status())
            try:
                if source == "rescore":
                    self._rescore(path, match_id)
                else:
                    self._process(path, source)
            except Exception as e:
                st = path.stat() if path.exists() else None
                self.store.set_file(str(path), st.st_size if st else 0, st.st_mtime if st else 0,
                                    None, "error", str(e)[:300])
                self.hub.publish("file", {"path": str(path), "status": "error", "reason": str(e)[:300]})
            finally:
                self.pending.discard(str(path))
                self.current = None
                self.hub.publish("status", self.status())

    def _process(self, path: Path, source: str) -> None:
        raw = path.read_bytes()
        sha1 = hashlib.sha1(raw).hexdigest()
        st = path.stat()
        if self.store.has_sha1(sha1):
            self.store.set_file(str(path), st.st_size, st.st_mtime, sha1, "skipped", "duplicate")
            return
        data = self._analyse(path, self.team_size)
        if "skipped" in data:
            self.store.set_file(str(path), st.st_size, st.st_mtime, sha1, "skipped", data["skipped"])
            self.hub.publish("file", {"path": str(path), "status": "skipped", "reason": data["skipped"]})
            return
        clips = data.pop("clips", [])
        # the same game can arrive twice: the Demos folder copy and the Rockpload upload have different ids
        twin = self.store.find_same_game(data)
        if twin:
            reason = f"same game as {twin}"
            self.store.set_file(str(path), st.st_size, st.st_mtime, sha1, "skipped", reason, twin)
            self.hub.publish("file", {"path": str(path), "status": "skipped", "reason": reason})
            return
        sess = self.store.active_session()
        # Only games *played* during the session join it (Rockpload may upload older history).
        in_session = sess is not None and played_ts(data.get("date")) >= sess["started_at"] - SESSION_GRACE
        self.store.add_match(data, sha1, str(path), source, sess["id"] if in_session else None, model_version())
        self.store.set_clips(data["replay_id"], clips)
        self.store.set_file(str(path), st.st_size, st.st_mtime, sha1, "done", None, data["replay_id"])
        self.hub.publish("match", {"id": data["replay_id"], "source": source})

    def _rescore(self, path: Path, match_id: str) -> None:
        data = self._analyse(path, team_size=0)
        clips = data.pop("clips", [])
        self.store.update_match_data(match_id, data, model_version())
        self.store.set_clips(match_id, clips)
        self.hub.publish("match", {"id": match_id, "rescored": True})

    def _analyse(self, path: Path, team_size: int) -> dict:
        r = subprocess.run([sys.executable, "-m", "rlxg.dashboard.worker", str(path), str(team_size),
                            self.skill or ""],
                           capture_output=True, text=True, timeout=180, cwd=ROOT)
        if r.returncode != 0:
            msg = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
            raise RuntimeError(msg)
        return json.loads(r.stdout)

    def status(self) -> dict:
        return {
            "queue": self.q.qsize() + (1 if self.current else 0),
            "current": Path(self.current).name if self.current else None,
            "watching": [str(d) for d in self.watch_dirs],
            "team_size": self.team_size,
            "model_version": model_version(),
            "session": self.store.active_session(),
        }


SESSION_GRACE = 10 * 60   # a game already under way when the session starts still counts


_MODEL_VERSION: tuple[float, str] | None = None
# Bump when the analysis itself changes (not just the model), so stored games are re-analysed.
# a2: shot end points (end_*) and per-shot 3D clips.
ANALYSIS_VERSION = "a2"


def model_version() -> str:
    """Short hash of the model file plus ANALYSIS_VERSION; changes on retraining or analysis changes."""
    global _MODEL_VERSION
    from rlxg.model import DEFAULT_MODEL_PATH
    mtime = DEFAULT_MODEL_PATH.stat().st_mtime
    if _MODEL_VERSION is None or _MODEL_VERSION[0] != mtime:
        _MODEL_VERSION = (mtime, hashlib.sha1(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()[:12])
    return f"{_MODEL_VERSION[1]}-{ANALYSIS_VERSION}"


def me_pid(engine: Engine, matches: list[dict]) -> str | None:
    pid = engine.store.get_setting("me")
    if pid:
        return pid
    c = Counter(p["pid"] for m in matches for p in m["data"]["players"])
    return c.most_common(1)[0][0] if c else None


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="RL xG dashboard")

    @app.on_event("startup")
    async def _start():
        engine.hub.loop = asyncio.get_running_loop()
        threading.Thread(target=engine.work_loop, daemon=True).start()
        threading.Thread(target=engine.watch_loop, daemon=True).start()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/ping")
    @app.get("/api")
    def ping():
        return {"ok": True}

    @app.post("/api/upload", status_code=201)
    async def upload(file: UploadFile = File(...)):
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "empty file")
        sha1 = hashlib.sha1(raw).hexdigest()
        dest = engine.uploads / f"{sha1}.replay"
        if engine.store.has_sha1(sha1) or str(dest) in engine.pending:
            return JSONResponse({"status": "duplicate", "sha1": sha1}, status_code=409)
        dest.write_bytes(raw)
        engine.enqueue(dest, "upload")
        return {"status": "queued", "id": sha1, "sha1": sha1}

    @app.get("/api/status")
    def status():
        return engine.status() | {"recent_files": engine.store.recent_files(15), "rockpload": rockpload.status()}

    @app.get("/api/matches")
    def matches():
        ms = engine.store.matches()
        me = me_pid(engine, ms)
        out = []
        for m in ms:
            d = m["data"]
            mine = next((p for p in d["players"] if p["pid"] == me), None)
            out.append({
                "id": m["id"], "date": d.get("date"), "source": m["source"], "added_at": m["added_at"],
                "session_id": m["session_id"],
                "score": d["score"], "xg": {t["team"]: t["xg"] for t in d["teams"]},
                "team_size": d.get("team_size"),
                "me": mine and {k: mine[k] for k in ("player", "team", "goals", "xg", "xgot", "shots") if k in mine},
            })
        return {"me": me, "matches": out}

    @app.get("/api/matches/{match_id}")
    def match(match_id: str):
        m = engine.store.match(match_id)
        if not m:
            raise HTTPException(404)
        return m

    @app.get("/api/matches/{match_id}/clips/{idx}")
    def clip(match_id: str, idx: int):
        raw = engine.store.clip(match_id, idx)
        if raw is None:
            raise HTTPException(404, "no clip for this shot (re-analysing older games adds them)")
        return Response(raw, media_type="application/json")

    @app.get("/api/misses")
    def misses(limit: int = 100, who: str = "me", session: int | None = None):
        """The biggest chances that weren't scored, across every stored game, by chance xG."""
        ms = engine.store.matches()
        me = me_pid(engine, ms)
        out = []
        for m in ms:
            if session is not None and m["session_id"] != session:
                continue
            d = m["data"]
            mine = next((p for p in d["players"] if p["pid"] == me), None)
            for i, s in enumerate(d["shots"]):
                if s.get("goal") or (who == "me" and s.get("pid") != me):
                    continue
                out.append({k: s.get(k) for k in ("player", "pid", "team", "xg", "xgot", "clock", "end_kind", "end_by",
                                                  "ball_x", "ball_y", "end_x", "end_y", "cross_x", "cross_z",
                                                  "on_target", "dist_goal", "ball_speed")}
                           | {"match_id": m["id"], "idx": i, "date": d.get("date"), "score": d["score"],
                              "my_team": mine and mine["team"]})
        out.sort(key=lambda r: -(r["xg"] or 0))
        return {"me": me, "misses": out[:limit]}

    @app.delete("/api/matches/{match_id}")
    def delete(match_id: str):
        engine.store.delete_match(match_id)
        engine.hub.publish("match", {"id": match_id, "deleted": True})
        return {"ok": True}

    @app.get("/api/players")
    def players():
        ms = engine.store.matches()
        c, names = Counter(), {}
        for m in ms:
            for p in m["data"]["players"]:
                c[p["pid"]] += 1
                names[p["pid"]] = p["player"]
        return {"me": me_pid(engine, ms),
                "players": [{"pid": k, "name": names[k], "games": n} for k, n in c.most_common(30)]}

    @app.get("/api/sessions")
    def sessions():
        return {"active": engine.store.active_session(), "sessions": engine.store.sessions()}

    @app.post("/api/sessions/start")
    async def session_start(req: Request):
        raw = await req.body()
        body = json.loads(raw) if raw else {}
        sess = engine.store.start_session(body.get("name"))
        engine.hub.publish("status", engine.status())
        return sess

    @app.post("/api/sessions/stop")
    def session_stop():
        engine.store.end_session()
        engine.hub.publish("status", engine.status())
        return {"ok": True}

    @app.put("/api/sessions/{sid}")
    async def session_rename(sid: int, req: Request):
        engine.store.rename_session(sid, (await req.json())["name"])
        return {"ok": True}

    @app.delete("/api/sessions/{sid}")
    def session_delete(sid: int):
        engine.store.delete_session(sid)
        engine.hub.publish("status", engine.status())
        return {"ok": True}

    @app.post("/api/rescore")
    def rescore():
        return {"queued": engine.queue_rescore()}

    @app.put("/api/settings")
    async def settings(req: Request):
        body = await req.json()
        if "me" in body:
            engine.store.set_setting("me", body["me"] or None)
        return {"ok": True}

    @app.get("/api/events")
    async def events(request: Request):
        q: asyncio.Queue = asyncio.Queue()
        engine.hub.subs.add(q)

        async def gen():
            try:
                yield f"event: status\ndata: {json.dumps(engine.status())}\n\n"
                while not await request.is_disconnected():
                    try:
                        ev, data = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {ev}\ndata: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                engine.hub.subs.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main(argv=None) -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(prog="python -m rlxg.dashboard")
    ap.add_argument("--port", type=int, default=4200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--watch", action="append", default=None,
                    help="folder to watch for new .replay files (repeatable). Default: RL Demos folder")
    ap.add_argument("--no-default-watch", action="store_true", help="don't watch the RL Demos folder")
    ap.add_argument("--team-size", type=int, default=2, help="only analyse NvN games (0 = any)")
    ap.add_argument("--skill", default=None, metavar="BAND",
                    help="rank band xG is calibrated to: c3-gc2 (default) or ssl")
    ap.add_argument("--backfill-hours", type=float, default=0,
                    help="also analyse replays saved up to N hours before start-up (default 0: only new games)")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "dashboard"))
    ap.add_argument("--no-rockpload", action="store_true",
                    help="don't start the bundled Rockpload uploader")
    args = ap.parse_args(argv)

    dirs = [] if args.no_default_watch else [DEFAULT_DEMOS]
    dirs += [Path(d) for d in (args.watch or [])]
    engine = Engine(Path(args.data_dir), dirs, args.team_size, args.backfill_hours)
    engine.skill = args.skill
    print(f"RL xG dashboard -> http://{args.host}:{args.port}")
    print("watching:", *[str(d) for d in dirs] or ["(nothing)"], sep="\n  ")
    if args.no_rockpload:
        print("rockpload: not started (--no-rockpload)")
    else:
        print("rockpload:", rockpload.launch(args.port))
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
