"""Replay parsing: turn a .replay file into tidy numpy arrays.

Uses `subtr-actor` (Rust/boxcars under the hood), which handles modern replay
versions. Everything downstream works on the `Replay` object defined here, so
the parser can be swapped without touching shot detection or the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import subtr_actor

BOOST_SCALE = 100.0 / 255.0


def _pid(remote_id: dict) -> str:
    """Stable string key for a player's remote id (Epic/Steam/PS/Xbox/...)."""
    (platform, value), = remote_id.items()
    if isinstance(value, dict):  # PlayStation / Switch carry a dict
        value = value.get("online_id") or value.get("name") or json.dumps(value, sort_keys=True)
    return f"{platform}:{value}"


@dataclass
class Player:
    pid: str
    name: str
    team: int                 # 0 = blue, 1 = orange
    pos: np.ndarray           # (F, 3) NaN when not replicated
    vel: np.ndarray           # (F, 3)
    quat: np.ndarray          # (F, 4) x, y, z, w
    boost: np.ndarray         # (F,) 0-100
    jump: np.ndarray          # (F,) bool
    double_jump: np.ndarray   # (F,) bool
    dodge: np.ndarray         # (F,) bool


@dataclass
class Goal:
    frame: int
    time: float
    team: int                 # scoring team
    scorer_pid: str | None


@dataclass
class StatEvent:
    frame: int
    time: float
    kind: str                 # 'Shot' | 'Save' | 'Assist'
    team: int
    pid: str | None


@dataclass
class Replay:
    path: str
    replay_id: str
    date: str | None
    playlist_id: int | None
    match_type: str | None
    team_size: int
    time: np.ndarray          # (F,) seconds since replay start
    seconds_remaining: np.ndarray  # (F,)
    ball_pos: np.ndarray      # (F, 3)
    ball_vel: np.ndarray      # (F, 3)
    players: list[Player]
    goals: list[Goal]
    stat_events: list[StatEvent]
    final_score: tuple[int, int] = (0, 0)
    scoreboard: dict = field(default_factory=dict)  # pid -> in-game stats

    @property
    def n_frames(self) -> int:
        return len(self.time)

    def player(self, pid: str) -> Player:
        return next(p for p in self.players if p.pid == pid)


def _vec(d: dict | None) -> tuple[float, float, float]:
    if not d:
        return (np.nan, np.nan, np.nan)
    return (d["x"], d["y"], d["z"])


def load_replay(path: str | Path) -> Replay:
    path = str(path)
    fd = subtr_actor.get_replay_frames_data(path)
    meta = fd["meta"]
    headers = dict((k, v) for k, v in meta.get("all_headers", []))
    mf = fd["frame_data"]["metadata_frames"]
    n = len(mf)

    t = np.array([m["time"] for m in mf], dtype=np.float64)
    secs = np.array([m["seconds_remaining"] for m in mf], dtype=np.float64)

    ball_pos = np.full((n, 3), np.nan)
    ball_vel = np.full((n, 3), np.nan)
    for i, f in enumerate(fd["frame_data"]["ball_data"]["frames"]):
        d = f.get("Data") if isinstance(f, dict) else None
        if d and d.get("rigid_body"):
            rb = d["rigid_body"]
            ball_pos[i] = _vec(rb.get("location"))
            ball_vel[i] = _vec(rb.get("linear_velocity"))

    # Team/name lookup from header metadata
    name_of, team_of = {}, {}
    for team_idx, key in ((0, "team_zero"), (1, "team_one")):
        for p in meta.get(key, []):
            pid = _pid(p["remote_id"])
            name_of[pid] = p.get("name", pid)
            team_of[pid] = team_idx

    players = []
    for remote_id, pdata in fd["frame_data"]["players"]:
        pid = _pid(remote_id)
        pos = np.full((n, 3), np.nan)
        vel = np.full((n, 3), np.nan)
        quat = np.full((n, 4), np.nan)
        boost = np.full(n, np.nan)
        jump = np.zeros(n, bool)
        djump = np.zeros(n, bool)
        dodge = np.zeros(n, bool)
        team_seen = None
        for i, f in enumerate(pdata["frames"]):
            d = f.get("Data") if isinstance(f, dict) else None
            if not d:
                continue
            rb = d.get("rigid_body")
            if rb:
                pos[i] = _vec(rb.get("location"))
                vel[i] = _vec(rb.get("linear_velocity"))
                q = rb.get("rotation") or {}
                quat[i] = (q.get("x", np.nan), q.get("y", np.nan), q.get("z", np.nan), q.get("w", np.nan))
            ba = d.get("boost_amount")
            boost[i] = ba * BOOST_SCALE if ba is not None else np.nan
            jump[i] = bool(d.get("jump_active"))
            djump[i] = bool(d.get("double_jump_active"))
            dodge[i] = bool(d.get("dodge_active"))
            if team_seen is None and d.get("is_team_0") is not None:
                team_seen = 0 if d["is_team_0"] else 1
            if pid not in name_of and d.get("player_name"):
                name_of[pid] = d["player_name"]
        team = team_of.get(pid, team_seen if team_seen is not None else 0)
        players.append(Player(pid, name_of.get(pid, pid), team, pos, vel, quat,
                              boost, jump, djump, dodge))

    goals = []
    for g in fd.get("goal_events", []):
        goals.append(Goal(
            frame=g["frame"], time=g["time"],
            team=0 if g["scoring_team_is_team_0"] else 1,
            scorer_pid=_pid(g["player"]) if g.get("player") else None,
        ))

    stat_events = []
    for e in fd.get("player_stat_events", []):
        stat_events.append(StatEvent(
            frame=e["frame"], time=e["time"], kind=e["kind"],
            team=0 if e["is_team_0"] else 1,
            pid=_pid(e["player"]) if e.get("player") else None,
        ))

    scoreboard = {}
    for key in ("team_zero", "team_one"):
        for p in meta.get(key, []):
            scoreboard[_pid(p["remote_id"])] = p.get("stats") or {}

    team_size = int(headers.get("TeamSize") or max(
        1, max((sum(p.team == k for p in players) for k in (0, 1)), default=1)))

    return Replay(
        path=path,
        replay_id=str(headers.get("Id") or Path(path).stem),
        date=headers.get("Date"),
        playlist_id=(meta.get("game_type") or {}).get("playlist_id"),
        match_type=(meta.get("game_type") or {}).get("game_type"),
        team_size=team_size,
        time=t,
        seconds_remaining=secs,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        players=players,
        goals=goals,
        stat_events=stat_events,
        final_score=(int(headers.get("Team0Score", 0) or 0), int(headers.get("Team1Score", 0) or 0)),
        scoreboard=scoreboard,
    )
