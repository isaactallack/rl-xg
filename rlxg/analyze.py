"""Replay -> xG statistics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import XGModel
from .parse import Replay, load_replay
from .outcome import shot_clip, shot_outcomes
from .shots import extract_shots
from .touches import detect_touches

TEAM_NAMES = {0: "blue", 1: "orange"}
SHOT_COLUMNS = ["time", "clock", "seconds_remaining", "team", "player", "pid", "xg", "xgot", "goal", "on_target",
                "dist_goal", "ball_speed", "ball_x", "ball_y", "ball_z", "cross_x", "cross_abs_x", "cross_z",
                "time_to_goal", "carry_time", "prev_by_self", "shooter_dodge",
                "shooter_airborne", "n_def_can_intercept", "def1_margin", "prev_by_teammate",
                "prev_by_opponent", "frame", "post_frame", "goal_frame",
                "end_x", "end_y", "end_z", "end_kind", "end_by", "end_frame"]


@dataclass
class MatchXG:
    replay: Replay
    shots: pd.DataFrame       # one row per detected shot with its xG
    players: pd.DataFrame     # per-player summary
    teams: pd.DataFrame       # per-team summary

    def to_dict(self) -> dict:
        r = self.replay
        return {
            "replay_id": r.replay_id, "path": r.path, "date": r.date,
            "team_size": r.team_size, "playlist_id": r.playlist_id,
            "score": {"blue": r.final_score[0], "orange": r.final_score[1]},
            "duration": float(r.time[-1] - r.time[0]) if r.n_frames else 0.0,
            "teams": self.teams.to_dict(orient="records"),
            "players": self.players.to_dict(orient="records"),
            "shots": self.shots[[c for c in SHOT_COLUMNS if c in self.shots]].to_dict(orient="records"),
            "goals": [
                {"time": g.time, "frame": g.frame, "team": TEAM_NAMES[g.team],
                 "clock": float(game_clock(r)[min(g.frame, r.n_frames - 1)]),
                 "seconds_remaining": float(r.seconds_remaining[min(g.frame, r.n_frames - 1)]),
                 "scorer": next((p.name for p in r.players if p.pid == g.scorer_pid), None),
                 "scorer_pid": g.scorer_pid}
                for g in r.goals
            ],
            "start_time": float(r.time[0]) if r.n_frames else 0.0,
            "game_length": float(game_clock(r).max()) if r.n_frames else 300.0,
        }


    def clips(self) -> list[dict]:
        """A short ball-and-cars clip around every shot, in the same order as `shots`."""
        return [shot_clip(self.replay, s) for s in self.shots.to_dict(orient="records")]


_MODEL_CACHE: dict[str, XGModel] = {}


def get_model(path: str | Path | None = None) -> XGModel:
    from .model import DEFAULT_MODEL_PATH
    key = str(path or DEFAULT_MODEL_PATH)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = XGModel.load(key)
    return _MODEL_CACHE[key]


def game_clock(rep: Replay) -> np.ndarray:
    """Elapsed match time in seconds (0-300 regulation, >300 in overtime).

    seconds_remaining counts down to 0, then counts *up* from 0 in overtime.
    """
    sr = rep.seconds_remaining
    hit_zero = np.flatnonzero(sr <= 0)
    clock = 300.0 - sr
    if len(hit_zero):
        clock[hit_zero[0]:] = 300.0 + sr[hit_zero[0]:]
    return clock


def analyze_replay(path: str | Path | Replay, model: XGModel | None = None,
                   band: str | None = None) -> MatchXG:
    """Parse a replay (or take an already-parsed one), detect shots and score them."""
    model = model or get_model()
    rep = path if isinstance(path, Replay) else load_replay(path)
    touches = detect_touches(rep)
    shots = extract_shots(rep, touches)
    if len(shots):
        shots = shots.join(shot_outcomes(rep, touches, shots))
        shots["clock"] = game_clock(rep)[shots["frame"].to_numpy()]
        shots["xg"] = model.predict_chance(shots, band=band)   # chance quality (pre-touch)
        shots["xgot"] = model.predict(shots, band=band)        # shot as struck (post-touch)
        shots["team"] = shots["team"].map(TEAM_NAMES)
        shots = shots.sort_values("time").reset_index(drop=True)
    else:
        shots = pd.DataFrame(columns=SHOT_COLUMNS + ["pid"])

    # Per-player table (everyone in the match, including players with no shots)
    rows = []
    for p in rep.players:
        s = shots[shots.pid == p.pid] if len(shots) else shots
        sb = rep.scoreboard.get(p.pid, {})
        xg = float(s.xg.sum()) if len(s) else 0.0
        xgot = float(s.xgot.sum()) if len(s) else 0.0
        goals_from_shots = int(s.goal.sum()) if len(s) else 0
        rows.append(dict(
            player=p.name, pid=p.pid, team=TEAM_NAMES[p.team],
            goals=int(sb.get("Goals", goals_from_shots)),
            shots=len(s), on_target=int(s.on_target.sum()) if len(s) else 0,
            xg=round(xg, 2), xgot=round(xgot, 2),
            xg_per_shot=round(xg / len(s), 3) if len(s) else 0.0,
            goals_minus_xg=round(int(sb.get("Goals", goals_from_shots)) - xg, 2),
            shooting=round(xgot - xg, 2),            # >0: struck chances better than average
            big_chances=int((s.xg >= 0.5).sum()) if len(s) else 0,
            big_chances_missed=int(((s.xg >= 0.5) & (s.goal == 0)).sum()) if len(s) else 0,
            scoreboard_shots=sb.get("Shots"), saves=sb.get("Saves"), assists=sb.get("Assists"),
        ))
    players = pd.DataFrame(rows).sort_values(["team", "xg"], ascending=[True, False]).reset_index(drop=True)

    teams = []
    for k, name in TEAM_NAMES.items():
        s = shots[shots.team == name] if len(shots) else shots
        opp = shots[shots.team == TEAM_NAMES[1 - k]] if len(shots) else shots
        xg, xga = (float(s.xg.sum()) if len(s) else 0.0), (float(opp.xg.sum()) if len(opp) else 0.0)
        teams.append(dict(
            team=name, goals=rep.final_score[k], xg=round(xg, 2),
            xgot=round(float(s.xgot.sum()) if len(s) else 0.0, 2), shots=len(s),
            on_target=int(s.on_target.sum()) if len(s) else 0,
            xg_per_shot=round(xg / len(s), 3) if len(s) else 0.0,
            xg_against=round(xga, 2), xg_diff=round(xg - xga, 2),
        ))
    return MatchXG(rep, shots, players, pd.DataFrame(teams))
