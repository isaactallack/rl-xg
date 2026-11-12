"""Extra shot context: goal-mouth coverage, pressure on the shooter, recent movement, game state.

All in the attacking frame (target goal at +y), like the rest of the features.

* Goal-mouth coverage (`cov_*`): the kinematic intercept (defence.py) against straight
  lines from the ball to a 7 x 3 grid of points across the goal mouth, not only the
  nearest one. "How much of the goal is open": a defender covering the near post
  leaves the far post open, which the single aim point can't see. Done for the ball's
  own speed (at least 1500 uu/s, as the chance features assume) and for a hard shot.
* Pressure (`press*`): how long each defender needs to get to the ball where it is now.
* Recent movement (`*_h25`, `*_h50`): shooter, ball, defenders and teammate 0.25 s and
  0.5 s before the touch, so the model sees who is turning, accelerating or retreating.
* Game state: score difference for the shooting team, seconds left, overtime.
"""
from __future__ import annotations

import numpy as np

from . import defence
from . import physics as ph

COV_SPEED_FAST = 2800.0
_xs = np.linspace(-ph.GOAL_HALF_W + 1.5 * ph.BALL_RADIUS, ph.GOAL_HALF_W - 1.5 * ph.BALL_RADIUS, 7)
_zs = np.array([ph.BALL_RADIUS, ph.GOAL_H / 2, ph.GOAL_H - 1.5 * ph.BALL_RADIUS])
AIMS = np.array([[x, ph.ARENA_Y, z] for x in _xs for z in _zs])
GOAL_CENTRE = np.array([0.0, ph.ARENA_Y, ph.GOAL_H / 2])
OPEN_EASY = 0.3     # s: a spot is "easily open" if the best defender is this late

COVERAGE_KEYS = [f"cov{s}_{k}" for s in ("", "_fast") for k in ("open", "open_easy", "best", "mean")]
PRESSURE_KEYS = ["press1_t", "press1_closing", "press1_facing", "press2_t", "press_n_close"]
HISTORY_KEYS = (
    [f"sh_{k}_h{h}" for h in (25, 50) for k in ("speed", "vy", "z")] + ["sh_boost_used_h50", "sh_accel_h25"]
    + [f"pre_ball_{k}_h50" for k in ("speed", "z")]
    + [f"{d}_{k}" for d in ("gk", "defb") for k in ("dd_goal_h50", "dd_ball_h50", "speed_h50")]
    + ["defb_d_goal", "defb_d_ball", "mate_dd_goal_h50", "mate_dd_ball_h50"]
)
GAME_KEYS = ["score_diff", "seconds_remaining", "overtime"]


def goal_coverage(defenders: list[dict], ball: np.ndarray, ball_speed: float) -> dict:
    out = {}
    fr = np.linspace(0.0, 1.0, defence.PATH_POINTS)
    pts = ball + fr[None, :, None] * (AIMS - ball)[:, None, :]          # (aims, points, 3): ball -> each aim
    pts_flat = pts.reshape(-1, 3)
    need = [defence.need_time(**d, path=pts_flat).reshape(len(AIMS), -1) for d in defenders]
    dist = np.linalg.norm(AIMS - ball, axis=1)
    for tag, v in (("", max(ball_speed, 1500.0)), ("_fast", COV_SPEED_FAST)):
        t = fr[None, :] * (dist / v)[:, None]                            # arrival time at each point
        # per aim point: the best defender's margin at their best point on that line
        best = np.min([(n - t).min(axis=1) for n in need], axis=0) if need else np.full(len(AIMS), defence.T_CAP)
        out.update({f"cov{tag}_open": float((best >= 0).mean()), f"cov{tag}_open_easy": float((best >= OPEN_EASY).mean()),
                    f"cov{tag}_best": float(best.max()), f"cov{tag}_mean": float(best.mean())})
    return out


def pressure(defenders: list[dict], ball: np.ndarray) -> dict:
    """Time each defender needs to reach the ball where it is (a single-point path at t=0)."""
    r = sorted((defence.intercept(**d, path=ball[None], t=np.zeros(1)) for d in defenders), key=lambda x: x["margin"])
    get = lambda i, k, dflt: r[i][k] if len(r) > i else dflt
    return dict(press1_t=get(0, "margin", defence.T_CAP), press1_closing=get(0, "closing", 0.0),
                press1_facing=get(0, "facing", 0.0), press2_t=get(1, "margin", defence.T_CAP),
                press_n_close=float(sum(x["margin"] < 0.5 for x in r)))


def _state(p, g: int, flip: np.ndarray):
    pos = p.pos[g] * flip
    return (pos, p.vel[g] * flip, p.boost[g]) if np.isfinite(pos).all() else (None, None, np.nan)


def history(rep, f: int, flip: np.ndarray, shooter, team: int) -> dict:
    """Movement over the last half second; NaN where a player isn't replicated."""
    nan = float("nan")
    out = dict.fromkeys(HISTORY_KEYS, nan)
    t0 = rep.time[f]
    g = {h: int(np.clip(np.searchsorted(rep.time, t0 - h / 100.0), 0, f)) for h in (25, 50)}
    ball = {k: rep.ball_pos[k] * flip for k in (f, g[50])}

    pos0, vel0, b0 = _state(shooter, f, flip)
    for h in (25, 50):
        pos, vel, b = _state(shooter, g[h], flip)
        if pos is not None:
            out.update({f"sh_speed_h{h}": float(np.linalg.norm(vel)), f"sh_vy_h{h}": float(vel[1]), f"sh_z_h{h}": float(pos[2])})
            if h == 50 and np.isfinite(b) and np.isfinite(b0):
                out["sh_boost_used_h50"] = float(b - b0)
            if h == 25 and vel0 is not None:
                out["sh_accel_h25"] = float((np.linalg.norm(vel0) - np.linalg.norm(vel)) / max(t0 - rep.time[g[25]], 1e-3))
    pv = rep.ball_vel[g[50]] * flip
    if np.isfinite(pv).all():
        out["pre_ball_speed_h50"] = float(np.linalg.norm(pv))
        out["pre_ball_z_h50"] = float(ball[g[50]][2])

    def moves(p):
        now, then = _state(p, f, flip)[0], _state(p, g[50], flip)
        if now is None or then[0] is None:
            return None
        d_goal = lambda x: float(np.linalg.norm(x - GOAL_CENTRE))
        d_ball = lambda x, k: float(np.linalg.norm(x - ball[k])) if np.isfinite(ball[k]).all() else nan
        return dict(d_goal=d_goal(now), d_ball=d_ball(now, f), dd_goal=d_goal(now) - d_goal(then[0]),
                    dd_ball=d_ball(now, f) - d_ball(then[0], g[50]), speed_then=float(np.linalg.norm(then[1])))

    defs = [m for m in (moves(p) for p in rep.players if p.team != team) if m]
    defs.sort(key=lambda m: m["d_goal"])           # gk = nearest own goal, defb = the other one
    for tag, i in (("gk", 0), ("defb", 1)):
        if len(defs) > i:
            out.update({f"{tag}_dd_goal_h50": defs[i]["dd_goal"], f"{tag}_dd_ball_h50": defs[i]["dd_ball"],
                        f"{tag}_speed_h50": defs[i]["speed_then"]})
    if len(defs) > 1:
        out.update(defb_d_goal=defs[1]["d_goal"], defb_d_ball=defs[1]["d_ball"])
    mates = [m for m in (moves(p) for p in rep.players if p.team == team and p.pid != shooter.pid) if m]
    if mates:
        m = min(mates, key=lambda m: m["d_goal"])
        out.update(mate_dd_goal_h50=m["dd_goal"], mate_dd_ball_h50=m["dd_ball"])
    return out


def game_state(rep, f: int, team: int) -> dict:
    own = sum(g.team == team and g.frame < f for g in rep.goals)
    opp = sum(g.team != team and g.frame < f for g in rep.goals)
    secs = float(rep.seconds_remaining[f])   # (seconds_remaining itself is already a column of the shot table)
    return dict(score_diff=float(own - opp), overtime=float(secs <= 0 and own == opp and rep.time[f] > 60))
