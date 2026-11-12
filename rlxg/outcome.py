"""What happened to each shot after it was struck, and a short clip of the moment.

`shot_outcomes` follows the real ball from the shot touch until the first of:

* a goal (shown as the predicted crossing point, even when it was deflected in),
* any car touching it (an opponent stopping a shot that was heading on target,
  near their goal = "saved"; anything else an opponent stops = "blocked"),
* a bounce off a post, the crossbar, the back wall beside/above the goal
  ("wide" / "over"), a side wall, ramp or the ceiling ("wall").

Floor bounces in the open don't stop it: rolling and bouncing shots are normal.
If nothing happens before the ball would have arrived (plus a second), the shot
ends where the touch was sending it.

End points are in the attacking frame (target goal at +y), like the shot's own
`ball_x` / `ball_y` columns.

`shot_clip` cuts the ball and every car out of the replay around one shot, for
the dashboard's 3D view. Clip coordinates are world coordinates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import physics as ph
from . import rsim
from .parse import Replay
from .touches import Touch

R = ph.BALL_RADIUS
GRAVITY = -650.0
BOUNCE_DV = 350.0          # uu/s unexplained velocity change with no car near = the ball hit something
AFTER_ARRIVAL = 1.0        # s: keep following this long after the predicted arrival
SAVE_ZONE = 2200.0         # uu from the goal line: an opponent touch on an on-target shot inside this is a save
CLIP_BEFORE = 3.0          # s of build-up in a clip
CLIP_AFTER_END = 1.5       # s shown after the shot ended
CLIP_MAX_AFTER = 7.0       # s: a clip never runs longer than this after the touch

KIND_LABELS = {
    "goal": "Goal", "saved": "Saved", "blocked": "Blocked", "post": "Hit the post", "bar": "Hit the bar",
    "wide": "Wide", "over": "Over the bar", "wall": "Off the wall", "short": "Fell short",
}


def _flip(team: int) -> np.ndarray:
    return np.array([1.0, 1.0, 1.0]) if team == 0 else np.array([-1.0, -1.0, 1.0])


def _classify_bounce(p: np.ndarray) -> str | None:
    """Kind of surface the ball bounced off at attacking-frame position p (None = open floor, keep going)."""
    x, y, z = abs(p[0]), p[1], p[2]
    if y > ph.ARENA_Y - R - 140:                       # at the target end
        if x < ph.GOAL_HALF_W - R - 40 and z < ph.GOAL_H - R - 40:
            return None                                # inside the goal mouth: net / line, not a miss
        # the frame bars are thin: a ball centre further out than this was against the back wall
        if x < ph.GOAL_HALF_W + 40 and ph.GOAL_H - R - 60 < z < ph.GOAL_H + 50:
            return "bar"
        if ph.GOAL_HALF_W - R - 60 < x < ph.GOAL_HALF_W + 50 and z < ph.GOAL_H + 50:
            return "post"
        if z < R + 60 and y < ph.ARENA_Y - R - 60:
            return None                                # floor bounce just short of the line
        return "over" if z >= ph.GOAL_H and x < ph.GOAL_HALF_W + R else "wide"
    if z < R + 60 and x < ph.ARENA_X - R - 400 and abs(y) < ph.ARENA_Y - R - 400:
        return None                                    # open floor
    return "wall"


def shot_outcomes(rep: Replay, touches: list[Touch], shots: pd.DataFrame) -> pd.DataFrame:
    """One row per shot (same order): end_x/y/z (attacking frame), end_kind, end_frame, end_by."""
    by_frame = {t.frame: i for i, t in enumerate(touches)}
    names = {p.pid: p.name for p in rep.players}
    n = rep.n_frames
    rows = []
    for s in shots.itertuples(index=False):
        team = int(s.team) if not isinstance(s.team, str) else (0 if s.team == "blue" else 1)
        flip = _flip(team)
        ti = by_frame.get(int(s.frame))
        cross = np.array([s.cross_x, ph.ARENA_Y, s.cross_z])
        if s.goal:
            rows.append(dict(end_x=cross[0], end_y=cross[1], end_z=cross[2], end_kind="goal",
                             end_frame=int(s.goal_frame), end_by=None))
            continue
        if ti is None:                                  # shouldn't happen; fall back to the prediction
            rows.append(dict(end_x=cross[0], end_y=cross[1], end_z=cross[2],
                             end_kind="wide" if not s.on_target else "short", end_frame=int(s.frame), end_by=None))
            continue
        tch = touches[ti]
        pf = min(tch.post_frame, n - 1)
        nxt = touches[ti + 1] if ti + 1 < len(touches) else None
        ttg = s.time_to_goal if np.isfinite(s.time_to_goal) else 5.0
        t_limit = rep.time[pf] + min(ttg, 5.0) + AFTER_ARRIVAL
        last = int(np.searchsorted(rep.time, t_limit))
        stop_touch = nxt is not None and nxt.frame <= last
        last = min(last, nxt.frame if stop_touch else last, n - 1)

        kind, end_k = None, None
        for k in range(pf + 1, last + 1):
            p, p0 = rep.ball_pos[k] * flip, rep.ball_pos[k - 1] * flip
            if not (np.isfinite(p).all() and np.isfinite(p0).all()) or np.linalg.norm(p - p0) > 1500:
                end_k = k - 1                           # ball reset / missing: stop here
                break
            dt = rep.time[k] - rep.time[k - 1]
            if not 0 < dt < 0.25:
                continue
            dv = rep.ball_vel[k] - rep.ball_vel[k - 1]
            dv[2] -= GRAVITY * dt
            if not np.isfinite(dv).all() or np.linalg.norm(dv) < BOUNCE_DV:
                continue
            if stop_touch and k >= nxt.frame:          # that impulse is the next touch, handled below
                break
            # use the position on the side of the impulse that is closest to the surface
            got = _classify_bounce(p0) or _classify_bounce(p)
            if got:
                kind, end_k = got, k - 1
                break

        if kind is None and end_k is None and stop_touch:
            end_k = nxt.frame
            if nxt.team != team:
                # a save is made near goal; cutting out an on-target shot further up the pitch is a block
                p_t = rep.ball_pos[nxt.frame] * flip
                kind = "saved" if s.on_target and p_t[1] > ph.ARENA_Y - SAVE_ZONE else "blocked"
                by = names.get(nxt.pid)
            else:
                kind = "short" if s.on_target else ("over" if s.cross_z >= ph.GOAL_H else "wide")
                by = None
            p = rep.ball_pos[end_k] * flip
            rows.append(dict(end_x=p[0], end_y=p[1], end_z=p[2], end_kind=kind, end_frame=int(end_k), end_by=by))
            continue
        if kind is None:
            # nothing hit it before it would have arrived
            if s.on_target:
                kind = "short"
            else:
                kind = "over" if s.cross_z >= ph.GOAL_H and abs(s.cross_x) < ph.GOAL_HALF_W + R else "wide"
            if end_k is None:
                end_k = last
            if kind in ("wide", "over") and np.isfinite(cross).all():
                p = cross
            else:
                p = rep.ball_pos[end_k] * flip
        else:
            p = rep.ball_pos[end_k] * flip
        rows.append(dict(end_x=p[0], end_y=p[1], end_z=p[2], end_kind=kind, end_frame=int(end_k), end_by=None))
    return pd.DataFrame(rows, index=shots.index)


def _r(a: np.ndarray, d: int = 0) -> list:
    """Round an array to a JSON-ready nested list, NaN -> None."""
    a = np.round(a, d)
    out = a.astype(object)
    out[~np.isfinite(a.astype(float))] = None
    if d == 0:
        f = np.isfinite(a.astype(float))
        out[f] = a[f].astype(int)
    return out.tolist()


def shot_clip(rep: Replay, shot: dict) -> dict:
    """Ball and cars from CLIP_BEFORE s before the shot to just after it ended (world coordinates)."""
    f = int(shot["frame"])
    t0 = float(rep.time[f])
    end_frame = int(shot.get("end_frame", f))
    t_end = max(float(rep.time[min(max(end_frame, f), rep.n_frames - 1)]), t0) + CLIP_AFTER_END
    t_end = min(t_end, t0 + CLIP_MAX_AFTER)
    a = int(np.searchsorted(rep.time, t0 - CLIP_BEFORE))
    b = min(int(np.searchsorted(rep.time, t_end)) + 1, rep.n_frames)
    # don't run into the kickoff reset after a goal
    if shot.get("goal") and 0 <= shot.get("goal_frame", -1) < b:
        b = min(b, int(np.searchsorted(rep.time, rep.time[shot["goal_frame"]] + 1.2)) + 1)
    sl = slice(a, b)
    team = 0 if shot["team"] in (0, "blue") else 1
    flip = _flip(team)

    # where the touch was sending the ball (the xGOT view), world coordinates
    post = min(int(shot.get("post_frame", f + 1)), rep.n_frames - 1)
    bp, bv = rep.ball_pos[post] * flip, rep.ball_vel[post] * flip
    ba = rep.ball_ang_vel[post] * flip if rep.ball_ang_vel is not None else np.zeros(3)
    pred = []
    if np.isfinite(bp).all() and np.isfinite(bv).all():
        # RocketSim path up to the first thing the ball would hit (the net, a post, the back wall...)
        P, V = rsim.run(bp, bv, np.nan_to_num(ba), int(5.0 * rsim.TICK_RATE))
        _, k = rsim.first_event(P, V)
        pts = np.vstack([P[:k:rsim.SAMPLE_EVERY // 2], P[k]])
        pred = _r(pts * flip)

    cars = []
    for p in rep.players:
        cars.append(dict(
            pid=p.pid, name=p.name, team="blue" if p.team == 0 else "orange",
            pos=_r(p.pos[sl]), quat=_r(p.quat[sl], 3), boost=_r(p.boost[sl]),
        ))
    end = [shot.get("end_x"), shot.get("end_y"), shot.get("end_z")]
    end = (np.array(end, float) * flip).round().tolist() if all(v is not None for v in end) else None
    return dict(
        t=np.round(rep.time[sl] - t0, 3).tolist(),
        ball=_r(rep.ball_pos[sl]),
        cars=cars,
        pred=pred,
        shooter=shot.get("pid"),
        team="blue" if team == 0 else "orange",
        t_end=round(float(rep.time[min(max(end_frame, f), rep.n_frames - 1)]) - t0, 3),
        end=end,
        end_kind=shot.get("end_kind"),
    )
