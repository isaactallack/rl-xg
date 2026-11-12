"""Can a defender get to the ball's path in time? A kinematic estimate.

The older intercept margin (shots.py: `def1_margin`) is straight-line distance over
top speed. It can't tell a defender driving at the path from one driving away, or a
rolling ball from one 600 uu up. Here each defender gets, for every point on the
path, the time they need to get there:

* along the ground: starting from their current speed *towards that point* (negative
  when they are heading away), accelerating while their boost lasts, then cruising;
* a turn penalty when they are on their wheels and facing the wrong way;
* a climb time when the ball is above what a car on the ground can touch: a jump,
  a double jump, or an aerial (which needs boost).

The margin is that time minus when the ball gets there; negative = they can make it.
The constants are rough game values, not fitted: the model learns how much to trust
them, and the raw ingredients (closing speed, facing, ball height at the meeting
point...) are given as features too.

Everything is in the attacking frame (target goal at +y).
"""
from __future__ import annotations

import numpy as np

from . import physics as ph

REACH = 160.0             # uu, car centre to ball centre at contact
GROUND_TOUCH_Z = 150.0    # ball centre this high can be hit without jumping
MAX_SPEED = 2300.0
THROTTLE_SPEED = 1410.0   # top speed without boost
BOOST_ACCEL = 1500.0      # uu/s^2, rough average of throttle + boost below top speed
THROTTLE_ACCEL = 700.0    # uu/s^2, rough average of throttle alone up to 1410
BOOST_PER_S = 33.3
TURN_RATE = 3.0           # rad/s
TURN_FREE = 0.3           # rad: no penalty for being this far off line
JUMP_SPEED = 500.0        # uu/s upwards from a (double) jump
AERIAL_ACCEL = 350.0      # uu/s^2 net climb while boosting upwards
JUMP_MAX_H = 450.0        # how far a double jump climbs without boost
T_CAP = 5.0
PATH_POINTS = 20          # samples on the straight ball->goal line (pre-touch view)

KEYS = ("margin", "t_int", "int_z", "int_gap_y", "d_int", "closing", "facing", "reach_frac", "t_first",
        "z", "vz", "speed", "vy", "boost", "on_ground", "jumped", "up_t", "drive_t")
DEFAULTS = dict(margin=T_CAP, t_int=T_CAP, int_z=0.0, int_gap_y=8000.0, d_int=8000.0, closing=0.0, facing=0.0,
                reach_frac=0.0, t_first=T_CAP, z=17.0, vz=0.0, speed=0.0, vy=0.0, boost=0.0, on_ground=1.0,
                jumped=0.0, up_t=0.0, drive_t=T_CAP)


def drive_time(dist: np.ndarray, v0: np.ndarray, boost: float) -> np.ndarray:
    """Seconds to cover `dist` starting at speed v0 along the line (v0 < 0: heading away)."""
    dist = np.maximum(np.asarray(dist, float), 0.0)
    v0 = np.clip(np.asarray(v0, float), -MAX_SPEED, MAX_SPEED)
    if boost > 1.0:
        a = BOOST_ACCEL
        v1 = np.clip(v0 + a * boost / BOOST_PER_S, THROTTLE_SPEED, MAX_SPEED)
    else:
        a = THROTTLE_ACCEL
        v1 = np.full_like(v0, THROTTLE_SPEED)
    v1 = np.maximum(v1, v0)                       # already faster than that: keep the speed
    t_a = (v1 - v0) / a
    d_a = v0 * t_a + 0.5 * a * t_a ** 2
    t_acc = (-v0 + np.sqrt(np.maximum(v0 ** 2 + 2 * a * dist, 0.0))) / a
    return np.where(dist <= d_a, t_acc, t_a + (dist - d_a) / np.maximum(v1, 1.0))


def climb_time(h: np.ndarray, boost: float) -> np.ndarray:
    """Seconds to get `h` uu higher than a grounded car can reach."""
    h = np.maximum(np.asarray(h, float), 0.0)
    if boost >= 10.0:
        return (-JUMP_SPEED + np.sqrt(JUMP_SPEED ** 2 + 2 * AERIAL_ACCEL * h)) / AERIAL_ACCEL
    return np.where(h <= JUMP_MAX_H, h / 400.0, T_CAP)


def need_time(pos, vel, fwd, up, boost: float, path: np.ndarray, **_) -> np.ndarray:
    """Seconds this defender needs to get to each point of `path` (N, 3); the core of intercept()."""
    pos, vel = np.asarray(pos, float), np.nan_to_num(np.asarray(vel, float))
    on_ground = bool(pos[2] < 40.0 and (not np.isfinite(up).all() or up[2] > 0.9))
    d = path - pos
    dh = np.linalg.norm(d[:, :2], axis=1)
    u = d[:, :2] / np.maximum(dh, 1e-6)[:, None]
    t_drive = drive_time(dh - REACH, u @ vel[:2], boost)
    if on_ground and np.isfinite(fwd).all() and np.linalg.norm(fwd[:2]) > 1e-6:
        cos_face = np.clip(u @ (fwd[:2] / np.linalg.norm(fwd[:2])), -1.0, 1.0)
        t_drive = t_drive + np.maximum(np.arccos(cos_face) - TURN_FREE, 0.0) / TURN_RATE
    t_up = climb_time(path[:, 2] - GROUND_TOUCH_Z - max(pos[2] - 17.0, 0.0), boost)
    return np.minimum(np.maximum(t_drive, t_up), T_CAP)


def intercept(pos, vel, fwd, up, boost: float, jumped: bool, path: np.ndarray, t: np.ndarray) -> dict:
    """Features for one defender against one path (path: (N, 3) points, t: (N,) arrival times)."""
    out = dict(DEFAULTS)
    pos, vel = np.asarray(pos, float), np.nan_to_num(np.asarray(vel, float))
    on_ground = bool(pos[2] < 40.0 and (not np.isfinite(up).all() or up[2] > 0.9))
    out.update(z=float(pos[2]), vz=float(vel[2]), speed=float(np.linalg.norm(vel)), vy=float(vel[1]),
               boost=float(boost), on_ground=float(on_ground), jumped=float(jumped))
    if len(path) == 0:
        return out
    d = path - pos
    dh = np.linalg.norm(d[:, :2], axis=1)
    u = d[:, :2] / np.maximum(dh, 1e-6)[:, None]
    closing = u @ vel[:2]
    t_drive = drive_time(dh - REACH, closing, boost)
    if np.isfinite(fwd).all() and np.linalg.norm(fwd[:2]) > 1e-6:
        f2 = fwd[:2] / np.linalg.norm(fwd[:2])
        cos_face = np.clip(u @ f2, -1.0, 1.0)
    else:
        cos_face = np.ones(len(path))
    if on_ground:
        t_drive = t_drive + np.maximum(np.arccos(cos_face) - TURN_FREE, 0.0) / TURN_RATE
    t_up = climb_time(path[:, 2] - GROUND_TOUCH_Z - max(pos[2] - 17.0, 0.0), boost)
    need = np.minimum(np.maximum(t_drive, t_up), T_CAP)
    m = need - t
    k = int(np.argmin(m))
    ok = m < 0
    out.update(
        margin=float(m[k]), t_int=float(t[k]), int_z=float(path[k, 2]),
        int_gap_y=float(ph.ARENA_Y - path[k, 1]),             # how far from the goal line they would meet it
        d_int=float(np.linalg.norm(d[k])), closing=float(closing[k]), facing=float(cos_face[k]),
        reach_frac=float(ok.mean()), t_first=float(t[ok][0]) if ok.any() else T_CAP,
        up_t=float(t_up[k]), drive_t=float(min(t_drive[k], T_CAP)),
    )
    return out


def straight_path(ball: np.ndarray, aim: np.ndarray, t_total: float) -> tuple[np.ndarray, np.ndarray]:
    """The straight ball->aim line as a path, for the pre-touch (chance) view."""
    fr = np.linspace(0.0, 1.0, PATH_POINTS)
    return ball + fr[:, None] * (aim - ball), fr * t_total


def defence_features(defenders: list[dict], path: np.ndarray, t: np.ndarray, prefix: str) -> dict:
    """`{prefix}1_*` / `{prefix}2_*` for the two defenders best placed to cut the path out, plus counts.

    defenders: dicts with pos, vel, fwd, up, boost, jumped (attacking frame).
    """
    feats = sorted((intercept(d["pos"], d["vel"], d["fwd"], d["up"], d["boost"], d["jumped"], path, t)
                    for d in defenders), key=lambda r: r["margin"])
    row = {}
    for i in (0, 1):
        r = feats[i] if len(feats) > i else DEFAULTS
        for key in KEYS:
            row[f"{prefix}{i + 1}_{key}"] = r[key]
    row[f"{prefix}_n_can"] = float(sum(r["margin"] < 0 for r in feats))
    row[f"{prefix}_n_can_easy"] = float(sum(r["margin"] < -0.3 for r in feats))
    return row


def feature_names(prefix: str) -> list[str]:
    return [f"{prefix}{i}_{k}" for i in (1, 2) for k in KEYS] + [f"{prefix}_n_can", f"{prefix}_n_can_easy"]
