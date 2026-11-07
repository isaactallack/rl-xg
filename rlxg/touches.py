"""Touch detection from raw physics.

subtr-actor's own touch events are team-level only (from the ball's HitTeamNum)
and miss most touches, so touches are re-derived here: a touch is a frame where
the ball's velocity changes by more than gravity explains *and* a car hitbox is
in contact range. The closest car (hitbox surface distance) gets the touch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parse import Replay

GRAVITY = -650.0
BALL_RADIUS = 92.75
# Octane-ish hitbox (half extents) and offset from car origin, in car local frame
HITBOX_HALF = np.array([59.0, 42.1, 18.1])
HITBOX_OFFSET = np.array([13.88, 0.0, 20.75])

DV_THRESHOLD = 120.0      # uu/s unexplained velocity change to call it an impulse
CONTACT_SLACK = 110.0     # uu beyond the ball radius (frame-rate / replication lag)
MERGE_GAP = 0.12          # s: impulses by same player closer than this are one touch


@dataclass
class Touch:
    frame: int            # last frame before the impulse (contact frame)
    post_frame: int       # first frame with post-touch velocity
    time: float
    pid: str
    team: int
    surface_dist: float   # car hitbox surface -> ball centre at contact
    dv: float             # magnitude of the impulse (uu/s)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N,4) quaternions (x,y,z,w) -> (N,3,3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def hitbox_distance(car_pos: np.ndarray, car_quat: np.ndarray, ball_pos: np.ndarray) -> np.ndarray:
    """Distance from ball centre to the car's oriented hitbox surface (0 if inside)."""
    rot = quat_to_matrix(car_quat)
    rel = ball_pos - car_pos
    local = np.einsum("nji,nj->ni", rot, rel) - HITBOX_OFFSET   # R^T @ rel
    excess = np.maximum(np.abs(local) - HITBOX_HALF, 0.0)
    d = np.linalg.norm(excess, axis=1)
    # Fall back to centre distance if the quaternion is missing
    bad = ~np.isfinite(d)
    if bad.any():
        d[bad] = np.linalg.norm(rel[bad], axis=1) - 60.0
    return d


def surface_distances(rep: Replay) -> np.ndarray:
    """(F, P) hitbox-surface distance of every player to the ball."""
    out = np.full((rep.n_frames, len(rep.players)), np.inf)
    for j, p in enumerate(rep.players):
        ok = np.isfinite(p.pos).all(1) & np.isfinite(rep.ball_pos).all(1)
        if ok.any():
            out[ok, j] = hitbox_distance(p.pos[ok], p.quat[ok], rep.ball_pos[ok])
    return out


def detect_touches(rep: Replay) -> list[Touch]:
    t, bp, bv = rep.time, rep.ball_pos, rep.ball_vel
    n = rep.n_frames
    if n < 3:
        return []
    dist = surface_distances(rep)
    dt = np.diff(t)
    dv = bv[1:] - bv[:-1]
    dv[:, 2] -= GRAVITY * dt
    resid = np.linalg.norm(dv, axis=1)
    # Ignore the teleport at kickoff resets / explosions
    jump = np.linalg.norm(bp[1:] - bp[:-1], axis=1)
    valid = np.isfinite(resid) & (dt > 0) & (dt < 0.25) & (jump < 1500)

    # contact range: min over the frame before and after the impulse
    near = np.minimum(dist[:-1], dist[1:])                     # (F-1, P)
    best = np.argmin(near, axis=1)
    best_d = near[np.arange(n - 1), best]
    cand = valid & (resid > DV_THRESHOLD) & (best_d < BALL_RADIUS + CONTACT_SLACK)

    touches: list[Touch] = []
    for i in np.flatnonzero(cand):
        p = rep.players[best[i]]
        tch = Touch(frame=int(i), post_frame=int(i + 1), time=float(t[i]), pid=p.pid,
                    team=p.team, surface_dist=float(best_d[i]), dv=float(resid[i]))
        if touches and touches[-1].pid == tch.pid and tch.time - touches[-1].time < MERGE_GAP:
            # same contact spread over consecutive frames: keep the latest post state
            prev = touches[-1]
            prev.post_frame = tch.post_frame
            prev.dv = max(prev.dv, tch.dv)
            continue
        touches.append(tch)
    return touches
