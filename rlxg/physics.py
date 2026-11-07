"""Vectorised ballistic ball predictor (gravity, drag, simple bounces).

Approximates RL ball physics well enough to answer "where would this touch send
the ball if nobody else touched it?". Curved ramps are ignored (flat planes).
Coordinates are always in the *attacking* frame: the target goal is at +y.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = -650.0
BALL_RADIUS = 92.75
DRAG = 0.0305            # fraction of velocity lost per second
MAX_SPEED = 6000.0
RESTITUTION = 0.6
FRICTION = 0.3          # effective ball-surface friction coefficient
ROLL_SPEED = 250.0       # normal speed below which a contact is rolling, not bouncing
ARENA_X = 4096.0
ARENA_Y = 5120.0
CEILING = 2044.0
GOAL_HALF_W = 892.755
GOAL_H = 642.775

DT = 1 / 60
SAMPLE_EVERY = 6         # keep a trajectory sample every 0.1 s


@dataclass
class Prediction:
    reaches_goal_plane: np.ndarray   # (N,) bool: ball centre reaches y = ARENA_Y in horizon
    cross_x: np.ndarray              # (N,) x at goal plane (nan if not reached)
    cross_z: np.ndarray              # (N,) z at goal plane
    time_to_goal: np.ndarray         # (N,) seconds until goal plane (nan if not reached)
    cross_speed: np.ndarray          # (N,) ball speed at the goal plane
    on_target: np.ndarray            # (N,) crosses inside the goal mouth
    sample_t: np.ndarray             # (S,)
    samples: np.ndarray              # (N, S, 3) ball centre; NaN after goal plane / horizon
    bounces: np.ndarray              # (N,) bounces before reaching the goal plane


def predict(pos: np.ndarray, vel: np.ndarray, horizon: float = 4.0) -> Prediction:
    pos = np.array(pos, dtype=np.float64, copy=True)
    vel = np.array(vel, dtype=np.float64, copy=True)
    n = len(pos)
    steps = int(round(horizon / DT))
    n_samples = steps // SAMPLE_EVERY + 1
    samples = np.full((n, n_samples, 3), np.nan)
    samples[:, 0] = pos
    alive = np.ones(n, bool)
    reached = np.zeros(n, bool)
    cross_x = np.full(n, np.nan)
    cross_z = np.full(n, np.nan)
    ttg = np.full(n, np.nan)
    cspeed = np.full(n, np.nan)
    bounces = np.zeros(n, int)
    r = BALL_RADIUS

    # ball already past the goal plane (touch on the line)
    inside = pos[:, 1] >= ARENA_Y
    reached[inside] = True
    cross_x[inside] = pos[inside, 0]
    cross_z[inside] = pos[inside, 2]
    ttg[inside] = 0.0
    cspeed[inside] = np.linalg.norm(vel[inside], axis=1)
    alive[inside] = False

    for k in range(1, steps + 1):
        a = alive
        if not a.any():
            break
        v = vel[a]
        v[:, 2] += GRAVITY * DT
        v *= (1 - DRAG * DT)
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(sp > MAX_SPEED, v * MAX_SPEED / np.maximum(sp, 1e-9), v)
        p_old = pos[a]
        p = p_old + v * DT

        # goal plane crossing (centre reaches y = ARENA_Y)
        crossing = (p[:, 1] >= ARENA_Y) & (p_old[:, 1] < ARENA_Y)
        if crossing.any():
            frac = (ARENA_Y - p_old[crossing, 1]) / np.maximum(p[crossing, 1] - p_old[crossing, 1], 1e-9)
            cp = p_old[crossing] + (p[crossing] - p_old[crossing]) * frac[:, None]
            idx = np.flatnonzero(a)[crossing]
            reached[idx] = True
            cross_x[idx] = cp[:, 0]
            cross_z[idx] = cp[:, 2]
            ttg[idx] = (k - 1 + frac) * DT
            cspeed[idx] = np.linalg.norm(v[crossing], axis=1)

        idx_a = np.flatnonzero(a)

        # bounces off floor / ceiling / side walls / own back wall
        def bounce(mask, axis):
            vn = v[:, axis].copy()
            real = mask & (np.abs(vn) > ROLL_SPEED)   # rolling/sliding: no friction loss
            soft = mask & ~real
            if real.any():
                # Coulomb friction: tangential speed loss bounded by mu * normal impulse
                vt = v[real].copy()
                vt[:, axis] = 0.0
                vt_mag = np.linalg.norm(vt, axis=1)
                loss = np.minimum(vt_mag, FRICTION * (1 + RESTITUTION) * np.abs(vn[real]))
                scale = 1 - loss / np.maximum(vt_mag, 1e-9)
                v[real] = vt * scale[:, None]
                v[real, axis] = -RESTITUTION * vn[real]
            v[soft, axis] = 0.0
            bounces[idx_a[real]] += 1

        hit = p[:, 2] < r
        p[hit, 2] = r
        bounce(hit & (v[:, 2] < 0), 2)
        hit = p[:, 2] > CEILING - r
        p[hit, 2] = CEILING - r
        bounce(hit & (v[:, 2] > 0), 2)
        for s in (-1, 1):
            hit = s * p[:, 0] > ARENA_X - r
            p[hit, 0] = s * (ARENA_X - r)
            bounce(hit & (s * v[:, 0] > 0), 0)
        hit = p[:, 1] < -ARENA_Y + r
        p[hit, 1] = -ARENA_Y + r
        bounce(hit & (v[:, 1] < 0), 1)

        pos[a] = p
        vel[a] = v
        if k % SAMPLE_EVERY == 0:
            samples[idx_a, k // SAMPLE_EVERY] = p
        # stop simulating once the goal plane is reached
        alive[np.flatnonzero(a)[crossing]] = False

    on_target = reached & (np.abs(cross_x) < GOAL_HALF_W) & (cross_z < GOAL_H)
    # blank samples after the goal plane
    st = np.arange(n_samples) * SAMPLE_EVERY * DT
    after = st[None, :] > np.where(reached, ttg, np.inf)[:, None]
    samples[after] = np.nan
    return Prediction(reached, cross_x, cross_z, ttg, cspeed, on_target, st, samples, bounces)
