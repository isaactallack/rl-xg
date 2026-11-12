"""Ball prediction with RocketSim: the real arena mesh, goal frame, ramps and ball spin.

This replaces the flat-plane predictor in physics.py for shot selection and the
trajectory features. On shots nobody touched again it calls goal / no goal right
98% of the time (the flat-plane model: 94%) and recognises ~93% of post and bar hits
(scripts/compare_physics.py).

RocketSim (https://github.com/ZealanL/RocketSim) needs the arena collision meshes,
which are dumped from your own copy of Rocket League with RLArenaCollisionDumper.
Put the dumped `collision_meshes` folder at `data/collision_meshes` (or point the
`RS_COLLISION_MESHES` environment variable at it).

Everything here works in the *attacking frame* (target goal at +y), like
physics.py. The soccar arena is symmetric under a half turn about the vertical
axis, so a state mirrored into the attacking frame (x, y -> -x, -y, the same for
spin) is simulated as is.
"""
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import physics as ph

TICK_RATE = 120
DT = 1.0 / TICK_RATE
SAMPLE_EVERY = 12                 # ticks: keep a trajectory sample every 0.1 s, like physics.py
BALL_RADIUS = 91.25               # RocketSim soccar collision radius (the visual radius is 92.75)
SCORE_Y = 5124.25 + BALL_RADIUS   # ball centre past this = goal (RocketSim's soccar rule)
# Where a shot "arrives": the ball centre reaching the plane where it would touch the back
# wall. Goals pass through it inside the mouth; wide and high shots meet the wall there.
CROSS_Y = ph.ARENA_Y - BALL_RADIUS - 1.0
END_KINDS = ("post", "bar", "wide", "over")   # first events that happen at the target end
IMPULSE_DV = 200.0                # uu/s velocity change in one tick beyond gravity/drag = a contact
BOUNCE_DV = 350.0                 # same threshold rlxg.outcome uses on the real ball

# Jitter for P(goal): how far off the replay's post-touch ball state plausibly is (network
# compression, sampling a frame late). Rough guesses, not fitted.
JITTER_RUNS = 16
JIT_POS = 5.0                     # uu, per axis
JIT_VEL = 0.01                    # fraction of speed, per axis
JIT_ANG = 0.3                     # rad/s, per axis

DEFAULT_MESHES = Path(__file__).resolve().parent.parent / "data" / "collision_meshes"

_arena = None


def meshes_dir() -> Path:
    return Path(os.getenv("RS_COLLISION_MESHES") or DEFAULT_MESHES)


def _get_arena():
    global _arena
    if _arena is None:
        try:
            import RocketSim as rs
        except ImportError as e:
            raise ImportError("RocketSim is required for ball prediction: pip install rocketsim") from e
        d = meshes_dir()
        if not (d / "soccar").is_dir():
            raise FileNotFoundError(
                f"RocketSim arena meshes not found at {d}. Dump them from Rocket League with "
                "RLArenaCollisionDumper (github.com/ZealanL/RLArenaCollisionDumper) and put the "
                "collision_meshes folder there, or set RS_COLLISION_MESHES.")
        rs.init(str(d))
        _arena = rs.Arena(rs.GameMode.SOCCAR, tick_rate=TICK_RATE)
    return _arena


def run(pos, vel, ang_vel, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities for ticks+1 states (the first is the start), nothing else in the arena."""
    import RocketSim as rs
    arena = _get_arena()
    st = rs.BallState()
    st.pos = rs.Vec(*map(float, pos))
    st.vel = rs.Vec(*map(float, vel))
    st.ang_vel = rs.Vec(*map(float, ang_vel))
    arena.ball.set_state(st)
    states = arena.get_ball_prediction(ticks + 1, 1)    # starts with the state itself; the arena doesn't advance
    P = np.array([s.pos.as_tuple() for s in states])
    V = np.array([s.vel.as_tuple() for s in states])
    return P, V


def _impulses(V: np.ndarray) -> np.ndarray:
    """Ticks where something other than gravity changed the ball's velocity (index of the state after)."""
    dv = np.diff(V, axis=0)
    dv[:, 2] -= ph.GRAVITY * DT
    return np.flatnonzero(np.linalg.norm(dv, axis=1) > IMPULSE_DV) + 1


def first_event(P: np.ndarray, V: np.ndarray) -> tuple[str, int]:
    """What the ball did first, by the rules rlxg.outcome applies to the real ball.

    Returns (kind, tick): kind is goal / post / bar / wide / over / wall / own_goal / none.
    """
    from .outcome import _classify_bounce
    scored = np.flatnonzero(np.abs(P[:, 1]) > SCORE_Y)
    t_score = scored[0] if len(scored) else len(P)
    dv = np.diff(V[:t_score], axis=0)
    dv[:, 2] -= ph.GRAVITY * DT
    for k in np.flatnonzero(np.linalg.norm(dv, axis=1) > BOUNCE_DV) + 1:
        got = _classify_bounce(P[k - 1]) or _classify_bounce(P[k])
        if got:
            return got, int(k)
    if len(scored):
        return ("goal" if P[t_score, 1] > 0 else "own_goal"), int(t_score)
    return "none", len(P) - 1


@dataclass
class Prediction(ph.Prediction):
    """physics.Prediction plus what RocketSim says happens next."""
    sim_kind: np.ndarray = None      # (N,) first event: goal / post / bar / wide / over / wall / own_goal / none
    sim_goal: np.ndarray = None      # (N,) bool: goes in unaided within the horizon
    sim_time: np.ndarray = None      # (N,) seconds until that first event


def predict(pos: np.ndarray, vel: np.ndarray, ang_vel: np.ndarray | None = None,
            horizon: float = 5.0) -> Prediction:
    """Drop-in for physics.predict (attacking frame), with spin and the real arena."""
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    n = len(pos)
    ang = np.zeros((n, 3)) if ang_vel is None else np.nan_to_num(np.asarray(ang_vel, float))
    ticks = int(round(horizon * TICK_RATE))
    n_samples = ticks // SAMPLE_EVERY + 1
    st = np.arange(n_samples) * SAMPLE_EVERY * DT
    samples = np.full((n, n_samples, 3), np.nan)
    reached = np.zeros(n, bool)
    cross_x = np.full(n, np.nan)
    cross_z = np.full(n, np.nan)
    ttg = np.full(n, np.nan)
    cspeed = np.full(n, np.nan)
    bounces = np.zeros(n, int)
    kind = np.empty(n, object)
    sim_t = np.full(n, np.nan)

    for i in range(n):
        P, V = run(pos[i], vel[i], ang[i], ticks)
        kind[i], k_ev = first_event(P, V)
        sim_t[i] = k_ev * DT
        # arrival: the centre reaching the back-wall plane, or first hitting the frame / end wall
        # (a ball rolling wide meets the curved base of the back wall before its centre gets there)
        hit = np.flatnonzero(P[:, 1] >= CROSS_Y)
        k = hit[0] if len(hit) else None
        if kind[i] in END_KINDS and (k is None or k_ev < k):
            k, frac, c = k_ev, 0.0, P[k_ev]
        elif k is not None:
            if k == 0:
                c, frac = P[0], 0.0
            else:
                frac = (CROSS_Y - P[k - 1, 1]) / max(P[k, 1] - P[k - 1, 1], 1e-9)
                c = P[k - 1] + (P[k] - P[k - 1]) * frac
        if k is not None:
            reached[i] = True
            cross_x[i], cross_z[i] = c[0], c[2]
            ttg[i] = max(k - 1 + frac, 0.0) * DT
            cspeed[i] = float(np.linalg.norm(V[k]))
            last = k
        else:
            last = len(P) - 1
        imp = _impulses(V[:last + 1])
        bounces[i] = int(np.sum(np.diff(np.concatenate([[-10], imp])) > 2))    # merge multi-tick contacts
        keep = np.arange(0, last + 1, SAMPLE_EVERY)
        samples[i, keep // SAMPLE_EVERY] = P[keep]

    on_target = reached & (np.abs(cross_x) < ph.GOAL_HALF_W) & (cross_z < ph.GOAL_H)
    return Prediction(reached, cross_x, cross_z, ttg, cspeed, on_target, st, samples, bounces,
                      sim_kind=kind, sim_goal=kind == "goal", sim_time=sim_t)


def p_goal(pos, vel, ang_vel, horizon: float = 5.0, runs: int = JITTER_RUNS, seed: str = "") -> float:
    """Fraction of slightly perturbed copies of the start state that go in unaided."""
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    pos, vel, ang = (np.asarray(a, float) for a in (pos, vel, np.nan_to_num(ang_vel)))
    spd = float(np.linalg.norm(vel))
    ticks = int(round(horizon * TICK_RATE))
    n = 0
    for _ in range(runs):
        P, V = run(pos + rng.normal(0, JIT_POS, 3), vel + rng.normal(0, JIT_VEL * spd, 3),
                   ang + rng.normal(0, JIT_ANG, 3), ticks)
        n += first_event(P, V)[0] == "goal"
    return n / runs


# --- used by scripts/compare_physics.py ----------------------------------------------------
@dataclass
class SimPath:
    t: np.ndarray            # (T,) seconds from the start state
    pos: np.ndarray          # (T, 3)
    vel: np.ndarray          # (T, 3)
    scored_y_sign: int       # +1 ball went fully into the +y goal, -1 the -y goal, 0 neither


def simulate(pos, vel, ang_vel, seconds: float) -> SimPath:
    """One path, cut where the ball is fully over a goal line."""
    P, V = run(pos, vel, ang_vel, int(round(seconds * TICK_RATE)))
    over = np.flatnonzero(np.abs(P[:, 1]) > SCORE_Y)
    scored = 0
    if len(over):
        k = over[0]
        scored = int(np.sign(P[k, 1]))
        P, V = P[:k + 1], V[:k + 1]
    return SimPath(np.arange(len(P)) * DT, P, V, scored)
