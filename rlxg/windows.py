"""Shot windows: the raw ball and car states around every shot, kept so that new
features can be computed after the replays are deleted.

For each shot: 2 s before the touch to 1.5 s after, sampled at 15 Hz (53 samples),
for the ball and every player in the replay. World coordinates (not the attacking
frame); `team` says who attacked which goal (team 0 attacks +y; for team 1, negate x and y
of positions and velocities, and the same for ball spin).

One compressed .npz per replay, next to the parse cache:

    ball     (S, W, 9)  float16   pos xyz, vel xyz, angular velocity xyz (rad/s)
    cars     (S, W, P, 14) float16  pos xyz, vel xyz, quat xyzw, boost 0-100, jump, double jump, dodge
    t_rel    (S, W)     float32   sample time relative to the touch (nearest replay frame)
    frame, post_frame, goal, team, shooter (index into pids)   (S,)
    pids, player_teams  (P,)

float16 keeps positions to within ~4 uu and velocities to ~2-4 uu/s, plenty for features.
NaN marks a car that isn't replicated at that moment (demolished, not yet spawned).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HZ = 15
BEFORE, AFTER = 2.0, 1.5
OFFSETS = np.arange(-round(BEFORE * HZ), round(AFTER * HZ) + 1) / HZ
CAR_FIELDS = ["x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "boost", "jump", "double_jump", "dodge"]
BALL_FIELDS = ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]


def shot_windows(rep, shots: pd.DataFrame) -> dict:
    n = len(shots)
    P, W = len(rep.players), len(OFFSETS)
    frames = shots.frame.to_numpy(int) if n else np.zeros(0, int)
    idx = np.clip(np.searchsorted(rep.time, rep.time[frames][:, None] + OFFSETS[None, :]), 0, rep.n_frames - 1)
    ang = rep.ball_ang_vel if rep.ball_ang_vel is not None else np.full_like(rep.ball_vel, np.nan)
    ball_all = np.concatenate([rep.ball_pos, rep.ball_vel, ang], axis=1)
    car_all = np.stack([np.concatenate([p.pos, p.vel, p.quat, p.boost[:, None], p.jump[:, None], p.double_jump[:, None],
                                        p.dodge[:, None]], axis=1) for p in rep.players], axis=1) if P else \
        np.zeros((rep.n_frames, 0, len(CAR_FIELDS)))
    pids = np.array([p.pid for p in rep.players])
    return dict(
        ball=ball_all[idx].astype(np.float16),
        cars=car_all[idx].astype(np.float16),
        t_rel=(rep.time[idx] - rep.time[frames][:, None]).astype(np.float32),
        frame=frames, post_frame=shots.post_frame.to_numpy(int) if n else frames,
        goal=shots.goal.to_numpy(np.int8) if n else np.zeros(0, np.int8),
        team=shots.team.to_numpy(np.int8) if n else np.zeros(0, np.int8),
        shooter=np.array([int(np.flatnonzero(pids == pid)[0]) for pid in shots.pid]) if n else np.zeros(0, int),
        pids=pids, player_teams=np.array([p.team for p in rep.players], np.int8),
        replay_id=np.array(rep.replay_id),
    )


def save(path: Path, w: dict) -> None:
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **w)
    tmp.replace(path)


def load(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
