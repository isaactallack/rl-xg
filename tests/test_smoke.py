"""Smoke tests: physics sanity and an end-to-end run on any available replay.

    .venv/Scripts/python -m pytest tests
"""
from pathlib import Path

import numpy as np
import pytest

from rlxg import physics as ph

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = sorted((ROOT / "data" / "raw").glob("*/*.replay"))


def test_rolling_ball_reaches_goal():
    # ground ball straight at goal from midfield must arrive on target
    p = ph.predict(np.array([[0.0, 0.0, 93.0]]), np.array([[0.0, 2000.0, 0.0]]))
    assert p.on_target[0] and 2.0 < p.time_to_goal[0] < 3.5


def test_ball_away_from_goal_never_arrives():
    p = ph.predict(np.array([[0.0, 0.0, 93.0]]), np.array([[0.0, -2000.0, 0.0]]))
    assert not p.reaches_goal_plane[0]


def test_wide_shot_is_off_target():
    p = ph.predict(np.array([[2500.0, 3000.0, 93.0]]), np.array([[0.0, 2500.0, 0.0]]))
    assert p.reaches_goal_plane[0] and not p.on_target[0]


from rlxg import rsim

needs_meshes = pytest.mark.skipif(not (rsim.meshes_dir() / "soccar").is_dir(), reason="no RocketSim arena meshes")


@needs_meshes
def test_rocketsim_straight_shot_scores():
    p = rsim.predict(np.array([[0.0, 0.0, 93.0]]), np.array([[0.0, 2000.0, 0.0]]))
    assert p.on_target[0] and p.sim_goal[0] and 2.0 < p.time_to_goal[0] < 3.5


@needs_meshes
def test_rocketsim_crossbar_and_post():
    # ball centre at bar height / post line: hits the frame, doesn't score
    bar = rsim.predict(np.array([[0.0, 3000.0, 642.0]]), np.array([[0.0, 3000.0, 218.0]]))
    assert bar.sim_kind[0] == "bar" and not bar.sim_goal[0]
    post = rsim.predict(np.array([[893.0, 3000.0, 200.0]]), np.array([[0.0, 3000.0, 216.0]]))
    assert post.sim_kind[0] == "post" and not post.sim_goal[0]


@needs_meshes
def test_rocketsim_wide_ground_shot_arrives():
    # rolls wide into the curved base of the back wall: still a (wide) arrival at the target end
    p = rsim.predict(np.array([[-1040.0, 3000.0, 93.0]]), np.array([[0.0, 2000.0, 0.0]]))
    assert p.reaches_goal_plane[0] and not p.on_target[0] and p.sim_kind[0] == "wide"


@needs_meshes
@pytest.mark.skipif(not REPLAYS, reason="no replays under data/raw")
def test_end_to_end():
    from rlxg import analyze_replay
    m = analyze_replay(REPLAYS[0])
    assert len(m.shots) > 0
    assert m.shots.xg.between(0, 1).all()
    assert m.shots.xgot.between(0, 1).all()
    # every chance-xG feature is extracted (none silently filled with NaN by prepare)
    from rlxg.model import CHANCE_FEATURES, prepare
    from rlxg.shots import extract_shots
    raw = extract_shots(m.replay)
    missing = [c for c in CHANCE_FEATURES if c not in raw and c != "ball_vx_to_centre"]
    assert not missing, missing
    assert prepare(raw, CHANCE_FEATURES).notna().mean().min() > 0.5
    # xG totals are consistent between team and player tables
    assert abs(m.teams.xg.sum() - m.players.xg.sum()) < 0.05


@pytest.mark.skipif(not REPLAYS, reason="no replays under data/raw")
def test_shot_windows_round_trip(tmp_path):
    from rlxg import windows
    from rlxg.parse import load_replay
    from rlxg.shots import extract_shots
    from rlxg.situation import AIMS
    rep = load_replay(REPLAYS[0])
    shots = extract_shots(rep)
    windows.save(tmp_path / "w.npz", windows.shot_windows(rep, shots))
    w = windows.load(tmp_path / "w.npz")
    S, W = len(shots), len(windows.OFFSETS)
    assert w["ball"].shape == (S, W, 9) and w["cars"].shape == (S, W, len(rep.players), 14)
    # the sample at offset 0 is the touch frame itself, and matches the replay
    k = int(np.flatnonzero(windows.OFFSETS == 0)[0])
    assert np.allclose(w["t_rel"][:, k], 0)
    assert np.allclose(w["ball"][:, k, :3].astype(float), rep.ball_pos[shots.frame.to_numpy()], atol=8)
    # coverage features are fractions of the goal-mouth grid
    assert shots.cov_open.between(0, 1).all() and len(AIMS) == 21
    assert shots.score_diff.abs().max() <= 15
