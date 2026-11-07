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
