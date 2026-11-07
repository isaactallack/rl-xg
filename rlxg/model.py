"""xG model: feature preparation, training artefact and inference.

Two classifiers share the shot set:
  * xGOT ("shot xG"): uses where the touch actually sent the ball (predicted
    goal-line crossing, margins to the posts, defender intercept along that path).
    Rates the shot as struck: a scuffed open-net miss is near zero.
  * chance xG (`xg`): uses only the situation the instant *before* the touch
    (ball, shooter approach, defenders relative to the ball->goal line). Rates the
    chance, like football xG, so missing an open net still costs a big chance.

Each has two parts:
  * a LightGBM classifier that learns shot physics, trained mostly on high-level
    (SSL) 2v2 games, where there is lots of data;
  * per-skill-band calibrations: a logistic correction on the logit of the
    classifier's output, fitted on games from that rank band. The physics/ranking
    transfers across ranks, but conversion rates differ (better defenders at SSL),
    so the band sets the overall level. Default band: C3-GC2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xg_model.joblib"
DEFAULT_BAND = "c3-gc2"

# Signed x-quantities are folded so the model is left/right symmetric.
FEATURES = [
    # ball state just after the touch
    "ball_abs_x", "ball_y", "ball_z", "ball_vx_to_centre", "ball_vy", "ball_vz", "ball_speed",
    "dist_goal", "goal_angle", "angle_off_centre", "dir_cos_goal", "touch_dv",
    # predicted free-flight trajectory
    "on_target", "cross_abs_x", "cross_z", "cross_margin", "time_to_goal", "cross_speed", "n_bounces",
    # shooter
    "shooter_z", "shooter_speed", "shooter_vy", "shooter_airborne", "shooter_boost", "shooter_dodge",
    "carry_time",
    # build-up
    "since_prev_touch", "prev_by_self", "prev_by_teammate", "prev_by_opponent",
    # defence
    "n_def", "n_def_goal_side", "n_def_in_cone", "n_def_can_intercept",
    "def1_margin", "def1_d_path", "def1_d_ball", "def1_boost", "def1_z", "def1_speed",
    "def2_margin", "def2_d_path",
    "gk_d_goal", "gk_d_cross", "gk_goal_side", "gk_vy", "nearest_def_ball",
    # support
    "n_att", "mate_d_goal", "mate_d_ball",
]

# Chance xG: nothing that depends on where the touch sent the ball.
# (ball_x/y/z are read one frame after contact; the ball has barely moved.)
CHANCE_FEATURES = [
    # ball before the touch
    "ball_abs_x", "ball_y", "pre_ball_z", "dist_goal", "goal_angle", "angle_off_centre",
    "pre_ball_vx_to_centre", "pre_ball_vy", "pre_ball_vz", "pre_ball_speed", "pre_ball_dir_cos_goal",
    # shooter approach
    "shooter_z", "shooter_speed", "shooter_vy", "shooter_airborne", "shooter_boost", "shooter_dodge",
    "shooter_d_ball", "shooter_align", "shooter_facing_goal", "shooter_facing_ball",
    "shooter_v_to_goal", "shooter_v_to_ball", "carry_time",
    # build-up
    "since_prev_touch", "prev_by_self", "prev_by_teammate", "prev_by_opponent",
    # defence, relative to the straight ball->goal line
    "n_def", "n_def_goal_side", "n_def_in_cone", "nearest_def_ball", "gk_d_goal", "gk_goal_side", "gk_vy",
    "chance_def1_t", "chance_def1_d", "chance_def1_u", "chance_def2_t", "chance_def_goal_t", "chance_ball_goal_t",
    # support
    "n_att", "mate_d_goal", "mate_d_ball",
]

# Minimal "classic" model (distance + angle) used as a baseline in evaluation.
BASELINE_FEATURES = ["dist_goal", "goal_angle"]


def prepare(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    features = features or FEATURES
    X = df.copy()
    X["ball_vx_to_centre"] = -np.sign(X["ball_x"]) * X["ball_vx"]
    for c in features:
        if c not in X:
            X[c] = np.nan
    return X[features].astype(float)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_calibration(p: np.ndarray, ab: tuple[float, float] | None) -> np.ndarray:
    if ab is None:
        return p
    a, b = ab
    return 1 / (1 + np.exp(-(a + b * logit(p))))


@dataclass
class XGModel:
    booster: object
    calibrator: object | None = None                    # legacy (unused by new models)
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)
    calibrations: dict = field(default_factory=dict)    # band -> (intercept, slope) on logit
    default_band: str = DEFAULT_BAND
    # chance xG (pre-touch) model; absent in models trained before it existed
    chance_booster: object | None = None
    chance_features: list[str] = field(default_factory=lambda: list(CHANCE_FEATURES))
    chance_calibrations: dict = field(default_factory=dict)

    @property
    def has_chance(self) -> bool:
        return getattr(self, "chance_booster", None) is not None

    @property
    def bands(self) -> list[str]:
        return list(getattr(self, "calibrations", {}) or {})

    def predict_raw(self, shots: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(prepare(shots, self.features))[:, 1]

    def predict_chance(self, shots: pd.DataFrame, band: str | None = None) -> np.ndarray:
        """Chance xG (pre-touch). Falls back to xGOT for models without a chance part."""
        if not self.has_chance:
            return self.predict(shots, band)
        if len(shots) == 0:
            return np.zeros(0)
        p = self.chance_booster.predict_proba(prepare(shots, self.chance_features))[:, 1]
        cals = self.chance_calibrations or {}
        if cals:
            band = band or getattr(self, "default_band", DEFAULT_BAND)
            if band not in cals:
                raise ValueError(f"unknown skill band {band!r}; choose from {list(cals)}")
            p = apply_calibration(p, cals[band])
        return np.clip(p, 1e-4, 1 - 1e-4)

    def predict(self, shots: pd.DataFrame, band: str | None = None) -> np.ndarray:
        """xGOT (post-touch)."""
        if len(shots) == 0:
            return np.zeros(0)
        p = self.predict_raw(shots)
        cals = getattr(self, "calibrations", {}) or {}
        if cals:
            band = band or getattr(self, "default_band", DEFAULT_BAND)
            if band not in cals:
                raise ValueError(f"unknown skill band {band!r}; choose from {list(cals)}")
            p = apply_calibration(p, cals[band])
        elif self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path = DEFAULT_MODEL_PATH) -> "XGModel":
        return joblib.load(path)
