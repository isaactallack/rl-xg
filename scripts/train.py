"""Train the 2v2 models, xGOT (post-touch) and chance xG (pre-touch): each a classifier plus per-rank-band calibrations.

    python scripts/train.py data/shots.csv

Data roles (by replay folder = `source` column):
  * physics training : hlrlrd_2v2 (high-level/SSL, minus a 10% replay holdout) + legacy_2v2
  * SSL calibration  : the 10% hlrlrd_2v2 holdout
  * C3-GC2 calibration: ballchasing_c3_gc2 (never used to train the classifier)
  * your games       : Demos (never used for training or calibration) - a true holdout

Metrics printed are all out-of-sample: the C3-GC2 numbers use 5-fold grouped CV
of the calibration step; SSL and Demos are clean holdouts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from rlxg.model import CHANCE_FEATURES, FEATURES, XGModel, apply_calibration, logit, prepare

LGBM_PARAMS = dict(
    n_estimators=4000, learning_rate=0.02, num_leaves=63, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
    random_state=0, verbose=-1,
)
PHYSICS_SOURCES = ["hlrlrd_2v2", "legacy_2v2"]
SSL_SOURCE = "hlrlrd_2v2"
C3GC2_SOURCE = "ballchasing_c3_gc2"
OWN_SOURCE = "Demos"


def metrics(y, p):
    return dict(n=len(y), goals=int(y.sum()), xg=round(float(p.sum()), 1),
                log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan)


def calibration_table(y, p, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)):
    b = pd.cut(p, bins, right=False)
    return pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True).agg(
        shots=("y", "size"), mean_xg=("p", "mean"), goal_rate=("y", "mean"))


def fit_cal(p, y) -> tuple[float, float]:
    lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="?", default="data/shots.csv")
    ap.add_argument("-o", "--out", default=str(ROOT / "models" / "xg_model.joblib"))
    ap.add_argument("--compare", default=None, help="previous model to compare against (default: current saved model)")
    args = ap.parse_args()

    df = pd.read_csv(args.shots, low_memory=False)
    df = df[df.team_size == 2].reset_index(drop=True)
    y = df.goal.to_numpy()
    rid = df.replay_id.to_numpy()

    # 10% of SSL replays held out for the SSL calibration + evaluation
    rng = np.random.default_rng(0)
    ssl_replays = df.loc[df.source == SSL_SOURCE, "replay_id"].unique()
    ssl_hold = set(rng.choice(ssl_replays, size=len(ssl_replays) // 10, replace=False))
    is_ssl_hold = np.isin(rid, list(ssl_hold))
    phys = df.source.isin(PHYSICS_SOURCES).to_numpy() & ~is_ssl_hold
    c3 = (df.source == C3GC2_SOURCE).to_numpy()
    own = (df.source == OWN_SOURCE).to_numpy()
    print(f"physics training: {phys.sum()} shots / {df[phys].replay_id.nunique()} replays")
    if df[phys].replay_id.nunique() < 500 or not c3.any():
        sys.exit(f"Not enough data to train: {df[phys].replay_id.nunique()} high-level replays (need 500+) and "
                 f"{df[c3].replay_id.nunique()} C3-GC2 replays. If the rebuild is still running, let it parse "
                 f"more (it does your games and C3-GC2 first, then the high-level ones), then re-assemble.")
    print(f"C3-GC2: {c3.sum()} shots / {df[c3].replay_id.nunique()} replays | SSL holdout: {is_ssl_hold.sum()} shots"
          f" | your games: {own.sum()} shots / {df[own].replay_id.nunique()} replays\n")

    old = None
    old_path = Path(args.compare or args.out)
    if old_path.exists():
        try:
            old = XGModel.load(old_path)
        except Exception as e:
            print("could not load previous model:", e)

    def fit_part(label: str, feats: list[str]):
        """Classifier + per-band calibrations for one feature set; prints out-of-sample metrics."""
        print(f"\n==================== {label} ====================")
        X = prepare(df, feats)
        t = time.time()
        booster = LGBMClassifier(**LGBM_PARAMS).fit(X[phys], y[phys])
        print(f"trained classifier in {time.time() - t:.0f}s")
        p_raw = booster.predict_proba(X)[:, 1]
        cal_c3 = fit_cal(p_raw[c3], y[c3])
        cal_ssl = fit_cal(p_raw[is_ssl_hold], y[is_ssl_hold])
        # out-of-sample C3-GC2 calibration via grouped CV
        p_c3_cv = np.zeros(c3.sum())
        ci = np.flatnonzero(c3)
        for tr, te in GroupKFold(5).split(ci, groups=rid[ci]):
            p_c3_cv[te] = apply_calibration(p_raw[ci[te]], fit_cal(p_raw[ci[tr]], y[ci[tr]]))

        rows = []
        def add(name, subset, yy, pp):
            rows.append({"part": label, "model": name, "games": subset, **metrics(yy, pp)})
        add("new, C3-GC2 calibration", "C3-GC2 (CV)", y[c3], p_c3_cv)
        add("new, uncalibrated", "C3-GC2", y[c3], p_raw[c3])
        add("new, C3-GC2 calibration", "your games", y[own], apply_calibration(p_raw[own], cal_c3))
        add("new, SSL calibration", "SSL holdout", y[is_ssl_hold], apply_calibration(p_raw[is_ssl_hold], cal_ssl))
        if old is not None and label == "xGOT":
            add("previous model", "C3-GC2", y[c3], old.predict(df[c3]))
            add("previous model", "SSL holdout", y[is_ssl_hold], old.predict(df[is_ssl_hold]))
            add("previous model (saw these in training)", "your games", y[own], old.predict(df[own]))
        res = pd.DataFrame(rows)
        pd.set_option("display.width", 170)
        print(res.drop(columns="part").round(4).to_string(index=False))
        print("\nCalibration on C3-GC2 (cross-validated)")
        print(calibration_table(y[c3], p_c3_cv).round(3).to_string())
        print("\nCalibration on your games (never used for training)")
        print(calibration_table(y[own], apply_calibration(p_raw[own], cal_c3)).round(3).to_string())
        imp = pd.Series(booster.booster_.feature_importance("gain"), index=feats)
        print("\nTop features by gain")
        print((imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())
        return booster, {"c3-gc2": cal_c3, "ssl": cal_ssl}, res

    booster, cals, res = fit_part("xGOT", FEATURES)
    ch_booster, ch_cals, ch_res = fit_part("chance xG", CHANCE_FEATURES)
    res = pd.concat([res, ch_res], ignore_index=True)

    model = XGModel(
        booster=booster, features=list(FEATURES),
        calibrations=cals, default_band="c3-gc2",
        chance_booster=ch_booster, chance_features=list(CHANCE_FEATURES), chance_calibrations=ch_cals,
        meta=dict(
            trained_at=time.strftime("%Y-%m-%d %H:%M"),
            n_shots_physics=int(phys.sum()), n_replays_physics=int(df[phys].replay_id.nunique()),
            n_shots_c3gc2=int(c3.sum()), n_shots_ssl_holdout=int(is_ssl_hold.sum()),
            params=LGBM_PARAMS,
            eval=json.loads(res.to_json(orient="records")),
        ),
    )
    model.save(args.out)
    print(f"\nSaved {args.out}  (bands: {model.bands}, default {model.default_band})")


if __name__ == "__main__":
    main()
