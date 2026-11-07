"""How should high-level (SSL) data be used to predict C3-GC2 shots?

Evaluates on shots from --target sources with grouped 5-fold CV by replay; the
non-target data is always fully available for training. Variants:

  target only        train on the C3-GC2 folds only
  other only         train on everything else (mostly SSL), no adjustment
  pooled             train on everything, no adjustment
  skill flag         pooled + a 'high_level' feature (1 for SSL data); predict with 0
  other + recal      'other only' model, then logistic recalibration fitted on target train folds
  pooled + recal     'pooled' model + the same recalibration

    python scripts/compare_sources.py data/shots.csv --target ballchasing_c3_gc2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from rlxg.model import FEATURES, prepare
from train import LGBM_PARAMS, calibration_table, metrics

HIGH_LEVEL_SOURCES = {"hlrlrd_2v2"}


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="?", default="data/shots.csv")
    ap.add_argument("--target", nargs="+", default=["ballchasing_c3_gc2"])
    ap.add_argument("--team-size", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.shots, low_memory=False)
    df = df[df.team_size == args.team_size].reset_index(drop=True)
    X = prepare(df)
    X["high_level"] = df.source.isin(HIGH_LEVEL_SOURCES).astype(float).to_numpy()
    y = df.goal.to_numpy()
    tgt = df.source.isin(args.target).to_numpy()
    ti, oi = np.flatnonzero(tgt), np.flatnonzero(~tgt)
    print(f"target shots {len(ti)} ({df[tgt].replay_id.nunique()} replays), other {len(oi)}")

    F, FS = FEATURES, FEATURES + ["high_level"]
    names = ["target only", "other only", "pooled", "skill flag", "other + recal", "pooled + recal"]
    preds = {k: np.zeros(len(ti)) for k in names}
    groups = df.replay_id.to_numpy()[ti]
    # models trained without target data are the same for every fold: fit once
    m_other = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[oi][F], y[oi])
    p_other_all = m_other.predict_proba(X.iloc[ti][F])[:, 1]
    for k, (tr, te) in enumerate(GroupKFold(5).split(ti, groups=groups)):
        tr_i, te_i = ti[tr], ti[te]
        pool = np.concatenate([tr_i, oi])
        preds["target only"][te] = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[tr_i][F], y[tr_i]) \
            .predict_proba(X.iloc[te_i][F])[:, 1]
        preds["other only"][te] = p_other_all[te]
        m_pool = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[pool][F], y[pool])
        preds["pooled"][te] = m_pool.predict_proba(X.iloc[te_i][F])[:, 1]
        m_flag = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[pool][FS], y[pool])
        Xte = X.iloc[te_i][FS].copy()
        Xte["high_level"] = 0.0
        preds["skill flag"][te] = m_flag.predict_proba(Xte)[:, 1]
        # recalibration: logistic on logit(p) fitted on the target training folds
        for base, model in (("other", m_other), ("pooled", m_pool)):
            p_tr = model.predict_proba(X.iloc[tr_i][F])[:, 1]
            cal = LogisticRegression(C=1e6).fit(logit(p_tr)[:, None], y[tr_i])
            p_te = model.predict_proba(X.iloc[te_i][F])[:, 1]
            preds[f"{base} + recal"][te] = cal.predict_proba(logit(p_te)[:, None])[:, 1]
        print(f"  fold {k + 1}/5 done", flush=True)

    rows = [{"variant": k, **metrics(y[ti], p)} for k, p in preds.items()]
    res = pd.DataFrame(rows).sort_values("log_loss")
    pd.set_option("display.width", 160)
    print(res.round(4).to_string(index=False))
    best = res.iloc[0]["variant"]
    print(f"\nCalibration on target, '{best}'")
    print(calibration_table(y[ti], preds[best]).round(3).to_string())


if __name__ == "__main__":
    main()
