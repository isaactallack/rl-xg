"""Do the kinematic defender features (rlxg/defence.py) help? Old feature lists vs old + new.

    python scripts/compare_defence.py                      # reads data/cache_v6 as it stands

Works on a partial rebuild: it reads whatever replays are finished in the cache, so
it can be run while build_dataset.py is still going, or after it was stopped.

Both models are trained on the same shots with the same (quick) settings:
  * train: 80% of the high-level replays in the cache
  * test : the other 20% (raw), C3-GC2 (grouped-CV logit calibration), your games
           (calibrated on C3-GC2), exactly as scripts/train.py scores them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from rlxg.model import (BASE_CHANCE_FEATURES, BASE_FEATURES, CHANCE_DEFENCE_FEATURES, DEFENCE_FEATURES,
                        apply_calibration, logit, prepare)

PARAMS = dict(n_estimators=1200, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, random_state=0, verbose=-1)


def load_cache(cache: Path) -> pd.DataFrame:
    dfs = []
    for marker in cache.glob("*.json"):
        f = marker.with_suffix(".csv")
        try:
            d = pd.read_csv(f, low_memory=False)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if len(d):
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def fit_cal(p, y):
    lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_v6"))
    args = ap.parse_args()

    df = load_cache(Path(args.cache))
    df = df[df.team_size == 2].reset_index(drop=True)
    y, rid = df.goal.to_numpy(), df.replay_id.to_numpy()
    ssl = (df.source == "hlrlrd_2v2").to_numpy()
    c3 = (df.source == "ballchasing_c3_gc2").to_numpy()
    own = (df.source == "Demos").to_numpy()
    reps = np.sort(df.loc[ssl, "replay_id"].unique())
    np.random.default_rng(0).shuffle(reps)
    test = ssl & np.isin(rid, reps[: len(reps) // 5])
    train = ssl & ~test
    print(f"{df.replay_id.nunique()} replays in the cache | train {train.sum()} shots / {len(reps) - len(reps) // 5} "
          f"high-level replays | test {test.sum()} | C3-GC2 {c3.sum()} | your games {own.sum()}\n")
    if train.sum() < 20000:
        sys.exit("Not enough high-level shots yet (want 20,000+). Let the build run longer.")

    # the shots the old model finds hardest: going in unaided, a defender who can just about get there
    hard = (df.sim_goal == 1).to_numpy() & df.def1_margin.between(-1.0, -0.1).to_numpy()

    for label, base, extra in (("xGOT", BASE_FEATURES, DEFENCE_FEATURES),
                               ("chance xG", BASE_CHANCE_FEATURES, CHANCE_DEFENCE_FEATURES)):
        print(f"==================== {label} ====================")
        rows = []
        for name, feats in (("old features", base), ("old + kinematic defence", base + extra)):
            X = prepare(df, feats)
            b = LGBMClassifier(**PARAMS).fit(X[train], y[train])
            p = b.predict_proba(X)[:, 1]
            ci = np.flatnonzero(c3)
            p_c3 = np.zeros(len(ci))
            if len(ci):
                for tr, te in GroupKFold(5).split(ci, groups=rid[ci]):
                    p_c3[te] = apply_calibration(p[ci[te]], fit_cal(p[ci[tr]], y[ci[tr]]))
            sets = [("high-level test", y[test], p[test]),
                    ("high-level test, hard shots", y[test & hard], p[test & hard])]
            if len(ci):
                sets.append(("C3-GC2 (CV cal)", y[c3], p_c3))
                if own.any():
                    sets.append(("your games", y[own], apply_calibration(p[own], fit_cal(p[c3], y[c3]))))
            for games, yy, pp in sets:
                rows.append(dict(features=name, games=games, n=len(yy), goals=int(yy.sum()), xg=round(float(pp.sum()), 1),
                                 log_loss=ll(yy, pp), auc=roc_auc_score(yy, pp)))
            if feats is not base:
                imp = pd.Series(b.booster_.feature_importance("gain"), index=feats)
                top = (imp / imp.sum()).sort_values(ascending=False).head(12).round(3)
        res = pd.DataFrame(rows)
        wide = res.pivot(index="games", columns="features", values="log_loss")[["old features", "old + kinematic defence"]]
        wide["change"] = wide["old + kinematic defence"] - wide["old features"]
        wide["change %"] = 100 * wide["change"] / wide["old features"]
        auc = res.pivot(index="games", columns="features", values="auc")[["old features", "old + kinematic defence"]]
        pd.set_option("display.width", 170)
        print("log loss\n" + wide.round(4).to_string())
        print("\nAUC\n" + auc.round(4).to_string())
        print("\ntop features by gain (old + kinematic defence)\n" + top.to_string() + "\n")


if __name__ == "__main__":
    main()
