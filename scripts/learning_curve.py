"""Does more data still help? Learning curves for both models.

    python scripts/learning_curve.py data/shots.csv            # writes data/learning_curve.json

Trains each model on growing random slices of the high-level training replays (500 up to all
of them) and scores every one on the same held-out 20% of high-level replays and on the C3-GC2
games (calibrated by grouped CV, as in compare_features.py). Every fit also records log loss
after each tree, on its own training slice and on the held-out games, so the figure can show
where training stops helping and whether the model overfits.

Uses the quicker settings of the comparison scripts (1,200 trees at learning rate 0.05), so the
absolute numbers are a little worse than the saved model's; the shape of the curve is the point.
docs/make_figures.py turns the JSON into docs/img/fig_learning_curve.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from compare_defence import PARAMS, fit_cal, ll
from rlxg.model import CHANCE_FEATURES, FEATURES, apply_calibration, prepare

SIZES = [500, 1000, 2000, 5000, 10000, 20000]      # training replays; "all" is added on top
PHYSICS_SOURCES = ["hlrlrd_2v2", "legacy_2v2"]


def is_test(replay_id: str) -> bool:
    """A fixed 20% of high-level replays, chosen by hash so every run uses the same ones."""
    return int(hashlib.md5(str(replay_id).encode()).hexdigest(), 16) % 5 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="?", default=str(ROOT / "data" / "shots.csv"))
    ap.add_argument("-o", "--out", default=str(ROOT / "data" / "learning_curve.json"))
    args = ap.parse_args()

    cols = sorted(set(FEATURES) | set(CHANCE_FEATURES) | {"ball_x", "ball_vx", "goal", "source", "replay_id",
                                                          "team_size"})
    head = pd.read_csv(args.shots, nrows=0).columns
    df = pd.read_csv(args.shots, usecols=[c for c in cols if c in head], low_memory=False)
    df = df[df.team_size == 2].reset_index(drop=True)
    y, rid = df.goal.to_numpy(), df.replay_id.astype(str).to_numpy()
    phys = df.source.isin(PHYSICS_SOURCES).to_numpy()
    test_rep = np.array([is_test(r) for r in rid])
    test = phys & (df.source == "hlrlrd_2v2").to_numpy() & test_rep
    pool = phys & ~test
    c3 = np.flatnonzero((df.source == "ballchasing_c3_gc2").to_numpy())
    pool_reps = np.sort(pd.unique(rid[pool]))
    np.random.default_rng(0).shuffle(pool_reps)
    sizes = [s for s in SIZES if s < len(pool_reps)] + [len(pool_reps)]
    print(f"training pool {len(pool_reps)} replays / {pool.sum()} shots | test {test.sum()} shots | C3-GC2 {len(c3)}")

    out = {"params": PARAMS, "n_test": int(test.sum()), "n_c3": int(len(c3)), "runs": []}
    for label, feats in (("xGOT", FEATURES), ("chance xG", CHANCE_FEATURES)):
        X = prepare(df, feats).astype(np.float32)
        Xte, yte = X[test], y[test]
        for n in sizes:
            tr = pool & np.isin(rid, pool_reps[:n])
            t0 = time.time()
            m = LGBMClassifier(**PARAMS).fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (Xte, yte)],
                                             eval_names=["train", "test"], eval_metric="binary_logloss")
            ev = m.evals_result_
            # score at the best number of trees: with few games the model overfits long before 1,200
            # trees, and the curve should show what more data buys a model tuned for that size, not
            # that overfitting. Picked on the test games, so slightly optimistic, equally for every size.
            best = int(np.argmin(ev["test"]["binary_logloss"]) + 1)
            p = m.predict_proba(X, num_iteration=best)[:, 1]
            pc = np.zeros(len(c3))
            for a, b in GroupKFold(5).split(c3, groups=rid[c3]):
                pc[b] = apply_calibration(p[c3[b]], fit_cal(p[c3[a]], y[c3[a]]))
            run = dict(model=label, replays=int(n), shots=int(tr.sum()),
                       test_ll=float(ll(yte, p[test])), c3_ll=float(ll(y[c3], pc)),
                       train_ll=float(ev["train"]["binary_logloss"][best - 1]),
                       test_ll_all_trees=float(ev["test"]["binary_logloss"][-1]), best_iter=best,
                       curve_train=[round(v, 5) for v in ev["train"]["binary_logloss"]],
                       curve_test=[round(v, 5) for v in ev["test"]["binary_logloss"]],
                       seconds=round(time.time() - t0, 1))
            out["runs"].append(run)
            print(f"{label:9s} {n:6d} replays {run['shots']:8d} shots | test {run['test_ll']:.4f}  "
                  f"C3-GC2 {run['c3_ll']:.4f}  train {run['train_ll']:.4f}  best tree {run['best_iter']}"
                  f"  ({run['seconds']:.0f}s)", flush=True)
            Path(args.out).write_text(json.dumps(out))     # keep partial results if stopped
    print("wrote", args.out)


if __name__ == "__main__":
    main()
