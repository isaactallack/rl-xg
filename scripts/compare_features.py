"""Which of the extra feature groups (rlxg/situation.py) help? Each one added on its own, then all of them.

    python scripts/compare_features.py --cache data/cache_v7

Same shots, split and quick settings as compare_defence.py: train on 80% of the high-level
replays, test on the other 20%, C3-GC2 (grouped-CV calibration) and your games
(calibrated on C3-GC2). Differences in log loss are paired, on the same shots, with a
replay-clustered bootstrap standard error, so small gains can be told from noise.
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
from sklearn.model_selection import GroupKFold

from compare_defence import PARAMS, fit_cal, ll, load_cache
from rlxg.model import CHANCE_FEATURES, FEATURE_GROUPS, FEATURES, apply_calibration, prepare


def pointwise(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_se(y, p_new, p_old, groups, n=300):
    d = pd.Series(pointwise(y, p_new) - pointwise(y, p_old)).groupby(groups).agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(1)
    vals = [(lambda k: d[k, 0].sum() / d[k, 1].sum())(rng.integers(0, len(d), len(d))) for _ in range(n)]
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_v7"))
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
    print(f"{df.replay_id.nunique()} replays | train {train.sum()} shots | test {test.sum()} | "
          f"C3-GC2 {c3.sum()} | your games {own.sum()}\n")
    if train.sum() < 20000:
        sys.exit("Not enough high-level shots yet (want 20,000+).")
    ci = np.flatnonzero(c3)
    extra = {c for g in FEATURE_GROUPS.values() for c in g}

    def evaluate(feats):
        X = prepare(df, feats)
        p = LGBMClassifier(**PARAMS).fit(X[train], y[train]).predict_proba(X)[:, 1]
        out = {"high-level test": (np.flatnonzero(test), p[test])}
        if len(ci):
            pc = np.zeros(len(ci))
            for tr, te in GroupKFold(5).split(ci, groups=rid[ci]):
                pc[te] = apply_calibration(p[ci[te]], fit_cal(p[ci[tr]], y[ci[tr]]))
            out["C3-GC2 (CV cal)"] = (ci, pc)
            if own.any():
                out["your games"] = (np.flatnonzero(own), apply_calibration(p[own], fit_cal(p[c3], y[c3])))
        return out

    for label, all_feats in (("xGOT", FEATURES), ("chance xG", CHANCE_FEATURES)):
        print(f"==================== {label} ====================")
        base = [c for c in all_feats if c not in extra]
        groups = {g: [c for c in cols if c in all_feats] for g, cols in FEATURE_GROUPS.items()}
        variants = {"base": base, **{f"+ {g}": base + cols for g, cols in groups.items() if cols}, "+ all": all_feats}
        res = {name: evaluate(feats) for name, feats in variants.items()}
        rows = []
        for name in variants:
            for games, (idx, p) in res[name].items():
                p0 = res["base"][games][1]
                rows.append(dict(variant=name, games=games, n=len(idx), log_loss=ll(y[idx], p),
                                 change=ll(y[idx], p) - ll(y[idx], p0),
                                 se=paired_se(y[idx], p, p0, rid[idx]) if name != "base" else 0.0))
        r = pd.DataFrame(rows)
        pd.set_option("display.width", 170)
        print(r.pivot(index="variant", columns="games", values="change").loc[list(variants)].round(4).to_string())
        print("\n(standard errors)")
        print(r.pivot(index="variant", columns="games", values="se").loc[list(variants)].round(4).to_string())
        print("\nbase log loss: " + ", ".join(f"{g} {ll(y[i], p):.4f}" for g, (i, p) in res["base"].items()) + "\n")


if __name__ == "__main__":
    main()
