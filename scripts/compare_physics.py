"""How well does each ball simulator predict shots that nobody touched again?

For every shot whose ball wasn't touched again before it ended (a clean goal, the post or
bar, wide, over, off the wall, or falling short), the ball is played forward from its state
just after the touch with:

* the built-in predictor (rlxg/physics.py: flat planes, no spin, no goal frame), and
* RocketSim (rlxg/rsim.py: real arena mesh, goal frame, ramps, spin), once from the
  replay state and --jitter more times from slightly perturbed copies,

and both are compared with what the real ball did. Saved shots and blocks are left out,
because a car changed the ball's path.

    python scripts/compare_physics.py "%USERPROFILE%/Documents/My Games/Rocket League/TAGame/Demos"
    python scripts/compare_physics.py data/raw/ballchasing_c3_gc2 --limit 300 --workers 8
    python scripts/compare_physics.py path/to/game.replay --shot 17 --verbose   # one shot in detail

Writes one row per shot to --out (default data/physics_compare.csv) and prints a summary.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlxg import physics as ph  # noqa: E402
from rlxg import rsim  # noqa: E402
from rlxg.analyze import analyze_replay  # noqa: E402
from rlxg.outcome import BOUNCE_DV, GRAVITY, _classify_bounce  # noqa: E402
from rlxg.parse import load_replay  # noqa: E402
from rlxg.touches import detect_touches  # noqa: E402

CHECK_T = (0.5, 1.0, 2.0)        # s after the touch: compare positions here
END_Y_REF = 4900.0               # attacking-frame y where lateral/vertical error at the goal end is measured
AFTER_END = 1.0                  # s simulated beyond the real end, so the sim can reach its own event
MAX_SIM = 6.0

# Jitter: how far off the replay's post-touch ball state plausibly is (network compression,
# sampling a frame late). These are guesses; the deterministic run doesn't use them.
JIT_POS = 5.0                    # uu, per axis
JIT_VEL = 0.01                   # fraction of speed, per axis
JIT_ANG = 0.3                    # rad/s, per axis

UNTOUCHED_KINDS = {"goal", "post", "bar", "wide", "over", "wall", "short"}


def _flip(team) -> np.ndarray:
    blue = team in (0, "blue")
    return np.array([1.0, 1.0, 1.0]) if blue else np.array([-1.0, -1.0, 1.0])


def _interp_path(t: np.ndarray, p: np.ndarray, tq: float) -> np.ndarray:
    ok = np.isfinite(p).all(axis=1)
    t, p = t[ok], p[ok]
    if len(t) == 0 or tq < t[0] or tq > t[-1]:
        return np.full(3, np.nan)
    return np.array([np.interp(tq, t, p[:, i]) for i in range(3)])


def _at_y(p: np.ndarray, y: float) -> np.ndarray:
    """x, z where the path first reaches y (NaN if it never does)."""
    ok = np.isfinite(p).all(axis=1)
    p = p[ok]
    hit = np.flatnonzero(p[:, 1] >= y)
    if len(hit) == 0:
        return np.full(2, np.nan)
    k = hit[0]
    if k == 0:
        return p[0, [0, 2]]
    a, b = p[k - 1], p[k]
    f = (y - a[1]) / max(b[1] - a[1], 1e-9)
    q = a + (b - a) * f
    return q[[0, 2]]


def _rs_kind(sp: rsim.SimPath, flip: np.ndarray) -> str:
    """What the simulated ball did first, using the same rules as rlxg.outcome uses on the real ball."""
    P = sp.pos * flip
    V = sp.vel * flip
    if sp.scored_y_sign * flip[1] > 0:
        return "goal"
    dt = 1.0 / rsim.TICK_RATE
    for k in range(1, len(P)):
        dv = V[k] - V[k - 1]
        dv[2] -= GRAVITY * dt
        if np.linalg.norm(dv) < BOUNCE_DV:
            continue
        got = _classify_bounce(P[k - 1]) or _classify_bounce(P[k])
        if got:
            return got
    if sp.scored_y_sign:
        return "own_goal"
    return "none"


def _old_goal(pred, i=0) -> bool:
    """Built-in predictor: the whole ball crosses inside the frame."""
    return bool(pred.reaches_goal_plane[i]
                and abs(pred.cross_x[i]) < ph.GOAL_HALF_W - ph.BALL_RADIUS
                and pred.cross_z[i] < ph.GOAL_H - ph.BALL_RADIUS)


def _old_path(pred, i=0) -> tuple[np.ndarray, np.ndarray]:
    t = pred.sample_t.copy()
    p = pred.samples[i].copy()
    if pred.reaches_goal_plane[i]:                       # add the exact crossing point
        t = np.append(t, pred.time_to_goal[i])
        p = np.vstack([p, [pred.cross_x[i], ph.ARENA_Y, pred.cross_z[i]]])
        o = np.argsort(t)
        t, p = t[o], p[o]
    return t, p


def compare_replay(path: str, jitter: int = 16, spin: str = "replay", only_shot: int | None = None,
                   verbose: bool = False) -> list[dict]:
    rep = load_replay(path)
    if rep.team_size != 2:
        return []
    m = analyze_replay(rep)
    shots = m.shots
    if not len(shots):
        return []
    touches = detect_touches(rep)
    tframes = np.array([t.frame for t in touches])
    rows = []
    for idx, s in enumerate(shots.to_dict(orient="records")):
        if only_shot is not None and idx != only_shot:
            continue
        kind = "goal" if s["goal"] else s["end_kind"]
        if kind not in UNTOUCHED_KINDS or (isinstance(s.get("end_by"), str) and s["end_by"]):
            continue
        f, pf = int(s["frame"]), int(s["post_frame"])
        ef = int(s["goal_frame"]) if s["goal"] else int(s["end_frame"])
        if ef <= pf or np.any((tframes > f) & (tframes <= ef)):
            continue                                    # touched again on the way (deflection, tap-in)
        flip = _flip(s["team"])
        t0 = rep.time[pf]
        te = float(rep.time[ef] - t0)
        pos, vel = rep.ball_pos[pf], rep.ball_vel[pf]
        ang = rep.ball_ang_vel[pf] if rep.ball_ang_vel is not None else np.zeros(3)
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            continue
        ang = np.nan_to_num(ang)
        if spin == "none":
            ang = np.zeros(3)
        elif spin == "neg":
            ang = -ang

        real_t = rep.time[pf:ef + 1] - t0
        real_p = rep.ball_pos[pf:ef + 1] * flip

        pred = ph.predict((pos * flip)[None], (vel * flip)[None], horizon=5.0)
        old_t, old_p = _old_path(pred)
        horizon = min(te + AFTER_END, MAX_SIM)
        sp = rsim.simulate(pos, vel, ang, horizon)
        rs_t, rs_p = sp.t, sp.pos * flip
        rs_kind = _rs_kind(sp, flip)

        n_goal = 0
        if jitter:
            rng = np.random.default_rng(zlib.crc32(f"{rep.replay_id}:{idx}".encode()))
            spd = np.linalg.norm(vel)
            for _ in range(jitter):
                jp = pos + rng.normal(0, JIT_POS, 3)
                jv = vel + rng.normal(0, JIT_VEL * spd, 3)
                ja = ang + rng.normal(0, JIT_ANG, 3)
                n_goal += _rs_kind(rsim.simulate(jp, jv, ja, horizon), flip) == "goal"

        row = dict(replay=Path(path).name, replay_id=rep.replay_id, shot=idx, clock=s.get("clock"),
                   player=s["player"], team=s["team"], xg=s["xg"], xgot=s["xgot"],
                   real_kind=kind, real_goal=bool(s["goal"]), real_t_end=te,
                   spin=float(np.linalg.norm(ang)), speed=float(np.linalg.norm(vel)),
                   bounced=bool(pred.bounces[0] > 0 or np.nanmin(real_p[:, 2]) < ph.BALL_RADIUS + 30),
                   old_goal=_old_goal(pred), old_cross_x=pred.cross_x[0], old_cross_z=pred.cross_z[0],
                   rs_kind=rs_kind, rs_goal=rs_kind == "goal",
                   rs_p_goal=(n_goal / jitter) if jitter else float(rs_kind == "goal"))
        for tq in CHECK_T:
            key = f"{tq:g}s"
            if tq > te:
                row[f"err_old_{key}"] = row[f"err_rs_{key}"] = np.nan
                continue
            r = _interp_path(real_t, real_p, tq)
            row[f"err_old_{key}"] = float(np.linalg.norm(_interp_path(old_t, old_p, tq) - r))
            row[f"err_rs_{key}"] = float(np.linalg.norm(_interp_path(rs_t, rs_p, tq) - r))
        end_y = real_p[-1, 1]
        yref = min(end_y, END_Y_REF)
        if end_y > 4500 and real_p[0, 1] < yref - 300:  # ended at the target end: error across the goal face
            r = _at_y(real_p, yref)
            o = _at_y(old_p[old_t <= MAX_SIM], yref)
            q = _at_y(rs_p, yref)
            row.update(face_dx_old=abs(o[0] - r[0]), face_dz_old=abs(o[1] - r[1]),
                       face_dx_rs=abs(q[0] - r[0]), face_dz_rs=abs(q[1] - r[1]))
        rows.append(row)

        if verbose:
            print(f"\nshot {idx}: {s['player']} at {s.get('clock')}  real={kind}  old_goal={row['old_goal']}  "
                  f"rocketsim={rs_kind}  P(goal, jitter)={row['rs_p_goal']:.2f}  xGOT={s['xgot']:.3f}")
            print(f"  spin at touch {ang.round(2)} rad/s   speed {np.linalg.norm(vel):.0f} uu/s")
            print("     t   real (attacking frame)      built-in sim               rocketsim")
            for tq in np.arange(0, te + 1e-9, 0.2).tolist() + [te]:
                fmt = lambda v: "      -              " if not np.isfinite(v).all() else \
                    f"{v[0]:6.0f} {v[1]:6.0f} {v[2]:5.0f}  "
                print(f"  {tq:4.2f}  {fmt(_interp_path(real_t, real_p, tq))}  "
                      f"{fmt(_interp_path(old_t, old_p, tq))}  {fmt(_interp_path(rs_t, rs_p, tq))}")
    return rows


def _worker(args):
    path, jitter, spin = args
    try:
        return path, compare_replay(path, jitter=jitter, spin=spin), None
    except Exception as e:  # noqa: BLE001  one bad replay shouldn't stop the run
        return path, [], f"{type(e).__name__}: {e}"


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarise(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n=== {n} untouched shots from {df.replay.nunique()} replays ===")
    print("real outcome:", df.real_kind.value_counts().to_dict())

    print("\n-- Goal or not (deterministic run) --")
    for name, col in (("built-in", "old_goal"), ("RocketSim", "rs_goal")):
        acc = (df[col] == df.real_goal).mean()
        fp = (df[col] & ~df.real_goal).sum()
        fn = (~df[col] & df.real_goal).sum()
        print(f"  {name:10s} accuracy {acc:6.1%}   said goal but missed: {fp:4d}   said miss but scored: {fn:4d}")

    print("\n-- Did it say what the ball would hit? (RocketSim; real outcome rows) --")
    print(pd.crosstab(df.real_kind, df.rs_kind, margins=True).to_string())
    fr = df[df.real_kind.isin(["post", "bar"])]
    if len(fr):
        print(f"  post/bar shots: RocketSim also hit the frame on {fr.rs_kind.isin(['post', 'bar']).mean():.0%}, "
              f"said goal on {fr.rs_goal.mean():.0%}; built-in said goal on {fr.old_goal.mean():.0%}")

    print("\n-- Path error vs the real ball (median / 90th percentile, uu) --")
    for key in [f"{t:g}s" for t in CHECK_T]:
        a, b = df[f"err_old_{key}"].dropna(), df[f"err_rs_{key}"].dropna()
        if len(a):
            print(f"  at {key:>4s} (n={len(a):4d})  built-in {a.median():6.0f} / {a.quantile(.9):6.0f}   "
                  f"RocketSim {b.median():6.0f} / {b.quantile(.9):6.0f}")
    if "face_dx_old" in df:
        print("  where it arrived at the goal end:")
        for label, sub in (("bounced", df[df.bounced]), ("in the air", df[~df.bounced])):
            for ax in ("dx", "dz"):
                a, b = sub[f"face_{ax}_old"].dropna(), sub[f"face_{ax}_rs"].dropna()
                if not len(a):
                    continue
                name = f"{label}, {'across' if ax == 'dx' else 'height'}"
                print(f"    {name:20s} (n={len(a):4d})  built-in {a.median():6.0f} / {a.quantile(.9):6.0f}   "
                      f"RocketSim {b.median():6.0f} / {b.quantile(.9):6.0f}")

    print("\n-- As a probability of scoring (lower is better) --")
    y = df.real_goal
    for name, p in (("model xGOT", df.xgot), ("RocketSim P(goal)", df.rs_p_goal),
                    ("built-in yes/no", df.old_goal.astype(float))):
        print(f"  {name:18s} Brier {_brier(p, y):.4f}   log loss {_logloss(p, y):.4f}")
    print("  (xGOT also allows for defenders, who by construction didn't get there on these shots)")

    df = df.assign(model_err=(df.xgot - df.real_goal.astype(float)).abs(),
                   rs_err=(df.rs_p_goal - df.real_goal.astype(float)).abs())
    worst = df[df.model_err > 0.4].sort_values("model_err", ascending=False).head(15)
    if len(worst):
        print("\n-- Shots the model got most wrong, and what RocketSim said --")
        print(worst[["replay", "shot", "clock", "player", "real_kind", "xgot", "rs_p_goal", "rs_kind",
                     "old_goal"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="replay files or folders of replays")
    ap.add_argument("--limit", type=int, help="random sample of this many replays")
    ap.add_argument("--jitter", type=int, default=16, help="perturbed RocketSim runs per shot (0 = off)")
    ap.add_argument("--spin", choices=["replay", "none", "neg"], default="replay",
                    help="ball spin given to RocketSim: from the replay, zero, or negated (sign check)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shot", type=int, help="only this shot index (with a single replay)")
    ap.add_argument("--verbose", action="store_true", help="print each shot's path side by side")
    ap.add_argument("--out", default="data/physics_compare.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # player names aren't always cp1252

    files: list[str] = []
    for p in map(Path, args.paths):
        files += sorted(map(str, p.rglob("*.replay"))) if p.is_dir() else [str(p)]
    if args.limit and len(files) > args.limit:
        files = random.Random(0).sample(files, args.limit)
    print(f"{len(files)} replays, RocketSim meshes at {rsim.meshes_dir()}")

    t0 = time.time()
    rows: list[dict] = []
    if args.verbose or args.workers <= 1 or len(files) == 1:
        for f in files:
            rows += compare_replay(f, jitter=args.jitter, spin=args.spin, only_shot=args.shot,
                                   verbose=args.verbose)
    else:
        errors = 0
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_worker, (f, args.jitter, args.spin)) for f in files]
            for i, fut in enumerate(as_completed(futs), 1):
                path, r, err = fut.result()
                rows += r
                if err:
                    errors += 1
                    if "meshes not found" in err:
                        sys.exit(err)
                if i % 25 == 0 or i == len(files):
                    print(f"  {i}/{len(files)} replays, {len(rows)} shots, {errors} errors, "
                          f"{time.time() - t0:.0f}s", flush=True)
    if not rows:
        print("no untouched shots found")
        return
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    summarise(df)


if __name__ == "__main__":
    main()
