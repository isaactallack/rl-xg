"""Regenerate the explanatory figures in docs/img/ used by the README.

    pip install matplotlib
    python docs/make_figures.py --shots data/shots.csv --replay path/to/a_2v2_game.replay
    python docs/make_figures.py --replay x --only intercept coverage jitter     # just some

Needs the trained model (models/xg_model.joblib), a shots table built with
scripts/build_dataset.py, one 2v2 replay for the touch -> shot walkthrough, the Kaggle
replays (data/raw/hlrlrd_2v2) to find examples for the feature explainers, and
data/learning_curve.json from scripts/learning_curve.py for the learning curves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from rlxg import physics as ph
from rlxg import rsim
from rlxg.analyze import get_model
from rlxg.parse import load_replay
from rlxg.shots import SHOT_X_MARGIN, SHOT_Z_MARGIN, candidate_touches, extract_shots
from rlxg.touches import detect_touches

OUT = ROOT / "docs" / "img"
BG, PANEL, GRID, FG, MUTED = "#0d1117", "#161b22", "#2a313c", "#e6edf3", "#8b949e"
BLUE, ORANGE, GREEN, RED, PURPLE = "#3b82f6", "#f97316", "#22c55e", "#f43f5e", "#a78bfa"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": FG,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": FG, "axes.titlelocation": "left",
    "axes.titlepad": 10, "legend.frameon": False, "legend.labelcolor": FG,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def pitch(ax, y_min=-ph.ARENA_Y, alpha=1.0):
    """Top-down arena outline (x mirrored so it matches the in-game view from blue's side)."""
    X, Y, c = ph.ARENA_X, ph.ARENA_Y, 1152
    pts = [(-X, -Y + c), (-X + c, -Y), (X - c, -Y), (X, -Y + c), (X, Y - c), (X - c, Y), (-X + c, Y), (-X, Y - c)]
    ax.add_patch(Polygon(pts, closed=True, fc="#111826", ec=GRID, lw=1.5, alpha=alpha, zorder=0))
    for sy, col in ((1, ORANGE), (-1, BLUE)):
        ax.add_patch(Rectangle((-ph.GOAL_HALF_W, sy * Y if sy > 0 else -Y - 880), 2 * ph.GOAL_HALF_W, 880,
                               fc=col, alpha=0.18, ec=col, lw=1.2, zorder=0))
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.add_patch(plt.Circle((0, 0), 1000, fc="none", ec=GRID, lw=1, zorder=0))
    ax.set_xlim(X + 300, -X - 300)
    ax.set_ylim(y_min, Y + 1100)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- 0
def fig_pipeline():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    steps = [
        ("Replays", "Kaggle SSL games, ballchasing\nC3–GC2, original repo", MUTED),
        ("Parse", "subtr-actor → per-frame\nball and car physics", MUTED),
        ("Touches", "ball velocity jumps with a\ncar hitbox in contact range", BLUE),
        ("Shots", "last touch of a carry whose\nsimulated path reaches goal", BLUE),
        ("Label", "goal if scored before the\nteam touches again (7 s)", ORANGE),
        ("Features", "~190 features: ball, path,\ndefender timing, game state", ORANGE),
        ("Models", "LightGBM xG + xGOT,\ncalibrated per rank band", GREEN),
        ("Use", "CLI, Python API and\nlive dashboard", GREEN),
    ]
    W, H, GX, GY = 3.0, 1.55, 0.75, 1.1
    fig, ax = plt.subplots(figsize=(14.5, 5.0))
    ax.set_xlim(-0.2, 4 * W + 3 * GX + 0.2)
    ax.set_ylim(-0.35, 2 * H + GY + 0.5)
    ax.axis("off")
    pos = []
    for i, (title, body, col) in enumerate(steps):
        row, k = divmod(i, 4)
        x = k * (W + GX)
        y = (H + GY) if row == 0 else 0
        pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc=PANEL, ec=col, lw=1.8))
        ax.add_patch(plt.Circle((x + 0.42, y + H - 0.4), 0.24, fc=col, ec="none"))
        ax.text(x + 0.42, y + H - 0.4, str(i + 1), ha="center", va="center", color=BG, fontsize=12, fontweight="bold")
        ax.text(x + 0.8, y + H - 0.4, title, ha="left", va="center", color=FG, fontsize=14, fontweight="bold")
        ax.text(x + 0.3, y + 0.5, body, ha="left", va="center", color=MUTED, fontsize=10.5, linespacing=1.45)
    arrow = dict(arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.6)
    for i in range(7):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if i == 3:   # wrap from the end of row 1 to the start of row 2 with an elbow
            ym = H + GY / 2
            ax.plot([x0 + W / 2, x0 + W / 2, x1 + W / 2], [y0, ym, ym], color=MUTED, lw=1.6, solid_capstyle="round")
            ax.add_patch(FancyArrowPatch((x1 + W / 2, ym), (x1 + W / 2, y1 + H), **arrow, shrinkA=0, shrinkB=3))
        else:
            ax.add_patch(FancyArrowPatch((x0 + W, y0 + H / 2), (x1, y1 + H / 2), **arrow, shrinkA=3, shrinkB=3))
    save(fig, "fig_pipeline.png")


# --------------------------------------------------------------------------- 1
def fig_touch_to_shot(replay_path: str):
    rep = load_replay(replay_path)
    touches = detect_touches(rep)
    shots = extract_shots(rep, touches)
    cands = set(candidate_touches(touches))
    t_arr = np.array([t.time for t in touches])

    # pick a goal with a busy build-up (several touches in the 6 s before it), struck from far
    # enough out that the predicted path is visible
    best, best_n = None, -1
    for _, s in shots[shots.goal == 1].iterrows():
        n = int(((t_arr > s.time - 6) & (t_arr <= s.time)).sum()) + (100 if s.dist_goal > 2000 else 0)
        if 3 <= n % 100 and n > best_n:
            best, best_n = s, n
    s = best if best is not None else shots.iloc[0]
    ti = int(np.flatnonzero(t_arr == s.time)[0])
    shot_touch = touches[ti]
    gt = next((g for g in rep.goals if g.frame >= shot_touch.frame), None)
    t0 = s.time - 6
    t1 = rep.time[gt.frame] + 0.15 if gt is not None else s.time + 2
    fr = (rep.time >= t0) & (rep.time <= t1)
    bp, bv = rep.ball_pos[fr], rep.ball_vel[fr]
    tt = rep.time[fr]

    flip = np.array([1, 1, 1.0]) if shot_touch.team == 0 else np.array([-1, -1, 1.0])
    pf = min(shot_touch.post_frame, rep.n_frames - 1)
    pred = rsim.predict((rep.ball_pos[pf] * flip)[None], (rep.ball_vel[pf] * flip)[None],
                        (np.nan_to_num(rep.ball_ang_vel[pf]) * flip)[None], 5.0)
    path = pred.samples[0] * flip
    path = path[np.isfinite(path).all(1)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1, 1.45]))
    pitch(a1)
    a1.plot(bp[:, 0], bp[:, 1], color=MUTED, lw=1.6, alpha=0.8, zorder=2, label="actual ball path")
    a1.plot(path[:, 0], path[:, 1], color=GREEN, lw=2.2, ls=(0, (4, 3)), zorder=3, label="predicted free flight")
    win = [(i, t) for i, t in enumerate(touches) if t0 <= t.time <= t1]
    for i, t in win:
        p = rep.ball_pos[t.frame]
        col = BLUE if t.team == 0 else ORANGE
        if i == ti:
            a1.scatter(p[0], p[1], s=420, marker="*", c=col, ec="white", lw=1.2, zorder=5)
        else:
            a1.scatter(p[0], p[1], s=70 if i in cands else 40, c=col if i in cands else PANEL, ec=col, lw=1.6, zorder=4)
    a1.set_title("Ball path in the 6 s before a goal")
    a1.legend(loc="upper center", fontsize=9.5, ncol=2, bbox_to_anchor=(0.5, 0.0))

    speed = np.linalg.norm(bv, axis=1) * 0.036  # uu/s -> km/h
    a2.fill_between(tt - s.time, speed, color=BLUE, alpha=0.08)
    a2.plot(tt - s.time, speed, color=FG, lw=1.6)
    for i, t in win:
        col = BLUE if t.team == 0 else ORANGE
        x = t.time - s.time
        a2.axvline(x, color=col, lw=1, alpha=0.35, ls="-" if i in cands else ":")
        y = np.interp(t.time, tt, speed)
        if i == ti:
            a2.scatter(x, y, s=420, marker="*", c=col, ec="white", lw=1.2, zorder=5)
        else:
            a2.scatter(x, y, s=70 if i in cands else 40, c=col if i in cands else PANEL, ec=col, lw=1.6, zorder=4)
    if gt is not None:
        a2.axvline(rep.time[gt.frame] - s.time, color=GREEN, lw=2)
        a2.text(rep.time[gt.frame] - s.time, np.nanmax(speed) * 1.04, " GOAL", color=GREEN, fontweight="bold", va="bottom")
    a2.set_xlabel("seconds relative to the shot")
    a2.set_ylabel("ball speed (km/h)")
    a2.set_ylim(0, np.nanmax(speed) * 1.15)
    a2.set_title("Touches are unexplained jumps in ball velocity")
    a2.grid(color=GRID, lw=0.6, alpha=0.6)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=PANEL, mec=MUTED, ms=7, mew=1.6, label="touch inside a carry"),
        Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, ms=8, label="last touch of a carry (candidate)"),
        Line2D([], [], marker="*", ls="", mfc=MUTED, mec="white", ms=16, label="candidate whose predicted path hits the goal = shot"),
    ]
    a2.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_touch_to_shot.png")


# --------------------------------------------------------------------------- 2
def fig_goal_face(df):
    d = df.dropna(subset=["cross_x", "cross_z"]).sample(min(30000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    miss, goal = d[d.goal == 0], d[d.goal == 1]
    ax.scatter(miss.cross_x, miss.cross_z, s=3, c=BLUE, alpha=0.25, lw=0, label="no goal")
    ax.scatter(goal.cross_x, goal.cross_z, s=3, c=ORANGE, alpha=0.35, lw=0, label="goal")
    W, H = ph.GOAL_HALF_W, ph.GOAL_H
    ax.add_patch(Rectangle((-W, 0), 2 * W, H, fc="none", ec="white", lw=2.5, zorder=4))
    ax.add_patch(Rectangle((-W - SHOT_X_MARGIN, 0), 2 * (W + SHOT_X_MARGIN), H + SHOT_Z_MARGIN,
                           fc="none", ec=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=4))
    ax.text(0, H + 30, "goal mouth", ha="center", va="bottom", color="white", fontweight="bold")
    ax.text(-W - SHOT_X_MARGIN + 30, H + SHOT_Z_MARGIN - 30, "shot window (mouth + 500 uu wide, 250 uu over)",
            ha="left", va="top", color=MUTED, fontsize=9.5)
    ax.axhline(0, color=FG, lw=1)
    ax.set_xlim(-W - SHOT_X_MARGIN - 120, W + SHOT_X_MARGIN + 120)
    ax.set_ylim(0, H + SHOT_Z_MARGIN + 80)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted crossing point across the goal line (uu)")
    ax.set_ylabel("height (uu)")
    ax.set_title(f"Where {len(d):,} shots were heading when they left the shooter")
    lg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, markerscale=5, fontsize=10)
    for h in lg.legend_handles:
        h.set_alpha(1)
    save(fig, "fig_goal_face.png")


# --------------------------------------------------------------------------- 3
def fig_conversion_map(df):
    d = df[(df.ball_y > -1500)]
    fig, ax = plt.subplots(figsize=(8, 7.2))
    pitch(ax, y_min=-1500)
    hb = ax.hexbin(d.ball_x, d.ball_y, C=d.goal, reduce_C_function=np.mean, gridsize=(26, 22),
                   extent=(-ph.ARENA_X, ph.ARENA_X, -1500, ph.ARENA_Y), mincnt=60, cmap="magma",
                   vmin=0, vmax=0.6, lw=0.2, edgecolors=PANEL, zorder=1)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("share of shots scored", color=MUTED)
    cb.outline.set_edgecolor(GRID)
    ax.text(0, ph.ARENA_Y + 440, "goal", ha="center", va="center", color=FG, fontweight="bold")
    ax.set_title(f"Conversion by shot location ({len(d):,} shots, attacking up)")
    save(fig, "fig_conversion_map.png")


# --------------------------------------------------------------------------- 4/5
def fig_holdout(df, model):
    d = df[(df.source == "Demos") & (df.team_size == 2)].copy()     # the model covers 2v2 only
    if len(d) < 200:
        print("not enough Demos shots for the holdout figure; skipped")
        return
    d["xg"] = model.predict_chance(d)
    d["xgot"] = model.predict(d)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    m, g = d[d.goal == 0], d[d.goal == 1]
    a1.scatter(m.xg, m.xgot, s=9, c=BLUE, alpha=0.35, lw=0, label="no goal")
    a1.scatter(g.xg, g.xgot, s=11, c=ORANGE, alpha=0.6, lw=0, label="goal")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a1.set_xlim(0, 1), a1.set_ylim(0, 1)
    a1.set_xlabel("chance xG (situation before the touch)")
    a1.set_ylabel("xGOT (where the touch sent the ball)")
    a1.set_title("Two numbers per shot")
    kw = dict(fontsize=9.5, color=FG, bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID))
    a1.text(0.97, 0.04, "big chance,\nfluffed it", ha="right", va="bottom", **kw)
    a1.text(0.03, 0.96, "half-chance,\nstruck perfectly", ha="left", va="top", **kw)
    lg = a1.legend(loc="upper left", markerscale=2.5, bbox_to_anchor=(0.0, 0.85))
    for h in lg.legend_handles:
        h.set_alpha(1)
    a1.grid(color=GRID, lw=0.6, alpha=0.5)

    bins = np.array([0, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    for col, name, c in (("xg", "chance xG", PURPLE), ("xgot", "xGOT", GREEN)):
        b = pd.cut(d[col], bins, right=False)
        t = d.groupby(b, observed=True).agg(p=(col, "mean"), y=("goal", "mean"), n=("goal", "size"))
        t = t[t.n >= 15]
        a2.plot(t.p, t.y, color=c, lw=2, marker="o", ms=0)
        a2.scatter(t.p, t.y, s=np.sqrt(t.n) * 4, color=c, ec=BG, lw=1, zorder=3,
                   label=f"{name}: {d[col].sum():.0f} predicted vs {int(d.goal.sum())} goals")
    a2.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a2.set_xlim(0, 1), a2.set_ylim(0, 1)
    a2.set_xlabel("predicted probability")
    a2.set_ylabel("actual share scored")
    a2.set_title(f"Calibration on {len(d):,} shots the model never saw")
    a2.legend(loc="upper left", fontsize=10)
    a2.grid(color=GRID, lw=0.6, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_xg_vs_xgot.png")


LABELS = {
    "goal_angle": "angle the goal mouth subtends", "dist_goal": "distance to goal",
    "chance_def1_u": "where best defender meets the shot line", "chance_def1_t": "best defender's time to the shot line",
    "chance_def1_d": "best defender's distance to the shot line", "chance_def2_t": "2nd defender's time to the shot line",
    "chance_def_goal_t": "quickest defender to the aim point", "chance_ball_goal_t": "ball's time to goal (straight line)",
    "n_def_goal_side": "defenders goal-side of the ball", "shooter_v_to_ball": "shooter speed towards the ball",
    "gk_goal_side": "goalkeeper goal-side", "gk_vy": "goalkeeper speed along the pitch", "pre_ball_z": "ball height",
    "def1_margin": "best defender's intercept margin", "cross_margin": "how far inside the posts / bar",
    "n_def_can_intercept": "defenders who can intercept", "time_to_goal": "predicted time to goal",
    "on_target": "on target", "def2_margin": "2nd defender's intercept margin", "cross_z": "crossing height",
    "cross_abs_x": "crossing distance from centre", "def1_speed": "best defender's speed", "touch_dv": "strength of the touch",
    "def1_d_path": "best defender's distance to the ball path", "gk_d_goal": "goalkeeper distance from goal",
    "ball_speed": "ball speed", "nearest_def_ball": "nearest defender to the ball", "cross_speed": "speed at the goal line",
    "shooter_facing_goal": "shooter facing goal", "shooter_speed": "shooter speed", "ball_y": "ball distance up the pitch",
    "gk_d_cross": "goalkeeper distance to the crossing point", "pre_ball_speed": "ball speed before the touch",
    "sim_p_goal": "RocketSim: share of runs that go in", "sim_goal": "RocketSim: goes in unaided",
    "sim_frame": "RocketSim: hits post / bar", "ball_spin": "ball spin",
    # kinematic defender timing (defence.py): k* along the predicted path, ck* along the straight shot line
    "k1_margin": "best defender's time margin to the path", "k2_margin": "2nd defender's time margin to the path",
    "k1_reach_frac": "share of the path best defender can reach", "k1_t_int": "when best defender meets the path",
    "k1_up_t": "best defender's jump / aerial time", "k_n_can": "defenders who can reach the path",
    "k1_int_z": "height where best defender meets the path", "k1_closing": "best defender's closing speed",
    "ck1_margin": "best defender's time margin to the shot line", "ck2_margin": "2nd defender's margin to the shot line",
    # goal-mouth coverage (situation.py)
    "cov_fast_mean": "goal coverage, hard shot (mean margin)", "cov_fast_best": "most open spot, hard shot",
    "cov_fast_open": "share of goal open, hard shot", "cov_mean": "goal coverage (mean margin)",
    "cov_best": "most open spot", "cov_open": "share of goal open",
    "shooter_align": "shooter lined up behind the ball", "shooter_v_to_goal": "shooter speed towards goal",
    "press1_t": "time for nearest defender to reach the ball", "seconds_remaining": "seconds left",
    "score_diff": "score difference",
}


def fig_importance(model, top=12):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, booster, feats, title, col in (
        (axes[0], model.chance_booster, model.chance_features, "Chance xG: what matters most", PURPLE),
        (axes[1], model.booster, model.features, "xGOT: what matters most", GREEN),
    ):
        gain = booster.booster_.feature_importance("gain")
        s = pd.Series(gain / gain.sum() * 100, index=feats).sort_values().tail(top)
        s.index = [LABELS.get(f, f) for f in s.index]
        ax.barh(s.index, s.values, color=col, alpha=0.9, height=0.65)
        for i, v in enumerate(s.values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, color=MUTED)
        ax.set_title(title)
        ax.set_xlabel("share of total split gain (%)")
        ax.tick_params(axis="y", labelcolor=FG, labelsize=10)
        ax.set_xlim(0, s.max() * 1.18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_importance.png")


# --------------------------------------------------------------------------- 6/7/8: how the harder features work
def _flip(team):
    return np.array([1.0, 1.0, 1.0]) if team == 0 else np.array([-1.0, -1.0, 1.0])


def _opp_state(rep, f, team, flip):
    """Defender states as shots.py builds them (attacking frame), plus a 0.5 s trail of positions."""
    from rlxg.touches import quat_to_matrix
    g = int(np.clip(np.searchsorted(rep.time, rep.time[f] - 0.5), 0, f))
    out = []
    for p in rep.players:
        if p.team == team or not np.isfinite(p.pos[f]).all():
            continue
        q = p.quat[f]
        rot = quat_to_matrix(q[None])[0] if np.isfinite(q).all() else np.full((3, 3), np.nan)
        out.append(dict(pos=p.pos[f] * flip, vel=p.vel[f] * flip, fwd=rot[:, 0] * flip, up=rot[:, 2] * flip,
                        boost=float(p.boost[f]) if np.isfinite(p.boost[f]) else 33.0,
                        jumped=bool(p.jump[f] or p.double_jump[f] or p.dodge[f]),
                        trail=p.pos[g:f + 1] * flip))
    return out


def find_examples(folder: str, max_replays: int = 200) -> dict:
    """Search high-level replays for one clear example shot of each feature (deterministic)."""
    want = {
        # best defender just makes it, the second one doesn't, both on the ground
        "intercept": lambda s: (-0.5 < s.k1_margin < -0.1) and (0.2 < s.k2_margin < 1.5) and s.k1_on_ground == 1
                     and s.k2_on_ground == 1 and s.n_def == 2 and 2500 < s.dist_goal < 5000
                     and s.def1_d_ball > 900 and s.ball_abs_x < 2000,
        # part of the goal covered by a defender in the way, nobody pressuring the ball
        "coverage": lambda s: 0.3 <= s.cov_open <= 0.7 and s.n_def == 2 and 2000 < s.dist_goal < 4500
                    and s.press1_t > 0.5 and s.ball_abs_x < 2000 and s.gk_goal_side == 1 and s.gk_d_goal < 1500,
        "jitter": lambda s: 0.35 <= s.sim_p_goal <= 0.65 and s.sim_kind in ("post", "goal", "bar"),
    }
    found = {}
    for path in sorted(Path(folder).glob("*.replay"))[:max_replays]:
        try:
            rep = load_replay(path)
            shots = extract_shots(rep, detect_touches(rep))
        except Exception:
            continue
        for key, ok in want.items():
            if key in found or not len(shots):
                continue
            hit = shots[shots.apply(ok, axis=1)]
            if len(hit):
                found[key] = (rep, hit.iloc[0])
        if len(found) == len(want):
            break
    print("examples:", {k: Path(v[0].path).stem[:8] for k, v in found.items()})
    return found


def _zoom(ax, pts, pad=700):
    """Zoom a pitch() axis onto these points (x stays mirrored, the goal stays in view)."""
    pts = np.asarray(pts)[:, :2]
    x0, x1 = max(pts[:, 0].min() - pad, -ph.ARENA_X - 300), min(pts[:, 0].max() + pad, ph.ARENA_X + 300)
    ax.set_xlim(x1, x0)
    ax.set_ylim(max(pts[:, 1].min() - pad, -ph.ARENA_Y), ph.ARENA_Y + 1100)


def _shot_path(rep, s):
    """RocketSim path of a shot, attacking frame, sampled every 0.1 s (as the features use it)."""
    flip = _flip(s.team)
    pf = int(s.post_frame)
    pred = rsim.predict((rep.ball_pos[pf] * flip)[None], (rep.ball_vel[pf] * flip)[None],
                        (np.nan_to_num(rep.ball_ang_vel[pf]) * flip)[None], 5.0)
    P, t = pred.samples[0], pred.sample_t
    ok = np.isfinite(P).all(1)
    return P[ok], t[ok]


def fig_intercept(rep, s):
    from rlxg import defence
    flip = _flip(s.team)
    f = int(s.frame)
    P, t = _shot_path(rep, s)
    opp = sorted(_opp_state(rep, f, s.team, flip),
                 key=lambda d: (defence.need_time(**d, path=P) - t).min())
    cols = (RED, PURPLE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[1, 1.2]))
    pitch(a1)
    _zoom(a1, np.vstack([P, *(d["trail"] for d in opp), *(d["pos"][None] for d in opp)]))
    sc = a1.scatter(P[:, 0], P[:, 1], c=t, cmap="viridis", s=26, zorder=3, ec="none")
    a1.scatter(*rep.ball_pos[f][:2] * flip[:2], s=140, c=FG, ec=BG, zorder=5)
    a1.text(*(rep.ball_pos[f][:2] * flip[:2] + [0, -260]), "ball at the touch", ha="center", color=FG, fontsize=9)
    for i, (d, c) in enumerate(zip(opp, cols)):
        need = defence.need_time(**d, path=P)
        k = int(np.argmin(need - t))
        a1.plot(d["trail"][:, 0], d["trail"][:, 1], color=c, lw=2, alpha=0.5, zorder=3)
        a1.scatter(*d["pos"][:2], s=120, marker="s", c=c, ec=BG, zorder=5)
        a1.annotate("", xy=d["pos"][:2] + d["vel"][:2] * 0.4, xytext=d["pos"][:2],
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2), zorder=5)
        a1.plot([d["pos"][0], P[k, 0]], [d["pos"][1], P[k, 1]], color=c, lw=1.2, ls=":", zorder=4)
        a1.scatter(P[k, 0], P[k, 1], s=160, facecolors="none", edgecolors=c, lw=2, zorder=6)
        a1.text(d["pos"][0], d["pos"][1] - 330, f"defender {i + 1}", ha="center", color=c, fontsize=9.5,
                fontweight="bold")
        a2.plot(t, need, color=c, lw=2.2,
                label=f"defender {i + 1}: time to reach each point (margin {need[k] - t[k]:+.2f} s)")
        a2.fill_between(t, need, t, where=need <= t, color=c, alpha=0.25, lw=0, interpolate=True)
        a2.plot([t[k], t[k]], [t[k], need[k]], color=c, lw=3, solid_capstyle="butt")
    a2.plot(t, t, color=FG, lw=2, ls="--", label="ball arrives there")
    cb = fig.colorbar(sc, ax=a1, fraction=0.04, pad=0.02)
    cb.set_label("seconds after the touch", color=MUTED)
    cb.outline.set_edgecolor(GRID)
    a1.set_title("Predicted ball path and each defender")
    a1.text(0.02, 0.02, "trail: last 0.5 s · arrow: velocity\ncircle: their best point on the path",
            transform=a1.transAxes, color=MUTED, fontsize=9, va="bottom")
    a2.set_xlabel("time the ball reaches a point on its path (s)")
    a2.set_ylabel("time (s)")
    a2.set_ylim(0, min(max(t.max() + 1.0, 2.0), 5.0))
    a2.set_xlim(0, t.max())
    a2.set_title("Shaded = the defender gets there before the ball")
    a2.text(0.98, 0.03, "margin = time needed − ball's arrival, at their best point (thick bar)",
            transform=a2.transAxes, ha="right", color=MUTED, fontsize=9)
    a2.legend(loc="upper left", fontsize=9.5)
    a2.grid(color=GRID, lw=0.6, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_intercept.png")


def fig_coverage(rep, s):
    from matplotlib.colors import TwoSlopeNorm
    from rlxg import defence, situation
    flip = _flip(s.team)
    f = int(s.frame)
    pb = rep.ball_pos[f] * flip
    speed = max(float(np.linalg.norm(rep.ball_vel[f])), 1500.0)
    opp = _opp_state(rep, f, s.team, flip)
    fr = np.linspace(0.0, 1.0, defence.PATH_POINTS)
    best = np.full(len(situation.AIMS), defence.T_CAP)
    for j, aim in enumerate(situation.AIMS):
        line = pb + fr[:, None] * (aim - pb)
        tt = fr * np.linalg.norm(aim - pb) / speed
        best[j] = min(((defence.need_time(**d, path=line) - tt).min() for d in opp), default=defence.T_CAP)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    W, H = ph.GOAL_HALF_W, ph.GOAL_H
    xs, zs = np.unique(situation.AIMS[:, 0]), np.unique(situation.AIMS[:, 2])
    dx, dz = 2 * W / len(xs), H / len(zs)
    for aim, b in zip(situation.AIMS, best):
        i, j = int(np.flatnonzero(xs == aim[0])[0]), int(np.flatnonzero(zs == aim[2])[0])
        a1.add_patch(Rectangle((-W + i * dx + dx * 0.03, j * dz + dz * 0.03), dx * 0.94, dz * 0.94,
                               fc=cmap(norm(np.clip(b, -1, 1))), ec="none", zorder=2))
        a1.text(-W + (i + 0.5) * dx, (j + 0.5) * dz, f"{b:+.2f}", ha="center", va="center",
                fontsize=9.5, color=BG, fontweight="bold", zorder=3)
    a1.add_patch(Rectangle((-W, 0), 2 * W, H, fc="none", ec="white", lw=2.5, zorder=4))
    a1.set_xlim(W + 100, -W - 100), a1.set_ylim(-30, H + 60)      # mirrored like the top-down view
    a1.set_aspect("equal")
    a1.set_xticks([]), a1.set_yticks([])
    a1.set_title(f"Goal mouth seen from the shooter: {np.mean(best >= 0):.0%} open")
    a1.text(0, -25, "best defender's margin (s) on the straight line to each spot: + = open, − = covered",
            ha="center", va="top", color=MUTED, fontsize=9)
    pitch(a2)
    _zoom(a2, np.vstack([pb[None], situation.AIMS, *(d["trail"] for d in opp)]), pad=500)
    for aim, b in zip(situation.AIMS, best):
        a2.plot([pb[0], aim[0]], [pb[1], aim[1]], color=cmap(norm(np.clip(b, -1, 1))), lw=1.4, alpha=0.8, zorder=2)
    a2.scatter(pb[0], pb[1], s=140, c=FG, ec=BG, zorder=5)
    for d in opp:
        a2.plot(d["trail"][:, 0], d["trail"][:, 1], color=RED, lw=2, alpha=0.5, zorder=3)
        a2.scatter(*d["pos"][:2], s=120, marker="s", c=RED, ec=BG, zorder=5)
        a2.annotate("", xy=d["pos"][:2] + d["vel"][:2] * 0.4, xytext=d["pos"][:2],
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2), zorder=5)
    a2.set_title("Lines to the 7 × 3 grid, before the touch\n(from above, the 3 heights overlap)")
    fig.tight_layout()
    save(fig, "fig_coverage.png")


def fig_jitter(rep, s):
    import zlib
    flip = _flip(s.team)
    pf = int(s.post_frame)
    pos, vel = rep.ball_pos[pf] * flip, rep.ball_vel[pf] * flip
    ang = np.nan_to_num(rep.ball_ang_vel[pf] * flip)
    rng = np.random.default_rng(zlib.crc32(f"{rep.replay_id}:{int(s.frame)}".encode()))  # same draws as p_goal
    spd = float(np.linalg.norm(vel))
    runs = []
    for _ in range(rsim.JITTER_RUNS):
        P, V = rsim.run(pos + rng.normal(0, rsim.JIT_POS, 3), vel + rng.normal(0, rsim.JIT_VEL * spd, 3),
                        ang + rng.normal(0, rsim.JIT_ANG, 3), int(5.0 * rsim.TICK_RATE))
        kind, k = rsim.first_event(P, V)
        runs.append((P[:k + 1], kind))
    n_goal = sum(k == "goal" for _, k in runs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    W, H = ph.GOAL_HALF_W, ph.GOAL_H
    for P, kind in runs:
        c = ORANGE if kind == "goal" else BLUE
        a1.plot(P[:, 0], P[:, 1], color=c, lw=1.2, alpha=0.75)
        end = P[np.flatnonzero(P[:, 1] >= rsim.CROSS_Y)[0]] if (P[:, 1] >= rsim.CROSS_Y).any() else P[-1]
        a2.scatter(end[0], end[2], s=70, c=c, ec=BG, lw=1, zorder=4)
    pitch(a1)
    ends = np.array([P[-1] for P, _ in runs])
    # the last 1.2 s before the goal: where the copies split
    a1.set_xlim(min(ends[:, 0].max() + 1400, ph.ARENA_X + 300), max(ends[:, 0].min() - 1400, -ph.ARENA_X - 300))
    a1.set_ylim(ph.ARENA_Y - 2200, ph.ARENA_Y + 1000)
    a1.set_title(f"{rsim.JITTER_RUNS} slightly different copies of one shot (end of their paths)")
    a2.add_patch(Rectangle((-W, 0), 2 * W, H, fc="none", ec="white", lw=2.5, zorder=3))
    a2.axhline(0, color=FG, lw=1)
    a2.set_xlim(W + 600, -W - 600), a2.set_ylim(0, H + 400)      # mirrored like the top-down view
    a2.set_aspect("equal")
    a2.set_xlabel("across the goal line (uu)"), a2.set_ylabel("height (uu)")
    a2.set_title(f"Where each one arrives: {n_goal} of {rsim.JITTER_RUNS} go in → sim_p_goal = {n_goal / rsim.JITTER_RUNS:.2f}")
    a2.legend(handles=[Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=BG, ms=9, label="goal"),
                       Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BG, ms=9, label="post / bar / wide")],
              loc="upper right", fontsize=10)
    a2.text(0.02, 0.97, f"jitter: ±{rsim.JIT_POS:.0f} uu position, ±{rsim.JIT_VEL:.0%} speed,\n"
            f"±{rsim.JIT_ANG} rad/s spin per axis", transform=a2.transAxes, va="top", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_jitter.png")


def fig_learning_curve(path):
    import json
    runs = pd.DataFrame(json.loads(Path(path).read_text())["runs"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for model, c in (("xGOT", GREEN), ("chance xG", PURPLE)):
        r = runs[runs.model == model].sort_values("replays")
        a1.plot(r.replays, r.test_ll, color=c, lw=2.2, marker="o", label=f"{model}: high-level test")
        a1.plot(r.replays, r.c3_ll, color=c, lw=1.6, ls="--", marker="s", ms=5, label=f"{model}: C3–GC2")
        last = r.iloc[-1]
        a2.plot(np.arange(1, len(last.curve_test) + 1), last.curve_test, color=c, lw=2.2, label=f"{model}: test")
        a2.plot(np.arange(1, len(last.curve_train) + 1), last.curve_train, color=c, lw=1.4, ls=":",
                label=f"{model}: training")
        a2.axvline(last.best_iter, color=c, lw=1, alpha=0.4)
    a1.set_xscale("log")
    a1.set_xlabel("high-level replays used for training (log scale)")
    a1.set_ylabel("log loss on games not trained on (lower is better)")
    a1.set_title("Does more data still help?")
    a1.legend(fontsize=9, ncol=2, loc="upper right")
    a1.grid(color=GRID, lw=0.6, alpha=0.5)
    a2.set_xlabel("trees")
    a2.set_ylabel("log loss")
    a2.set_title(f"Training curve, all {int(runs.replays.max()):,} replays")
    a2.legend(fontsize=9, loc="upper right")
    a2.grid(color=GRID, lw=0.6, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_learning_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=str(ROOT / "data" / "shots.csv"))
    ap.add_argument("--replay", required=True)
    ap.add_argument("--examples", default=str(ROOT / "data" / "raw" / "hlrlrd_2v2"),
                    help="folder of replays to search for the feature explainer examples")
    ap.add_argument("--learning-curve", default=str(ROOT / "data" / "learning_curve.json"))
    ap.add_argument("--only", nargs="*", help="only these figures (e.g. intercept coverage jitter learning)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(a.only or [])
    run = lambda name: not only or name in only
    if run("pipeline"):
        fig_pipeline()
    if run("touch"):
        fig_touch_to_shot(a.replay)
    if only & {"intercept", "coverage", "jitter"} or not only:
        ex = find_examples(a.examples)
        for name, fn in (("intercept", fig_intercept), ("coverage", fig_coverage), ("jitter", fig_jitter)):
            if run(name) and name in ex:
                fn(*ex[name])
    if run("learning") and Path(a.learning_curve).exists():
        fig_learning_curve(a.learning_curve)
    if only and not only & {"importance", "goal_face", "conversion", "holdout"}:
        return
    model = get_model()
    if run("importance"):
        fig_importance(model)
    df = pd.read_csv(a.shots, low_memory=False)
    if run("goal_face"):
        fig_goal_face(df)
    if run("conversion"):
        fig_conversion_map(df)
    if run("holdout"):
        fig_holdout(df, model)


if __name__ == "__main__":
    main()
