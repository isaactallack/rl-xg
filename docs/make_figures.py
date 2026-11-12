"""Regenerate the explanatory figures in docs/img/ used by the README.

    pip install matplotlib
    python docs/make_figures.py --shots data/shots_v4.csv --replay path/to/a_2v2_game.replay

Needs the trained model (models/xg_model.joblib), a shots table built with
scripts/build_dataset.py, and one 2v2 replay for the touch -> shot walkthrough.
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
        ("Features", "~50 geometric features,\nbefore and after the touch", ORANGE),
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

    # pick a goal with a busy build-up (several touches in the 6 s before it)
    best, best_n = None, -1
    for _, s in shots[shots.goal == 1].iterrows():
        n = int(((t_arr > s.time - 6) & (t_arr <= s.time)).sum())
        if 3 <= n and n > best_n:
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
    d = df[df.source == "Demos"].copy()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=str(ROOT / "data" / "shots.csv"))
    ap.add_argument("--replay", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = get_model()
    fig_pipeline()
    fig_touch_to_shot(a.replay)
    fig_importance(model)
    df = pd.read_csv(a.shots, low_memory=False)
    fig_goal_face(df)
    fig_conversion_map(df)
    fig_holdout(df, model)


if __name__ == "__main__":
    main()
