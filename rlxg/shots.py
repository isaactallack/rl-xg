"""Shot detection, outcome labelling and feature extraction.

Definitions
-----------
* Candidate touch: the *last* touch of a run of consecutive touches by the same
  player (so a dribble/carry counts once, at the touch that releases the ball).
* Shot: a candidate whose predicted free-flight trajectory (RocketSim: the real
  arena, goal frame and ball spin; see rsim.py) reaches the opponent's end within
  the horizon, arriving inside the goal mouth plus a margin (so near-misses count
  as off-target shots). Selection never looks at the outcome, so there is no
  label leakage.
* Setups are not shots: if the shooting team's own next touch comes before the
  ball would have reached the goal line (the shooter dribbling on, a teammate
  following up), the touch was a pass/dribble and the later touch is the shot.
  Otherwise a slow roll into an empty net that someone taps in would be labelled
  a miss, and one attack would be counted twice.
* Two views of each shot: post-touch features (where the ball is actually going,
  used by xGOT) and pre-touch features (`pre_*`, `chance_*`: the situation the
  instant before contact, used by chance xG, which ignores how well the ball was
  struck, so an open-net miss still counts as a big chance).
* Goal (label): the shooting team scores before any of its players touches the
  ball again, and within GOAL_WINDOW seconds. Deflections / failed saves by the
  defence therefore still credit the shot.

All geometry is expressed in the attacking frame: the target goal is at +y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import defence
from . import physics as ph
from . import rsim
from . import situation
from .parse import Replay
from .touches import Touch, detect_touches, quat_to_matrix

HORIZON = 5.0
SHOT_X_MARGIN = 500.0       # wide misses up to ~1.5 ball widths outside the post
SHOT_Z_MARGIN = 250.0
GOAL_WINDOW = 7.0
CLUSTER_GAP = 0.6            # s: same-player touches closer than this form one carry
FOLLOWUP_SLACK = 0.3         # s: own-team touch this soon after the predicted arrival still makes it a setup
CAR_MAX_SPEED = 2300.0
CAR_NO_BOOST_SPEED = 1410.0
REACH = 160.0                # uu: a car this close to the ball path can play it
GOAL_CENTRE = np.array([0.0, ph.ARENA_Y, ph.GOAL_H / 2])


def _flip(team: int) -> np.ndarray:
    """Multiplier that maps world coordinates to the attacking frame."""
    return np.array([1.0, 1.0, 1.0]) if team == 0 else np.array([-1.0, -1.0, 1.0])


def _dead_mask(rep: Replay) -> np.ndarray:
    """True for frames between a goal and the following kickoff reset."""
    dead = np.zeros(rep.n_frames, bool)
    bp, bv = rep.ball_pos, rep.ball_vel
    at_centre = (np.abs(bp[:, 0]) < 5) & (np.abs(bp[:, 1]) < 5) & (np.linalg.norm(bv, axis=1) < 5)
    for g in rep.goals:
        nxt = np.flatnonzero(at_centre[g.frame:])
        end = g.frame + (nxt[0] if len(nxt) else rep.n_frames)
        dead[g.frame:end] = True
    return dead


def _seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to segment a-b, and the projection fraction along it."""
    ab = b - a
    u = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(pts - (a + u[:, None] * ab), axis=1), u


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def candidate_touches(touches: list[Touch]) -> list[int]:
    """Indices of touches that end a same-player run."""
    out = []
    for i, t in enumerate(touches):
        nxt = touches[i + 1] if i + 1 < len(touches) else None
        if nxt is not None and nxt.pid == t.pid and nxt.time - t.time < CLUSTER_GAP:
            continue
        out.append(i)
    return out


def _goal_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Horizontal angle subtended by the goal posts from (x, y)."""
    dy = np.maximum(ph.ARENA_Y - y, 1.0)
    return np.abs(np.arctan2(ph.GOAL_HALF_W - x, dy) - np.arctan2(-ph.GOAL_HALF_W - x, dy))


def extract_shots(rep: Replay, touches: list[Touch] | None = None) -> pd.DataFrame:
    if touches is None:
        touches = detect_touches(rep)
    if not touches:
        return pd.DataFrame()
    dead = _dead_mask(rep)
    cand = [i for i in candidate_touches(touches)
            if not dead[touches[i].frame] and not dead[min(touches[i].post_frame, rep.n_frames - 1)]]
    if not cand:
        return pd.DataFrame()

    # Ball state just after the touch, in the attacking frame
    post = np.array([min(touches[i].post_frame, rep.n_frames - 1) for i in cand])
    flips = np.array([_flip(touches[i].team) for i in cand])
    bpos = rep.ball_pos[post] * flips
    bvel = rep.ball_vel[post] * flips
    # spin turns with the half-turn into the attacking frame exactly like velocity does
    bang = np.nan_to_num(rep.ball_ang_vel[post] * flips) if rep.ball_ang_vel is not None else np.zeros_like(bvel)
    ok = np.isfinite(bpos).all(1) & np.isfinite(bvel).all(1)
    cand = [c for c, k in zip(cand, ok) if k]
    post, flips, bpos, bvel, bang = post[ok], flips[ok], bpos[ok], bvel[ok], bang[ok]
    if not cand:
        return pd.DataFrame()

    pred = rsim.predict(bpos, bvel, bang, HORIZON)
    is_shot = (pred.reaches_goal_plane
               & (np.abs(pred.cross_x) < ph.GOAL_HALF_W + SHOT_X_MARGIN)
               & (pred.cross_z < ph.GOAL_H + SHOT_Z_MARGIN)
               & (bvel[:, 1] > 0))
    # setups: own team plays the ball again before it would have arrived
    for k in np.flatnonzero(is_shot):
        ti = cand[k]
        if ti + 1 < len(touches):
            nxt = touches[ti + 1]
            # a shot RocketSim says had already missed (wide, over, off the frame) on arrival is
            # finished there: a follow-up after that is a rebound, so no slack
            slack = 0.0 if pred.sim_kind[k] in rsim.END_KINDS else FOLLOWUP_SLACK
            if nxt.team == touches[ti].team and nxt.time - touches[ti].time < pred.time_to_goal[k] + slack:
                is_shot[k] = False

    pidx ={p.pid: j for j, p in enumerate(rep.players)}
    goals_by_team = {0: [g for g in rep.goals], 1: [g for g in rep.goals]}
    rows = []
    for k in np.flatnonzero(is_shot):
        ti = cand[k]
        tch = touches[ti]
        f, pf = tch.frame, int(post[k])
        flip = flips[k]
        team = tch.team

        # ---- label ---------------------------------------------------------
        next_own = next((t for t in touches[ti + 1:] if t.team == team), None)
        limit_t = min(next_own.time if next_own else np.inf, tch.time + GOAL_WINDOW)
        goal = next((g for g in goals_by_team[team]
                     if g.team == team and tch.time <= g.time <= limit_t), None)

        # ---- ball / trajectory -------------------------------------------
        bx, by, bz = bpos[k]
        vx, vy, vz = bvel[k]
        speed = float(np.linalg.norm(bvel[k]))
        to_goal = GOAL_CENTRE - bpos[k]
        dist_goal = float(np.linalg.norm(to_goal))
        cos_dir = float(np.dot(bvel[k], to_goal) / max(speed * dist_goal, 1e-9))
        cx, cz = pred.cross_x[k], pred.cross_z[k]
        margin_x = ph.GOAL_HALF_W - ph.BALL_RADIUS - abs(cx)
        margin_z = ph.GOAL_H - ph.BALL_RADIUS - cz
        traj = pred.samples[k]
        st = pred.sample_t
        valid = np.isfinite(traj).all(1)
        traj, st = traj[valid], st[valid]
        cross_pt = np.array([cx, ph.ARENA_Y, cz])

        # ---- shooter ------------------------------------------------------
        sp = rep.players[pidx[tch.pid]]
        s_pos = sp.pos[f] * flip
        s_vel = sp.vel[f] * flip

        # ---- pre-touch situation (chance xG): nothing about where the ball went
        pb = rep.ball_pos[f] * flip
        pv = rep.ball_vel[f] * flip
        if not np.isfinite(pv).all():
            pv = np.zeros(3)
        to_goal_pre = GOAL_CENTRE - pb
        u_goal = _unit(to_goal_pre)
        pre_speed = float(np.linalg.norm(pv))
        q = sp.quat[f]
        if np.isfinite(q).all():
            fwd = quat_to_matrix(q[None])[0][:, 0] * flip
        else:
            fwd = _unit(s_vel) if np.linalg.norm(s_vel) > 1 else np.array([np.nan] * 3)
        car_to_ball = _unit(pb - s_pos)
        # aim target: nearest point of the goal mouth to the ball, pulled inside the posts
        aim = np.array([np.clip(pb[0], -ph.GOAL_HALF_W + 2 * ph.BALL_RADIUS, ph.GOAL_HALF_W - 2 * ph.BALL_RADIUS),
                        ph.ARENA_Y, np.clip(pb[2], ph.BALL_RADIUS, ph.GOAL_H - 2 * ph.BALL_RADIUS)])
        opp = [p for p in rep.players if p.team != team and np.isfinite(p.pos[f]).all()]
        seg_def = []
        for p in opp:
            ppos = p.pos[f] * flip
            boost = p.boost[f] if np.isfinite(p.boost[f]) else 33.0
            vmax = CAR_MAX_SPEED if boost > 5 else CAR_NO_BOOST_SPEED
            d_seg, u_seg = _seg_dist(ppos[None], pb, aim)
            seg_def.append((max(float(d_seg[0]) - REACH, 0.0) / vmax, float(d_seg[0]), float(u_seg[0]),
                            float(np.linalg.norm(ppos - aim)) / vmax))
        # defenders behind the ball (projection at the ball end) can't block the shot line
        seg_def.sort(key=lambda d: d[0] + (5.0 if d[2] <= 0.02 else 0.0))

        # ---- kinematic intercept (defence.py): velocity, facing, ball height, jump/aerial
        opp_state = []
        for p in opp:
            oq = p.quat[f]
            rot = quat_to_matrix(oq[None])[0] if np.isfinite(oq).all() else np.full((3, 3), np.nan)
            opp_state.append(dict(
                pos=p.pos[f] * flip, vel=p.vel[f] * flip, fwd=rot[:, 0] * flip, up=rot[:, 2] * flip,
                boost=float(p.boost[f]) if np.isfinite(p.boost[f]) else 33.0,
                jumped=bool(p.jump[f] or p.double_jump[f] or p.dodge[f])))
        chance_t = float(np.linalg.norm(aim - pb)) / max(pre_speed, 1500.0)
        kin = defence.defence_features(opp_state, traj, st, "k")
        kin.update(defence.defence_features(opp_state, *defence.straight_path(pb, aim, chance_t), "ck"))
        # ---- goal-mouth coverage, pressure, recent movement, game state (situation.py)
        kin.update(situation.goal_coverage(opp_state, pb, pre_speed))
        kin.update(situation.pressure(opp_state, pb))
        kin.update(situation.history(rep, f, flip, sp, team))
        kin.update(situation.game_state(rep, f, team))

        # ---- defenders / attackers ---------------------------------------
        defs, atts = [], []
        for p in rep.players:
            ppos = p.pos[f] * flip
            if not np.isfinite(ppos).all():
                continue
            pvel = p.vel[f] * flip
            if p.team != team:
                boost = p.boost[f] if np.isfinite(p.boost[f]) else 33.0
                vmax = CAR_MAX_SPEED if boost > 5 else CAR_NO_BOOST_SPEED
                d_path = np.linalg.norm(traj - ppos, axis=1) if len(traj) else np.array([np.inf])
                t_need = np.maximum(d_path - REACH, 0) / vmax
                margin_t = float(np.min(t_need - st)) if len(traj) else np.inf
                in_cone = (ppos[1] > by) and abs(ppos[0]) < ph.GOAL_HALF_W + (
                    abs(bx) - ph.GOAL_HALF_W) * max(ph.ARENA_Y - ppos[1], 0) / max(ph.ARENA_Y - by, 1)
                defs.append(dict(
                    d_ball=float(np.linalg.norm(ppos - bpos[k])),
                    d_goal=float(np.linalg.norm(ppos - GOAL_CENTRE)),
                    d_cross=float(np.linalg.norm(ppos - cross_pt)) if np.isfinite(cx) else np.nan,
                    d_path=float(d_path.min()),
                    margin_t=margin_t,
                    goal_side=float(ppos[1] > by),
                    in_cone=float(in_cone),
                    boost=float(boost),
                    z=float(ppos[2]),
                    speed=float(np.linalg.norm(pvel)),
                    vy=float(pvel[1]),
                ))
            elif p.pid != tch.pid:
                atts.append(dict(
                    d_ball=float(np.linalg.norm(ppos - bpos[k])),
                    d_goal=float(np.linalg.norm(ppos - GOAL_CENTRE)),
                ))

        by_margin = sorted(defs, key=lambda d: d["margin_t"])
        by_goal = sorted(defs, key=lambda d: d["d_goal"])
        def dget(lst, i, key, default):
            return lst[i][key] if len(lst) > i else default

        # time since this player's carry started (dribble length)
        j = ti
        while j > 0 and touches[j - 1].pid == tch.pid and touches[j].time - touches[j - 1].time < CLUSTER_GAP:
            j -= 1
        carry_len = tch.time - touches[j].time
        before_carry = touches[j - 1] if j > 0 else None

        rows.append(dict(
            # ---- identity / outcome
            replay_id=rep.replay_id, frame=f, post_frame=pf, time=float(rep.time[f]),
            seconds_remaining=float(rep.seconds_remaining[f]),
            pid=tch.pid, player=sp.name, team=team, team_size=rep.team_size,
            goal=int(goal is not None),
            goal_frame=goal.frame if goal else -1,
            # ---- ball
            ball_x=bx, ball_abs_x=abs(bx), ball_y=by, ball_z=bz,
            ball_vx=vx, ball_vy=vy, ball_vz=vz, ball_speed=speed,
            dist_goal=dist_goal, dist_goal_line=ph.ARENA_Y - by,
            goal_angle=float(_goal_angle(np.array([bx]), np.array([by]))[0]),
            angle_off_centre=float(np.arctan2(abs(bx), max(ph.ARENA_Y - by, 1.0))),
            dir_cos_goal=cos_dir,
            touch_dv=tch.dv,
            # ---- predicted trajectory
            on_target=float(pred.on_target[k]),
            cross_x=cx, cross_abs_x=abs(cx), cross_z=cz,
            cross_margin=float(min(margin_x, margin_z)),
            time_to_goal=float(pred.time_to_goal[k]),
            cross_speed=float(pred.cross_speed[k]),
            n_bounces=int(pred.bounces[k]),
            ball_spin=float(np.linalg.norm(bang[k])),
            # what RocketSim says happens if nobody touches it again
            sim_kind=pred.sim_kind[k],
            sim_goal=float(pred.sim_goal[k]),
            sim_frame=float(pred.sim_kind[k] in ("post", "bar")),
            sim_time=float(pred.sim_time[k]),
            sim_p_goal=rsim.p_goal(bpos[k], bvel[k], bang[k], HORIZON, seed=f"{rep.replay_id}:{f}"),
            # ---- pre-touch situation (chance xG)
            pre_ball_vx_to_centre=float(-np.sign(pb[0]) * pv[0]), pre_ball_vy=float(pv[1]),
            pre_ball_vz=float(pv[2]), pre_ball_speed=pre_speed,
            pre_ball_dir_cos_goal=float(pv @ u_goal / pre_speed) if pre_speed > 1 else 0.0,
            pre_ball_z=float(pb[2]),
            shooter_d_ball=float(np.linalg.norm(pb - s_pos)),
            shooter_align=float(car_to_ball @ u_goal),           # 1 = car directly behind ball, facing goal line
            shooter_facing_goal=float(fwd @ u_goal) if np.isfinite(fwd).all() else np.nan,
            shooter_facing_ball=float(fwd @ car_to_ball) if np.isfinite(fwd).all() else np.nan,
            shooter_v_to_goal=float(s_vel @ u_goal) if np.isfinite(s_vel).all() else np.nan,
            shooter_v_to_ball=float(s_vel @ car_to_ball) if np.isfinite(s_vel).all() else np.nan,
            chance_def1_t=seg_def[0][0] if seg_def else 5.0,       # time for best defender to reach the ball->goal line
            chance_def1_d=seg_def[0][1] if seg_def else 8000.0,
            chance_def1_u=seg_def[0][2] if seg_def else 1.0,
            chance_def2_t=seg_def[1][0] if len(seg_def) > 1 else 5.0,
            chance_def_goal_t=min((d[3] for d in seg_def), default=5.0),  # quickest defender to the aim point
            chance_ball_goal_t=float(np.linalg.norm(aim - pb)) / max(pre_speed, 1500.0),
            # ---- shooter
            shooter_z=float(s_pos[2]), shooter_speed=float(np.linalg.norm(s_vel)),
            shooter_vy=float(s_vel[1]),
            shooter_airborne=float(s_pos[2] > 60),
            shooter_boost=float(sp.boost[f]) if np.isfinite(sp.boost[f]) else np.nan,
            shooter_dodge=float(sp.dodge[f] or sp.dodge[pf]),
            carry_time=carry_len,
            # ---- previous touch context
            since_prev_touch=(tch.time - before_carry.time) if before_carry else np.nan,
            prev_by_self=float(before_carry is not None and before_carry.pid == tch.pid),
            prev_by_teammate=float(before_carry is not None and before_carry.team == team
                                   and before_carry.pid != tch.pid),
            prev_by_opponent=float(before_carry is not None and before_carry.team != team),
            # ---- defence
            n_def=len(defs),
            n_def_goal_side=sum(d["goal_side"] for d in defs),
            n_def_in_cone=sum(d["in_cone"] for d in defs),
            n_def_can_intercept=sum(d["margin_t"] < 0 for d in defs),
            def1_margin=dget(by_margin, 0, "margin_t", 5.0),
            def1_d_path=dget(by_margin, 0, "d_path", 8000.0),
            def1_d_ball=dget(by_margin, 0, "d_ball", 8000.0),
            def1_boost=dget(by_margin, 0, "boost", 0.0),
            def1_z=dget(by_margin, 0, "z", 17.0),
            def1_speed=dget(by_margin, 0, "speed", 0.0),
            def2_margin=dget(by_margin, 1, "margin_t", 5.0),
            def2_d_path=dget(by_margin, 1, "d_path", 8000.0),
            gk_d_goal=dget(by_goal, 0, "d_goal", 8000.0),
            gk_d_cross=dget(by_goal, 0, "d_cross", 8000.0),
            gk_goal_side=dget(by_goal, 0, "goal_side", 0.0),
            gk_vy=dget(by_goal, 0, "vy", 0.0),
            nearest_def_ball=min((d["d_ball"] for d in defs), default=8000.0),
            # ---- support
            n_att=len(atts),
            mate_d_goal=min((a["d_goal"] for a in atts), default=8000.0),
            mate_d_ball=min((a["d_ball"] for a in atts), default=8000.0),
            **kin,
        ))
    return pd.DataFrame(rows)
