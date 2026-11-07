"""Compare detected shots with the game's own Shot stat events and goals."""
import sys, glob, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rlxg.parse import load_replay
from rlxg.touches import detect_touches
from rlxg.shots import extract_shots

files = sys.argv[1:]
tot = dict(goals=0, goals_hit=0, rl_shots=0, rl_matched=0, ours=0, our_goals=0)
for f in files:
    rep = load_replay(f)
    tt = detect_touches(rep)
    df = extract_shots(rep, tt)
    gf = set(df.loc[df.goal == 1, "goal_frame"]) if len(df) else set()
    tot["goals"] += len(rep.goals)
    tot["goals_hit"] += sum(g.frame in gf for g in rep.goals)
    tot["ours"] += len(df); tot["our_goals"] += int(df.goal.sum()) if len(df) else 0
    # RL shot event matched if we have a shot by that player in the 5 s before it
    for e in rep.stat_events:
        if e.kind != "Shot":
            continue
        tot["rl_shots"] += 1
        if len(df) and ((df.pid == e.pid) & (df.time <= e.time + 0.05) & (df.time >= e.time - 5)).any():
            tot["rl_matched"] += 1
print(tot)
print(f"goal coverage {tot['goals_hit']/max(tot['goals'],1):.1%}  RL-shot recall {tot['rl_matched']/max(tot['rl_shots'],1):.1%}  "
      f"our shots / RL shots {tot['ours']/max(tot['rl_shots'],1):.2f}  conv {tot['our_goals']/max(tot['ours'],1):.1%}")
