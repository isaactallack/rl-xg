"""Command line: python -m rlxg <replay or folder> [...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .analyze import SHOT_COLUMNS, analyze_replay, get_model


def _fmt_clock(secs: float) -> str:
    secs = max(int(round(secs)), 0)
    return f"{secs // 60}:{secs % 60:02d}"


def print_match(m, show_shots: bool) -> None:
    r = m.replay
    print(f"\n=== {Path(r.path).name}  ({r.date or 'unknown date'}, {r.team_size}v{r.team_size})")
    print(f"Score  blue {r.final_score[0]} - {r.final_score[1]} orange   |   "
          f"xG  blue {m.teams.xg[0]:.2f} - {m.teams.xg[1]:.2f} orange   |   "
          f"xGOT  blue {m.teams.xgot[0]:.2f} - {m.teams.xgot[1]:.2f} orange\n")
    print(m.teams.to_string(index=False))
    print()
    cols = ["player", "team", "goals", "shots", "on_target", "xg", "xgot", "goals_minus_xg",
            "big_chances", "big_chances_missed", "saves"]
    print(m.players[cols].to_string(index=False))
    if show_shots and len(m.shots):
        s = m.shots.copy()
        s["clock"] = s.seconds_remaining.map(_fmt_clock)
        s["result"] = s.goal.map({1: "GOAL", 0: ""})
        s["xg"] = s.xg.round(3)
        s["xgot"] = s.xgot.round(3)
        s["dist"] = s.dist_goal.round(0).astype(int)
        s["speed"] = s.ball_speed.round(0).astype(int)
        print("\nShots  (xG = chance before the touch, xGOT = the shot as struck)")
        print(s[["clock", "team", "player", "xg", "xgot", "result", "dist", "speed"]].to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rlxg",
                                 description="Expected goals (xG) for Rocket League replays.")
    ap.add_argument("paths", nargs="+", help=".replay files or folders of them")
    ap.add_argument("--shots", action="store_true", help="also print every shot")
    ap.add_argument("--json", metavar="FILE", help="write full results (all replays) as JSON")
    ap.add_argument("--csv", metavar="FILE", help="write all shots as CSV")
    ap.add_argument("--model", metavar="FILE", help="path to a trained model (default: models/xg_model.joblib)")
    ap.add_argument("--skill", default=None, metavar="BAND",
                    help="rank band the xG is calibrated to: c3-gc2 (default) or ssl")
    args = ap.parse_args(argv)

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.glob("*.replay")) if p.is_dir() else [p]
    model = get_model(args.model)
    results, all_shots = [], []
    for f in files:
        try:
            m = analyze_replay(f, model, band=args.skill)
        except Exception as e:  # keep going on bad replays
            print(f"\n!!! {f}: {e}", file=sys.stderr)
            continue
        print_match(m, args.shots)
        results.append(m.to_dict())
        if len(m.shots):
            all_shots.append(m.shots.assign(replay=Path(f).name)[["replay"] + SHOT_COLUMNS])

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nWrote {args.json}")
    if args.csv and all_shots:
        pd.concat(all_shots).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
