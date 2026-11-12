"""Build the shot-level training table from folders of replays.

    python scripts/build_dataset.py data/raw/legacy_2v2 "C:/.../TAGame/Demos" -o data/shots.csv

Each replay is processed in its own subprocess (the Rust parser can hard-crash on
corrupt files) and cached under data/cache/, so re-runs only parse new replays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def worker(path: str, out_csv: str) -> None:
    from rlxg.parse import load_replay
    from dataclasses import asdict
    from rlxg.shots import extract_shots
    from rlxg.touches import detect_touches
    from rlxg import windows
    rep = load_replay(path)
    touches = detect_touches(rep)
    df = extract_shots(rep, touches)
    # keep every touch too, so shot-selection / labelling rules can be revised later without re-parsing
    # with the ball's state just before and after each one (world frame): enough to re-run the
    # RocketSim shot selection without the replay
    tdf = pd.DataFrame([asdict(t) for t in touches])
    if len(tdf):
        import numpy as np
        ang = rep.ball_ang_vel if rep.ball_ang_vel is not None else np.full_like(rep.ball_vel, np.nan)
        for tag, col in (("pre", "frame"), ("post", "post_frame")):
            fr = tdf[col].clip(upper=rep.n_frames - 1).to_numpy()
            for name, arr in (("", rep.ball_pos), ("v", rep.ball_vel), ("w", ang)):
                for i, ax in enumerate("xyz"):
                    tdf[f"{tag}_ball_{name}{ax}"] = arr[fr, i].round(2)
    tdf.to_csv(Path(out_csv).with_suffix(".touches.csv"), index=False)
    # and the raw ball/car states around each shot, so new features don't need the replay (rlxg/windows.py)
    win = Path(out_csv).with_suffix(".windows.npz")
    if os.environ.get("RLXG_WINDOWS", "1") != "0":
        windows.save(win, windows.shot_windows(rep, df))
    df["source"] = Path(path).parent.name
    df["date"] = rep.date
    df["playlist_id"] = rep.playlist_id
    df.to_csv(out_csv, index=False)
    info = dict(path=path, replay_id=rep.replay_id, date=rep.date, playlist_id=rep.playlist_id,
                match_type=rep.match_type, team_size=rep.team_size, n_goals=len(rep.goals),
                n_shots=len(df), n_rl_shots=sum(e.kind == "Shot" for e in rep.stat_events),
                # enough to re-label shots without the replay
                goals=[dict(frame=g.frame, time=g.time, team=g.team, scorer=g.scorer_pid) for g in rep.goals],
                players=[dict(pid=p.pid, name=p.name, team=p.team) for p in rep.players],
                final_score=list(rep.final_score))
    # The .json marks the replay as done, so it must only appear once the data is on disk:
    # fsync the CSVs, then write the marker to a temp file and rename it into place. Otherwise a
    # system crash can leave zero-filled files that later runs treat as finished.
    for f in (Path(out_csv), Path(out_csv).with_suffix(".touches.csv"), win):
        if not f.exists():
            continue
        with open(f, "rb+") as fh:
            os.fsync(fh.fileno())
    marker = Path(out_csv).with_suffix(".json")
    tmp = marker.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(info))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, marker)


def _game_ranks(d: dict) -> dict:
    """Rank tiers from a ballchasing listing entry (also what the Kaggle metadata holds under 'data')."""
    tier = lambda r: (r or {}).get("tier")
    teams = {}
    names = {}
    for k, colour in ((0, "blue"), (1, "orange")):
        ts = []
        for p in (d.get(colour) or {}).get("players", []):
            t = tier(p.get("rank"))
            if t is not None:
                ts.append(t)
                names[(p.get("name") or "").lower()] = t
        teams[k] = sum(ts) / len(ts) if ts else None
    return dict(min=tier(d.get("min_rank")), max=tier(d.get("max_rank")), teams=teams, names=names)


def load_rank_meta(dirs: list[str]) -> dict:
    """rocket_league_id (= the replay header Id) -> ranks, from the metadata saved next to the replays."""
    out = {}
    for d in map(Path, dirs):
        entries = []
        if (d / "metadata.jsonl").exists():                       # Kaggle
            entries += [json.loads(l)["data"] for l in (d / "metadata.jsonl").open(encoding="utf-8")]
        if (d / "_index.json").exists():                          # ballchasing, old downloader
            entries += json.loads((d / "_index.json").read_text(encoding="utf-8"))
        for f in sorted((d / "_index").glob("*.jsonl")):          # ballchasing, current downloader
            entries += [json.loads(l) for l in f.open(encoding="utf-8")]
        for e in entries:
            if e.get("rocket_league_id"):
                out[e["rocket_league_id"].upper()] = _game_ranks(e)
    return out


def add_ranks(shots: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Lobby rank columns (ballchasing tiers: 19 = GC1, 22 = SSL). Recorded for analysis; NaN when unknown."""
    rows = []
    for rid, team, name in zip(shots.replay_id.astype(str).str.upper(), shots.team, shots.player.astype(str)):
        m = meta.get(rid)
        if m is None:
            rows.append((None,) * 5)
            continue
        rows.append((m["min"], m["max"], m["teams"].get(int(team)), m["teams"].get(1 - int(team)),
                     m["names"].get(name.lower())))
    cols = ["lobby_min_tier", "lobby_max_tier", "att_tier", "def_tier", "shooter_tier"]
    ranks = pd.DataFrame(rows, columns=cols, index=shots.index).astype(float)
    return pd.concat([shots.drop(columns=cols, errors="ignore"), ranks], axis=1)


def run_one(path: Path, cache: Path, timeout: float) -> tuple[Path, str | None]:
    out = cache / (path.stem + ".csv")
    if out.exists() and out.with_suffix(".json").exists():
        return out, None
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", str(path), str(out)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return out, "timeout"
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return out, msg[:200]
    return out, None


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", default="data/shots.csv")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--rebuild", action="store_true", help="ignore cached per-replay results")
    ap.add_argument("--cached-only", action="store_true",
                    help="don't parse anything: assemble the table from replays already in the cache "
                         "(train on a partial rebuild)")
    ap.add_argument("-n", "--limit", type=int, default=0,
                    help="only the first N replays. Folders are taken in the order given and each folder's "
                         "replays in a fixed shuffled order, so a limit is a random sample of the last folder "
                         "reached, and a bigger limit later carries on from the same place")
    ap.add_argument("--no-windows", action="store_true",
                    help="don't save the raw ball/car states around each shot (rlxg/windows.py)")
    args = ap.parse_args()
    if args.no_windows:
        os.environ["RLXG_WINDOWS"] = "0"     # read by the worker subprocesses

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        for f in cache.glob("*"):
            f.unlink()
    files, seen = [], set()
    for d in args.dirs:
        batch = sorted({p.resolve() for p in Path(d).glob("*.replay")} - seen,
                       key=lambda p: hashlib.md5(p.stem.encode()).hexdigest())
        files += batch
        seen.update(batch)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} replays")

    errors = []
    outs = []
    if args.cached_only:
        outs = [o for o in (cache / (p.stem + ".csv") for p in files) if o.exists() and o.with_suffix(".json").exists()]
        print(f"using {len(outs)} cached replays (skipping {len(files) - len(outs)} not yet parsed)")
        files = []
    ex = ThreadPoolExecutor(args.jobs)
    try:
        for i, (out, err) in enumerate(ex.map(lambda p: run_one(p, cache, args.timeout), files), 1):
            (errors.append((out.stem, err)) if err else outs.append(out))
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    except KeyboardInterrupt:
        # finished replays are in the cache: the same command carries on from here
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(f"stopped after {len(outs)} replays. Re-run the same command to carry on, "
                 f"or add --cached-only to build the table from what is done so far.")
    ex.shutdown()

    dfs = []
    for o in outs:
        try:
            dfs.append(pd.read_csv(o))
        except pd.errors.EmptyDataError:
            pass
    shots = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    meta = load_rank_meta(args.dirs)
    shots = add_ranks(shots, meta)
    print(f"lobby ranks known for {shots.lobby_min_tier.notna().mean():.0%} of shots")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(out, index=False)
    infos = [json.loads(o.with_suffix(".json").read_text()) for o in outs]
    pd.DataFrame(infos).to_csv(out.with_name(out.stem + "_replays.csv"), index=False)
    print(f"{len(shots)} shots, {int(shots.goal.sum())} goals from {len(outs)} replays -> {out}")
    print(f"{len(errors)} failed")
    for stem, e in errors:
        print("  ", stem, e)


if __name__ == "__main__":
    main()
