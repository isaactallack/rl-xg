"""Build the shot-level training table from folders of replays.

    python scripts/build_dataset.py data/raw/legacy_2v2 "C:/.../TAGame/Demos" -o data/shots.csv

Each replay is processed in its own subprocess (the Rust parser can hard-crash on
corrupt files) and cached under data/cache/, so re-runs only parse new replays.
"""
from __future__ import annotations

import argparse
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
    rep = load_replay(path)
    touches = detect_touches(rep)
    df = extract_shots(rep, touches)
    # keep every touch too, so shot-selection / labelling rules can be revised later without re-parsing
    pd.DataFrame([asdict(t) for t in touches]).to_csv(Path(out_csv).with_suffix(".touches.csv"), index=False)
    df["source"] = Path(path).parent.name
    df["date"] = rep.date
    df["playlist_id"] = rep.playlist_id
    df.to_csv(out_csv, index=False)
    info = dict(path=path, replay_id=rep.replay_id, date=rep.date, playlist_id=rep.playlist_id,
                match_type=rep.match_type, team_size=rep.team_size, n_goals=len(rep.goals),
                n_shots=len(df), n_rl_shots=sum(e.kind == "Shot" for e in rep.stat_events))
    # The .json marks the replay as done, so it must only appear once the data is on disk:
    # fsync the CSVs, then write the marker to a temp file and rename it into place. Otherwise a
    # system crash can leave zero-filled files that later runs treat as finished.
    for f in (Path(out_csv), Path(out_csv).with_suffix(".touches.csv")):
        with open(f, "rb+") as fh:
            os.fsync(fh.fileno())
    marker = Path(out_csv).with_suffix(".json")
    tmp = marker.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(info))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, marker)


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
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        for f in cache.glob("*"):
            f.unlink()
    files = sorted({p.resolve() for d in args.dirs for p in Path(d).glob("*.replay")})
    print(f"{len(files)} replays")

    errors = []
    outs = []
    if args.cached_only:
        outs = [o for o in (cache / (p.stem + ".csv") for p in files) if o.exists() and o.with_suffix(".json").exists()]
        print(f"using {len(outs)} cached replays (skipping {len(files) - len(outs)} not yet parsed)")
        files = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for i, (out, err) in enumerate(ex.map(lambda p: run_one(p, cache, args.timeout), files), 1):
            (errors.append((out.stem, err)) if err else outs.append(out))
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    dfs = []
    for o in outs:
        try:
            dfs.append(pd.read_csv(o))
        except pd.errors.EmptyDataError:
            pass
    shots = pd.concat([d for d in dfs if len(d)], ignore_index=True)
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
