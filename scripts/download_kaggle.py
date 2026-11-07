"""Download replays from the High-Level Rocket League Replay Dataset (Kaggle).

https://www.kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset

Needs KAGGLE_API_TOKEN in the environment or .env. Downloads the per-mode
metadata (line-delimited JSON), then the newest N ranked replays file by file,
so only the games you want are fetched (the full dataset is ~174 GB).

    python scripts/download_kaggle.py --mode 2v2 -n 15000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET = "rolvarild/high-level-rocket-league-replay-dataset"
BASE = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/"


def read_token() -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    env = ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "KAGGLE_API_TOKEN":
                tok = v.strip().strip('"').strip("'")
    if not tok:
        sys.exit("No token: set KAGGLE_API_TOKEN in the environment or in .env")
    return tok


def fetch(session: requests.Session, path: str, out: Path, retries: int = 5) -> None:
    url = BASE + urllib.parse.quote(path, safe="")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(out)
                return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="2v2", choices=["1v1", "2v2", "3v3"])
    ap.add_argument("-n", "--count", type=int, default=15000)
    ap.add_argument("--playlist", default="ranked-doubles", help="'' for any playlist")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out or ROOT / "data" / "raw" / f"hlrlrd_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {read_token()}"

    meta_path = out / "metadata.jsonl"
    if not meta_path.exists():
        print("downloading metadata...")
        fetch(s, f"high_level/{args.mode}/metadata.json", meta_path)
    rows = [json.loads(l) for l in meta_path.open(encoding="utf-8")]
    if args.playlist:
        rows = [r for r in rows if r["data"].get("playlist_id") == args.playlist]
    rows.sort(key=lambda r: r["data"].get("date") or "", reverse=True)
    rows = rows[:args.count]
    if not rows:
        print("metadata ready; no replays selected (-n 0)")
        return
    print(f"selected {len(rows)} replays, {rows[-1]['data']['date'][:10]} -> {rows[0]['data']['date'][:10]}")

    with open(out / "selected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "date", "min_rank", "max_rank", "skill_score"])
        for r in rows:
            d = r["data"]
            w.writerow([r["id"], d.get("date"), (d.get("min_rank") or {}).get("id"),
                        (d.get("max_rank") or {}).get("id"), r.get("score")])

    todo = [r["id"] for r in rows if not (out / f"{r['id']}.replay").exists()]
    print(f"{len(rows) - len(todo)} already present, {len(todo)} to download")
    done = failed = 0
    t0 = time.time()

    def job(rid):
        fetch(s, f"high_level/{args.mode}/{rid}.replay", out / f"{rid}.replay")

    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(job, rid): rid for rid in todo}
        for fut in futs:
            try:
                fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"  {futs[fut]}: {e}")
            if (done + failed) % 250 == 0:
                rate = (done + failed) / (time.time() - t0)
                print(f"  {done + failed}/{len(todo)}  ({rate:.1f}/s, failed {failed})", flush=True)
    print(f"done: {done} downloaded, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
