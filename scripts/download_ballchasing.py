"""Download the newest ranked replays in a rank band from ballchasing.com.

Needs a ballchasing API token (free: log in at ballchasing.com with Steam, then
Upload -> API tokens). Put it in a `.env` file in the project root:

    BALLCHASING_TOKEN=xxxxxxxx

Default: newest ranked-doubles games between Champion 3 and Grand Champion 2,
from any uploader:

    python scripts/download_ballchasing.py -n 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://ballchasing.com/api"


def read_token() -> str:
    tok = os.environ.get("BALLCHASING_TOKEN")
    env = ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("BALLCHASING_TOKEN", "TOKEN"):
                tok = v.strip().strip('"').strip("'")
    if not tok:
        sys.exit("No token: set BALLCHASING_TOKEN in the environment or in .env")
    return tok


def request(url: str, token: str, retries: int = 6) -> bytes:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": token, "User-Agent": "rlxg/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:            # rate limited: back off and retry
                wait = 2 ** attempt * 5
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=1000, help="number of replays to fetch")
    ap.add_argument("--playlist", default="ranked-doubles")
    ap.add_argument("--min-rank", default="champion-3")
    ap.add_argument("--max-rank", default="grand-champion-2")
    ap.add_argument("--uploader", default=None, help="optional uploader steam id")
    ap.add_argument("-o", "--out", default=str(ROOT / "data" / "raw" / "ballchasing_c3_gc2"))
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between file downloads")
    args = ap.parse_args()

    token = read_token()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = {"playlist": args.playlist, "min-rank": args.min_rank, "max-rank": args.max_rank,
              "sort-by": "replay-date", "sort-dir": "desc", "count": 200}
    if args.uploader:
        params["uploader"] = args.uploader
    url = f"{API}/replays?{urllib.parse.urlencode(params)}"

    listed = []
    while url and len(listed) < args.count:
        page = json.loads(request(url, token))
        listed += page.get("list", [])
        url = page.get("next")
        print(f"listed {len(listed)}")
        time.sleep(0.5)
    listed = listed[:args.count]
    (out / "_index.json").write_text(json.dumps(listed, indent=1))

    got = skipped = failed = 0
    for i, r in enumerate(listed, 1):
        f = out / f"{r['id']}.replay"
        if f.exists():
            skipped += 1
            continue
        try:
            data = request(f"{API}/replays/{r['id']}/file", token)
            f.write_bytes(data)
            got += 1
        except Exception as e:
            failed += 1
            print(f"  {r['id']}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(listed)} (new {got}, existing {skipped}, failed {failed})")
        time.sleep(args.delay)
    print(f"done: {got} downloaded, {skipped} already present, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
