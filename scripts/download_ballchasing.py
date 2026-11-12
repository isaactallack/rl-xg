"""Download ranked replays in a rank band from ballchasing.com, paced to the API quota.

Needs a ballchasing API token (free: log in at ballchasing.com with Steam, then
Upload -> API tokens). Put it in `.env` in the project root, or the environment:

    BALLCHASING_TOKEN=xxxxxxxx

The free tier allows 1 file download a second and 200 an hour. Downloads are spaced
to stay under that (`--interval`, default 18.5 s, about 195 an hour), so a long run
keeps going instead of running into the limit.

The script walks back in time from the newest game, remembering how far it got in
`_state.json`, and skips replays it already has. Each run writes the ballchasing
listing entries of what it downloaded (ranks, players, date) to `_index/<time>.jsonl`.

Two places to keep the replays:

  local folder (default: data/raw/ballchasing_c3_gc2)
      python scripts/download_ballchasing.py fetch -n 1000

  Azure blob container (used by the GitHub Actions workflow, so nothing runs at
  home). Needs AZURE_BLOB_CONTAINER_URL: a container URL with a SAS token that
  allows read, add, create, write and list.
      python scripts/download_ballchasing.py fetch --blob --interval 1.1 --max-files 190
      python scripts/download_ballchasing.py pull      # copy new blobs to the local folder

Default band: ranked doubles where every player is Champion 3 to Grand Champion 2.
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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://ballchasing.com/api"
DEFAULT_OUT = ROOT / "data" / "raw" / "ballchasing_c3_gc2"
STATE = "_state.json"


class QuotaExhausted(Exception):
    """Ballchasing returned 429 and we were told not to wait for the quota to reset."""


def read_env(*keys: str) -> str | None:
    for k in keys:
        if os.environ.get(k):
            return os.environ[k]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in keys and v.strip():
                return v.strip().strip('"').strip("'")
    return None


# ---------------------------------------------------------------------------- storage

class LocalStore:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def replay_ids(self) -> set[str]:
        return {p.stem for p in self.folder.glob("*.replay")}

    def put(self, name: str, data: bytes) -> None:
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, name: str) -> bytes | None:
        path = self.folder / name
        return path.read_bytes() if path.exists() else None

    def __str__(self):
        return str(self.folder)


class BlobStore:
    def __init__(self, container_url: str):
        from azure.storage.blob import ContainerClient   # only needed for blob storage
        self.client = ContainerClient.from_container_url(container_url)

    def names(self, prefix: str | None = None) -> list[str]:
        return list(self.client.list_blob_names(name_starts_with=prefix))

    def replay_ids(self) -> set[str]:
        return {n[:-len(".replay")] for n in self.names() if n.endswith(".replay") and "/" not in n}

    def put(self, name: str, data: bytes) -> None:
        self.client.upload_blob(name, data, overwrite=True)

    def get(self, name: str) -> bytes | None:
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.client.download_blob(name).readall()
        except ResourceNotFoundError:
            return None

    def __str__(self):
        return f"blob container {self.client.container_name}"


def blob_store() -> BlobStore:
    url = read_env("AZURE_BLOB_CONTAINER_URL")
    if not url:
        sys.exit("No AZURE_BLOB_CONTAINER_URL (a container URL with a SAS token) in the environment or .env")
    return BlobStore(url)


# ---------------------------------------------------------------------------- ballchasing

def request(url: str, token: str, wait_on_quota: bool, retries: int = 8) -> bytes:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": token, "User-Agent": "rlxg/0.2"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if not wait_on_quota:
                    raise QuotaExhausted(url) from e
                # the hourly window may need a while to free up; never give up on it
                wait = float(e.headers.get("Retry-After") or min(60 * (attempt + 1), 600))
                print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def fetch(args) -> None:
    token = read_env("BALLCHASING_TOKEN", "TOKEN")
    if not token:
        sys.exit("No token: set BALLCHASING_TOKEN in the environment or in .env")
    store = blob_store() if args.blob else LocalStore(Path(args.out))
    have = store.replay_ids()
    state = json.loads(store.get(STATE) or b"{}")
    print(f"{store}: {len(have)} replays already there; resuming before {state.get('before') or 'now'}", flush=True)
    if args.stop_at and len(have) >= args.stop_at:
        print(f"already have {len(have)} >= --stop-at {args.stop_at}; nothing to do")
        _mark_full(len(have))
        return

    params = {"playlist": args.playlist, "min-rank": args.min_rank, "max-rank": args.max_rank,
              "sort-by": "replay-date", "sort-dir": "desc", "count": 200}
    if state.get("before"):
        params["replay-date-before"] = state["before"]
    url = f"{API}/replays?{urllib.parse.urlencode(params)}"

    want = args.count
    if args.stop_at:
        want = min(want, args.stop_at - len(have))
    got = failed = 0
    index = []
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    last_download = 0.0

    def save():
        store.put(STATE, json.dumps(state, indent=1).encode())
        if index:
            store.put(f"_index/{started}.jsonl", "".join(json.dumps(r) + "\n" for r in index).encode())

    try:
        while url and got < want:
            page = json.loads(request(url, token, wait_on_quota=not args.max_files))
            items = page.get("list", [])
            if not items:
                print("no more replays listed")
                break
            for r in items:
                if got >= want:
                    break
                if r["id"] not in have:
                    time.sleep(max(0.0, args.interval - (time.monotonic() - last_download)))
                    last_download = time.monotonic()
                    try:
                        data = request(f"{API}/replays/{r['id']}/file", token, wait_on_quota=not args.max_files)
                        store.put(f"{r['id']}.replay", data)
                        have.add(r["id"])
                        index.append(r)
                        got += 1
                    except QuotaExhausted:
                        raise
                    except Exception as e:
                        failed += 1
                        print(f"  {r['id']}: {e}", flush=True)
                        continue
                    if got % 25 == 0:
                        print(f"  {got}/{want} downloaded, {failed} failed, at {r.get('date')}", flush=True)
                # walk back: next time, start from the oldest game handled so far
                if r.get("date"):
                    state["before"] = r["date"]
            save()                           # once a page, so a killed run keeps its place
            url = page.get("next")
            time.sleep(0.6)                  # list calls: 2/s, 500/hour on the free tier
    except QuotaExhausted:
        print("hit the hourly quota; stopping until the next run", flush=True)
    finally:
        save()
    print(f"done: {got} downloaded, {failed} failed -> {store} ({len(have)} in total)")
    if args.stop_at and len(have) >= args.stop_at:
        _mark_full(len(have))


def _mark_full(n: int) -> None:
    """Tell the GitHub Actions workflow the target is reached (it then opens an issue and disables itself)."""
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"full=true\ncount={n}\n")


def pull(args) -> None:
    """Copy replays (and the index) from the blob container to the local folder."""
    blob, local = blob_store(), LocalStore(Path(args.out))
    names = blob.names()
    have = local.replay_ids()
    todo = [n for n in names if (n.endswith(".replay") and n[:-7] not in have)
            or (n.startswith("_index/") and local.get(n) is None)]
    print(f"{len(names)} blobs, {len(todo)} to copy -> {local}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(lambda n: local.put(n, blob.get(n)), todo), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done")


def status(args) -> None:
    """How many replays the Azure container holds, how far back it has got, and their size."""
    blob = blob_store()
    n = size = 0
    for b in blob.client.list_blobs():
        if b.name.endswith(".replay"):
            n += 1
            size += b.size
    state = json.loads(blob.get(STATE) or b"{}")
    print(f"{n} replays, {size / 1e9:.1f} GB; reached back to {state.get('before') or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch", help="download new replays from ballchasing (default)")
    f.add_argument("-n", "--count", type=int, default=1000, help="new replays to download this run")
    f.add_argument("--max-files", type=int, default=0,
                   help="download at most this many and stop at the first 429 instead of waiting (for scheduled runs)")
    f.add_argument("--interval", type=float, default=18.5, help="seconds between file downloads")
    f.add_argument("--stop-at", type=int, default=0, help="stop once the store holds this many replays")
    f.add_argument("--blob", action="store_true", help="store in the Azure container (AZURE_BLOB_CONTAINER_URL)")
    f.add_argument("--playlist", default="ranked-doubles")
    f.add_argument("--min-rank", default="champion-3")
    f.add_argument("--max-rank", default="grand-champion-2")
    f.add_argument("-o", "--out", default=str(DEFAULT_OUT), help="local folder (when not --blob)")
    p = sub.add_parser("pull", help="copy replays from the Azure container to the local folder")
    p.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    sub.add_parser("status", help="count the replays in the Azure container")
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("fetch", "pull", "status", "-h", "--help"):
        argv = ["fetch"] + argv               # old usage: download_ballchasing.py -n 1000
    args = ap.parse_args(argv)
    if args.cmd == "fetch" and args.max_files:
        args.count = min(args.count, args.max_files)
    {"pull": pull, "status": status}.get(args.cmd, fetch)(args)


if __name__ == "__main__":
    main()
