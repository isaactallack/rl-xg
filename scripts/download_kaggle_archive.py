"""Pull selected replays out of the Kaggle dataset archive without downloading all of it.

Kaggle rate-limits per-file API downloads, but a whole-dataset download is a single
API call that redirects to a signed Google Cloud Storage URL for one big zip. GCS
supports HTTP range requests, so we read the zip's central directory remotely and
then fetch only the members we want, in parallel, with no further Kaggle calls.

    python scripts/download_kaggle_archive.py --mode 2v2 -n 15000

Selection (newest ranked games first) uses the metadata downloaded by
download_kaggle.py, i.e. data/raw/hlrlrd_2v2/metadata.jsonl.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_kaggle import DATASET, read_token  # noqa: E402

API = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.s, self.pos = url, session, 0
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = get_range(self.s, self.url, self.pos, self.pos + n - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def get_range(s: requests.Session, url: str, a: int, b: int, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"range {a}-{b} failed")


def signed_url(token: str) -> str:
    for attempt in range(8):
        r = requests.get(API, headers={"Authorization": f"Bearer {token}"}, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 303, 307) and r.headers.get("Location"):
            return r.headers["Location"]
        if r.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"Kaggle API rate limited (429); waiting {wait}s")
            time.sleep(wait)
            continue
        sys.exit(f"Unexpected response {r.status_code}: {r.text[:200]}")
    sys.exit("Still rate limited; try again later")


def extract_member(s: requests.Session, url: str, zi: zipfile.ZipInfo, out: Path) -> int:
    # local header (30 bytes) + name + extra, then the compressed data
    guess = 30 + len(zi.filename.encode()) + 256 + zi.compress_size
    blob = get_range(s, url, zi.header_offset, zi.header_offset + guess - 1)
    sig, *_ = struct.unpack("<I", blob[:4])
    if sig != 0x04034B50:
        raise ValueError("bad local header")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    if start + zi.compress_size > len(blob):          # extra field bigger than guessed
        blob += get_range(s, url, zi.header_offset + len(blob), zi.header_offset + start + zi.compress_size - 1)
    data = blob[start:start + zi.compress_size]
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif zi.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {zi.compress_type}")
    if zlib.crc32(data) != zi.CRC:
        raise ValueError("CRC mismatch")
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="2v2", choices=["1v1", "2v2", "3v3"])
    ap.add_argument("-n", "--count", type=int, default=15000)
    ap.add_argument("--playlist", default="ranked-doubles")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out or ROOT / "data" / "raw" / f"hlrlrd_{args.mode}")
    meta = out / "metadata.jsonl"
    if not meta.exists():
        sys.exit(f"{meta} missing: run download_kaggle.py once to fetch the metadata")
    rows = [json.loads(l) for l in meta.open(encoding="utf-8")]
    if args.playlist:
        rows = [r for r in rows if r["data"].get("playlist_id") == args.playlist]
    rows.sort(key=lambda r: r["data"].get("date") or "", reverse=True)
    want = {f"high_level/{args.mode}/{r['id']}.replay": r["id"] for r in rows[:args.count]}
    todo = {k: v for k, v in want.items() if not (out / f"{v}.replay").exists()}
    print(f"selected {len(want)}, {len(want) - len(todo)} already present, {len(todo)} to fetch")
    if not todo:
        return

    url = signed_url(read_token())
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.jobs, pool_maxsize=args.jobs)
    s.mount("https://", adapter)
    raw = HTTPRangeFile(url, s)
    print(f"archive {raw.size / 1e9:.1f} GB; reading central directory...")
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=8 << 20))
    members = [zi for zi in zf.infolist() if zi.filename in todo]
    missing = len(todo) - len(members)
    print(f"{len(zf.infolist())} entries in archive; fetching {len(members)} ({missing} not found)")

    t0, done, failed, nbytes = time.time(), 0, 0, 0
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(extract_member, s, url, zi, out / f"{todo[zi.filename]}.replay"): zi for zi in members}
        for fut in as_completed(futs):
            try:
                nbytes += fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"  {futs[fut].filename}: {e}")
            if (done + failed) % 500 == 0:
                el = time.time() - t0
                print(f"  {done + failed}/{len(members)}  {done / el:.1f} files/s  {nbytes / el / 1e6:.1f} MB/s  failed {failed}",
                      flush=True)
    print(f"done: {done} extracted, {failed} failed in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
