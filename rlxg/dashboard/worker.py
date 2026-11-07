"""Subprocess entry point: analyse one replay and print the result as JSON.

Run in a child process so a parser crash on a corrupt replay can't take down
the dashboard server.
"""
from __future__ import annotations

import json
import math
import sys


def _clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):           # numpy scalars
        return _clean(o.item())
    return o


def main() -> None:
    from rlxg.parse import load_replay
    path = sys.argv[1]
    team_size = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "0" else None
    rep = load_replay(path)
    if team_size and rep.team_size != team_size:
        sys.stdout.write(json.dumps({"skipped": f"{rep.team_size}v{rep.team_size} (model is {team_size}v{team_size} only)",
                                     "replay_id": rep.replay_id}))
        return
    from rlxg.analyze import analyze_replay
    band = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    m = analyze_replay(rep, band=band)
    out = _clean(m.to_dict())
    out["clips"] = _clean(m.clips())
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
