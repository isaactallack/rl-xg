# Final rebuild and clean-up

The plan: collect every replay worth having, extract everything we might want from
them in one pass, train the final model, then shut down the cloud collection and
delete the replays. Nothing is left that costs money, and almost any future feature
can still be computed from what's kept.

## 1. Collect

**Kaggle (SSL).** All ranked 2v2 games in the dataset's metadata (~48k), straight to
your PC at ~20 files/s:

```bash
python scripts/download_kaggle_archive.py --mode 2v2 -n 100000
```

**Ballchasing (C3–GC2).** The GitHub Actions workflow `.github/workflows/ballchasing.yml`
downloads up to 190 replays an hour into the Azure container `strlxgreplays/ballchasing`
and stops once it holds 20,000 (about 5 days). When it gets there it opens a GitHub issue,
"Ballchasing collection complete", which GitHub emails you, and disables itself. To check on it
in the meantime:

```bash
python scripts/download_ballchasing.py status
```

```bash
gh run list --repo isaactallack/rl-xg --workflow ballchasing.yml --limit 5
```

It walks back in time from the newest C3–GC2 game (late April 2026). Collect more by raising
`STOP_AT` in the workflow; fewer by disabling it early (step 3).

## 2. Copy the ballchasing replays to your PC

```bash
python scripts/download_ballchasing.py pull
```

This copies every replay and its metadata (`_index/*.jsonl`, which holds the lobby
ranks) into `data/raw/ballchasing_c3_gc2`. It skips files you already have, so run it again
if it's interrupted. Downloading from Azure is free up to 100 GB a month.

Check that everything arrived before deleting anything in Azure: the count from `status`
should match the number of `.replay` files locally.

```bash
python -c "import pathlib; print(len(list(pathlib.Path('data/raw/ballchasing_c3_gc2').glob('*.replay'))))"
```

## 3. Shut down the cloud side

In this order, once step 2 checks out:

```bash
gh workflow disable ballchasing.yml --repo isaactallack/rl-xg
```

```bash
gh secret delete BALLCHASING_TOKEN --repo isaactallack/rl-xg
```

```bash
gh secret delete AZURE_BLOB_CONTAINER_URL --repo isaactallack/rl-xg
```

```bash
az group delete --name rg-rlxg --yes
```

The last command deletes the storage account and every replay in it, permanently. It's the
only thing that costs money, so after this the bill is zero. Confirm with:

```bash
az group exists --name rg-rlxg
```

(prints `false`). Then remove the `AZURE_BLOB_CONTAINER_URL=` line from `.env`. If you
won't collect again, delete the workflow file from the repo too.

## 4. Rebuild the shot table, once, with everything

```bash
python scripts/build_dataset.py data/raw/hlrlrd_2v2 data/raw/ballchasing_c3_gc2 data/raw/legacy_2v2 "C:/Users/Isaac/Documents/My Games/Rocket League/TAGame/Demos" -o data/shots.csv --cache data/cache_v7 -j 8
```

A sample of 450 replays took 5 minutes with `-j 8` on this PC, so the full ~68k is roughly
12–14 hours: start it in the evening. It resumes if stopped: re-run the same command.
For each replay the cache gets:

| File | What | Why |
|---|---|---|
| `<id>.csv` | every shot with all features | training |
| `<id>.touches.csv` | every touch, with the ball's state before and after | re-run shot selection without the replay |
| `<id>.windows.npz` | ball and all cars, 2 s before to 1.5 s after each shot, 15 Hz | compute new features without the replay |
| `<id>.json` | goals, players, scores, stats | re-label shots without the replay |

The shot table also gets lobby ranks (`lobby_min_tier`, `att_tier`, `def_tier`,
`shooter_tier`; ballchasing tiers, 19 = GC1, 22 = SSL) from the metadata files.

## 5. Decide which new features to keep, then train

```bash
python scripts/compare_features.py --cache data/cache_v7
```

This adds each new feature group (goal-mouth coverage, pressure, recent movement, game
state) on its own and all together, and prints the change in log loss with a standard error.
Drop any group from `FEATURES` / `CHANCE_FEATURES` in `rlxg/model.py` that doesn't
beat about two standard errors on the C3–GC2 and your-games rows. Then:

```bash
python scripts/train.py data/shots.csv
```

```bash
python -m pytest tests
```

Update the README numbers and figures (`docs/make_figures.py`).

**Ranks as a model input (optional experiment).** With ranks in the table you can
try training on everything with `att_tier`/`def_tier` as features, in place of the per-band
calibration. Your own games have no rank metadata, so you'd pass your rank in at
prediction time.

## 6. Before deleting the replays

Check that every replay made it into the cache with its windows:

```bash
python -c "import pathlib; c=pathlib.Path('data/cache_v7'); j=len(list(c.glob('*.json'))); w=len(list(c.glob('*.windows.npz'))); print(j, 'parsed,', w, 'with windows')"
```

The two numbers should match. Replays that failed to parse (a handful, listed at the end
of the build) are lost either way.

Then keep these, ideally with a copy on another drive:

- `data/cache_v7/` (shots, touches, windows, per-replay info; roughly 10–15 GB)
- `data/shots.csv` and `data/shots_replays.csv`
- `data/raw/hlrlrd_2v2/metadata.jsonl` and `data/raw/ballchasing_c3_gc2/_index*`
- `models/xg_model.joblib`
- your own replays (the `Demos` folder) and `data/raw/legacy_2v2` (small)

## 7. Delete

- `data/raw/hlrlrd_2v2/*.replay`
- `data/raw/ballchasing_c3_gc2/*.replay`
- old caches `data/cache`, `data/cache_v3` … `data/cache_v6`
- old shot tables `data/shots_v*.csv`, `data/shots_v*_replays.csv`

## What you can still do afterwards

- **New features about a shot** from the windows: anything the ball and cars did in the
  2 s before the touch or the 1.5 s after. Load one with `rlxg.windows.load(path)`;
  `rlxg/windows.py` documents the arrays.
- **Different shot rules** from `*.touches.csv` and the `.json` goals: re-run the RocketSim
  prediction from each touch's post-touch ball state, and relabel. New shots found this way
  would have no window, so only ball-only features for those.
- **Not possible without replays:** changing touch detection itself, or features that need
  more than 2 s of build-up.
