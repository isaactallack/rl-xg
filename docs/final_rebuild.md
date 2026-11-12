# Final rebuild and clean-up

The plan: collect every replay worth having, extract everything we might want from
them in one pass, train the final model, then shut down the cloud collection and
delete the replays. Nothing is left that costs money, and almost any future feature
can still be computed from what's kept.

Run every command here with the project's virtual environment (`.venv\Scripts\activate`, or
call `.venv/Scripts/python.exe` directly). A plain `python` in a new shell is the system
Python, which lacks `subtr_actor`: the build then fails on every replay without saying why
until it finishes.

## Where things stand (12 Nov 2026)

| Step | Status |
|---|---|
| 1. Kaggle download | done: 48,374 ranked 2v2 replays in `data/raw/hlrlrd_2v2` |
| 1. Ballchasing collection | running: 765 of 20,000 in Azure (GitHub Actions, ~190/hour, about 4 more days) |
| 2. Copy ballchasing to the PC | done for what exists so far: 844 replays locally (426 from 6 Nov + 418 from Azure) |
| 4. Shot table | interim build done: `data/shots_v7_interim.csv`, 1.23M shots from 49,847 replays (1 failed to parse) |
| 5. Feature comparison | done: every group helps, all kept (`data/compare_features_v7.log`) |
| 5. Training | done on the interim table: `models/xg_model.joblib` (previous model kept as `models/xg_model_v6_defence.joblib`), log in `data/train_v7.log` |
| 5. Learning curves, README | done: `data/learning_curve.json`, figures and numbers in the README |
| 4–5 again. Full table + recalibration | after the ballchasing collection completes |
| 3, 6, 7. Shut down, check, delete | after that |

Every replay parsed so far is in `data/cache_v7`, so the final build only parses the
ballchasing replays that arrive from now on (and any new games of yours).

The classifiers are trained only on the high-level games (Kaggle + the original repo's).
The C3–GC2 ballchasing games set the C3–GC2 calibration and measure accuracy at that rank,
and your own games are a holdout, so the rest of the ballchasing collection doesn't need a
retrain: the final step is a rebuild of the table plus a recalibration (about 4 minutes, nearly all of it reading the table)
(step 5).

## 1. Collect

**Kaggle (SSL).** Done. All ranked 2v2 games in the dataset's metadata (~48k), straight to
your PC at ~20 files/s:

```bash
python scripts/download_kaggle_archive.py --mode 2v2 -n 100000
```

**Ballchasing (C3–GC2).** The GitHub Actions workflow `.github/workflows/ballchasing.yml`
downloads up to 190 replays an hour into the Azure container `strlxgreplays/ballchasing`
and stops once it holds 20,000 (about 5 days from 12 Nov). When it gets there it opens a
GitHub issue, "Ballchasing collection complete", which GitHub emails you, and disables itself.
To check on it in the meantime:

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
ranks) into `data/raw/ballchasing_c3_gc2`. It only reads from Azure and skips files you
already have, so run it again if it's interrupted. Downloading from Azure is free up to
100 GB a month.

Check that everything arrived before deleting anything in Azure: every replay `status`
counts should be in the local folder (the local folder also has 79 older replays that were
never in Azure, so it holds a few more).

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
python scripts/build_dataset.py data/raw/hlrlrd_2v2 data/raw/ballchasing_c3_gc2 data/raw/legacy_2v2 "C:/Users/Isaac/Documents/My Games/Rocket League/TAGame/Demos" -o data/shots.csv --cache data/cache_v7 -j 12
```

Each worker process handles replay after replay (a crashed or stuck one is replaced, losing
only that replay), and `-j 12` uses every thread of the Ryzen 5 3600XT: about 200 replays a
minute. The 48k Kaggle replays took about 5 hours that way. They and everything else parsed
so far are cached, so this build only parses the ~19,000 new ballchasing replays: about
1.5 hours, plus a few minutes to write the 3.5 GB table. It resumes if stopped: re-run the
same command. For each replay the cache gets:

| File | What | Why |
|---|---|---|
| `<id>.csv` | every shot with all features | training |
| `<id>.touches.csv` | every touch, with the ball's state before and after | re-run shot selection without the replay |
| `<id>.windows.npz` | ball and all cars, 2 s before to 1.5 s after each shot, 15 Hz | compute new features without the replay |
| `<id>.json` | goals, players, scores, stats | re-label shots without the replay |

The shot table also gets lobby ranks (`lobby_min_tier`, `att_tier`, `def_tier`,
`shooter_tier`; ballchasing tiers, 19 = GC1, 22 = SSL) from the metadata files.

## 5. Decide which new features to keep, then train

Done on the interim table, `data/shots_v7_interim.csv`, which has all the high-level games the
classifiers train on. Timings on this PC: the comparison took 36 minutes, training 15 minutes
(about 6 per classifier) and the learning curves 53 minutes.

```bash
python scripts/compare_features.py --cache data/cache_v7
```

This adds each new feature group (goal-mouth coverage, pressure, recent movement, game
state) on its own and all together, and prints the change in log loss with a standard error.
It took about 25 minutes per model on the interim cache. Drop any group from `FEATURES` /
`CHANCE_FEATURES` in `rlxg/model.py` that doesn't beat about two standard errors on the
C3–GC2 and your-games rows. Then:

```bash
python scripts/train.py data/shots_v7_interim.csv
```

```bash
python -m pytest tests
```

Then the learning curves and README figures:

```bash
python scripts/learning_curve.py data/shots_v7_interim.csv
```

```bash
python docs/make_figures.py --shots data/shots_v7_interim.csv --replay data/raw/hlrlrd_2v2/0006fea2-b8d4-470d-a3eb-e06500b5f3c4.replay
```

The feature explainer figures search the Kaggle replays for examples, so regenerate them
before deleting the replays if anything about them changes.

**After the full build (step 4)**, keep the classifiers and refit only the calibrations on
the full C3–GC2 set:

```bash
python scripts/train.py data/shots.csv --recalibrate models/xg_model.joblib
```

The model file records which SSL games were held out, so the SSL numbers stay out-of-sample.
Then refresh the README numbers again (the C3–GC2 ones will have tighter error bars).

**Ranks as a model input (optional experiment).** With ranks in the table you can
try training on everything with `att_tier`/`def_tier` as features, in place of the per-band
calibration. That one needs a full retrain. Your own games have no rank metadata, so you'd
pass your rank in at prediction time.

## 6. Before deleting the replays

Check that every replay made it into the cache with its windows:

```bash
python -c "import pathlib; c=pathlib.Path('data/cache_v7'); j=len(list(c.glob('*.json'))); w=len(list(c.glob('*.windows.npz'))); print(j, 'parsed,', w, 'with windows')"
```

The two numbers should match (they did for the interim build: 49,847 each). Replays that
failed to parse (a handful, listed at the end of the build) are lost either way.

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
- old shot tables `data/shots_v*.csv`, `data/shots_v*_replays.csv`, `data/shots_kaggle*.csv`
  (`data/shots_v7_interim*.csv` too, once `data/shots.csv` is rebuilt)

## What you can still do afterwards

- **New features about a shot** from the windows: anything the ball and cars did in the
  2 s before the touch or the 1.5 s after. Load one with `rlxg.windows.load(path)`;
  `rlxg/windows.py` documents the arrays.
- **Different shot rules** from `*.touches.csv` and the `.json` goals: re-run the RocketSim
  prediction from each touch's post-touch ball state, and relabel. New shots found this way
  would have no window, so only ball-only features for those.
- **Not possible without replays:** changing touch detection itself, or features that need
  more than 2 s of build-up.
