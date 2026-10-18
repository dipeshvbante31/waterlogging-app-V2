# data/

Local, git-ignored working data. Nothing here is committed or uploaded through the website.

- `raw/`       cached provider responses (so re-running the preparation script does not re-download)
- `processed/` the training table `<name>.csv` and its `<name>.manifest.json`
- `waterlogging.db` default SQLite prediction history (development only)

Create the training table with `python scripts/prepare_dataset.py` (see the root README for the schema).
