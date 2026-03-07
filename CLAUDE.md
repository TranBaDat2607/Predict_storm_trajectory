# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.10, PostgreSQL with PostGIS, Windows (use Unix shell syntax in bash)
- `DATABASE_URL` env var must be set for any DB or ML work:
  ```bash
  export DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/storm_db
  ```

## Common Commands

```bash
# Crawl raw IBTrACS data → data/raw/storm_data.csv
python src/crawling_data/crawler.py

# Preprocess → data/processed/storm_data.csv
python -m src.data.preprocessor

# Ingest processed CSV into PostgreSQL
python -m src.db.ingest

# Train Transformer model (reads from DB, writes checkpoint to models/)
python -m src.models.train

# Evaluate on test set + generate plots
python -m src.models.evaluate
```

## Architecture

### Data Flow

```
IBTrACS (web) → crawler.py → data/raw/storm_data.csv
                            → preprocessor.py → data/processed/storm_data.csv
                                              → ingest.py → PostgreSQL (storm_db)
                                                          → dataset.py → StormTransformer
```

All ML data comes from PostgreSQL, not from CSV files.

### Database Schema (`src/db/schema.sql`)

Two tables:
- `storms` — one row per storm (`atcf_id` PK, `season`, `basin`, `subbasin`)
- `storm_observations` — one row per storm × 3h timestep; `geom` is a PostGIS GEOGRAPHY column generated from `lat`/`lon`; `nature` FK to `nature_types` lookup

`ON CONFLICT DO NOTHING` is used everywhere — ingestion is idempotent.

### ML Pipeline (`src/models/`)

- **`dataset.py`** — loads from DB via SQL join, engineers 16 features (including delta lat/lon and cyclical `storm_dir`), builds `SEQ_LEN=8` sliding windows, fits `StandardScaler` on train rows only (SEASON ≤ 2014). Exports `build_datasets()` which returns `(train_ds, val_ds, test_ds, scaler_X, scaler_y)`.
- **`transformer.py`** — `StormTransformer`: input projection + learned positional embeddings → 3-layer TransformerEncoder (4 heads, d_ff=256) → mean pool → MLP head → 3 outputs (d_lat, d_lon, wind_speed).
- **`train.py`** — AdamW + CosineAnnealingLR + early stopping (patience=10). Saves best checkpoint to `models/storm_transformer.pt` and `models/training_log.json`.
- **`evaluate.py`** — loads checkpoint + scalers, computes haversine distance and wind MAE, generates trajectory plots and loss curves.

### Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `SEQ_LEN` | 8 | 24-hour input window (8 × 3h) |
| `N_FEATURES` | 16 | Input feature dimension |
| `N_TARGETS` | 3 | d_lat, d_lon, wind_speed |
| Train split | SEASON ≤ 2014 | ~99K windows |
| Val split | 2015–2019 | ~7K windows |
| Test split | SEASON ≥ 2020 | ~7K windows |

### Coordinate Convention

Targets are **deltas** (d_lat, d_lon), not absolute positions. Use `predict_absolute()` from `dataset.py` to recover absolute lat/lon by adding deltas to the last known position in the window. `nature` is label-encoded 0–5 in `dataset.py` (DS=0, ET=1, MX=2, NR=3, SS=4, TS=5) — this differs from the DB schema's `nature_types` lookup which uses 0–4.

### Runtime Artifacts (git-ignored)

`models/storm_transformer.pt`, `models/scaler_X.pkl`, `models/scaler_y.pkl`, `models/training_log.json`, `models/trajectory_plots.png`, `models/loss_curves.png`, `data/raw/`, `data/processed/`.
