# Predict Storm Trajectory

Predicts tropical cyclone track and intensity using a Transformer model trained on historical IBTrACS data.
Given the last 24 hours of observations, the model outputs the next 8 positions and wind speeds at 3-hour intervals (up to +24 h ahead) in a single forward pass.

---

## Data

Source: [IBTrACS](https://ncics.org/ibtracs/) — International Best Track Archive for Climate Stewardship (NCICS/NOAA)

| Stat | Value |
|---|---|
| Total observations | 132,646 rows |
| Unique storms | 2,343 |
| Temporal resolution | 3-hourly |
| Coverage | 1945 - 2026 |
| Train split | Seasons up to 2014 (~99K windows) |
| Validation split | Seasons 2015 - 2019 (~7.5K windows) |
| Test split | Seasons 2020 and later (~7.2K windows) |

Each observation includes position (lat/lon), wind speed, pressure, storm heading and speed, distance to land, and Saffir-Simpson category. The pipeline engineers 16 input features plus a 2-element context vector (basin, season).

See `src/db/schema.sql` for the full PostgreSQL schema.

---

## Model

StormTransformer — a 3-layer Transformer encoder with ~158K parameters.

```
Input [batch, 8, 16] -> Linear(16->64) + positional embedding + context embedding
    -> TransformerEncoder (3 layers, 4 heads, d_ff=256)
    -> Last-token pool [batch, 64]
    -> MLP head -> [batch, 8, 3]
```

- Input: 8-step sliding window (24 h) of 16 storm features
- Context: basin embedding and season projection, added at every timestep
- Output: 8 future steps x 3 targets (d_lat, d_lon, wind speed) — all in one forward pass, no autoregressive rollout
- Padding: left-zero-padding with attention masking so the model can predict from the very first timestep of a storm

Training settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 512 |
| Early stopping patience | 10 epochs |
| Loss | Multi-step Haversine (km) + 0.1 * wind MAE |

For the full architecture diagram and feature reference see [docs/architecture.md](docs/architecture.md).

---

## Results

Test set: seasons 2020 and later.

| Horizon | Mean error | Median error |
|---|---|---|
| 3 h (direct, step 1) | 8.4 km | 7.0 km |
| 24 h (1 forward pass, 8 steps) | 146 km | 127 km |
| 48 h (2 chained passes, 16 steps) | 405 km | 358 km |

Wind speed MAE: ~3.1 knots

Trajectory plots show predicted (red) vs actual (blue) tracks for short, medium, and long storms from the test set.

Basic plot:

![Trajectory plots](result_images/trajectory_plots.png)

Earth background:

![Trajectory plots on Earth map](result_images/trajectory_plots_earth.png)

Training and validation loss curves:

![Loss curves](result_images/loss_curves.png)

---

## Project Structure

```
data/
  raw/                          # Raw crawled CSV (git-ignored)
  processed/                    # Preprocessed CSV (git-ignored)
docs/
  architecture.md               # Full model architecture and hyperparameters
notebooks/
  train_model.ipynb             # End-to-end training and evaluation notebook
src/
  crawling_data/
    crawler.py                  # Parallel IBTrACS web scraper
  data/
    preprocessor.py             # Full preprocessing pipeline
  db/
    schema.sql                  # PostgreSQL + PostGIS schema
    ingest.py                   # Load processed CSV into PostgreSQL
  model/
    dataset.py                  # Feature engineering, sliding windows, scalers
    transformer.py              # StormTransformer nn.Module
    train.py                    # Training loop, early stopping, checkpoint
    evaluate.py                 # Metrics, trajectory plots, loss curves
models/                         # Saved checkpoints and scalers (git-ignored)
result_images/                  # Output plots
requirements.txt
setup_db.bat                    # One-click database setup (Windows)
```

---

## How to Run

### Prerequisites

- Python 3.10+
- PostgreSQL with PostGIS extension installed
- A CUDA-capable GPU is recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the database connection string. Add this to a `.env` file at the project root or export it in your shell:

```bash
export DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/storm_db
```

---

### Step 1 - Crawl raw data

Downloads storm track records from IBTrACS and saves them to `data/raw/storm_data.csv`.

```bash
python src/crawling_data/crawler.py
```

Uses 10 parallel workers. Takes a few minutes depending on network speed.

---

### Step 2 - Preprocess

Cleans and transforms the raw CSV into an 18-column model-ready file at `data/processed/storm_data.csv`.

```bash
python -m src.data.preprocessor
```

---

### Step 3 - Set up the database

Option A — Automated (Windows):

```
setup_db.bat
```

Creates `storm_db`, enables PostGIS, applies the schema, and ingests the processed CSV.

Option B — Manual:

```bash
psql -U postgres -c "CREATE DATABASE storm_db;"
psql -U postgres -d storm_db -c "CREATE EXTENSION postgis;"
psql postgresql://postgres:yourpassword@localhost:5432/storm_db -f src/db/schema.sql
python -m src.db.ingest
```

Verify the ingestion:

```sql
SELECT COUNT(*) FROM storm_observations;  -- expect ~132,646
SELECT COUNT(*) FROM storms;              -- expect ~2,343
```

---

### Step 4 - Train

```bash
python -m src.model.train
```

Saves the best checkpoint to `models/storm_transformer.pt` and the fitted scalers to `models/scaler_X.pkl` and `models/scaler_y.pkl`.

---

### Step 5 - Evaluate

```bash
python -m src.model.evaluate
```

Prints test metrics (Haversine distance and wind MAE) and saves trajectory plots and loss curves to `models/`.

---

### Notebook

`notebooks/train_model.ipynb` runs the full pipeline interactively: dataset build, training, evaluation, multi-horizon error analysis, and trajectory plots including the Earth-background map view (requires `cartopy>=0.22.0`).

---

## Notes

- Ingestion is idempotent. Re-running `ingest.py` or `setup_db.bat` multiple times is safe.
- Rows without a valid `USA_ATCF_ID` are excluded from the database (about 35K rows).
- `data/raw/`, `data/processed/`, and `models/` are git-ignored. Regenerate them by running Steps 1 to 4.
