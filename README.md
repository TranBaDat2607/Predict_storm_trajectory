# Predict Storm Trajectory

Predict tropical storm trajectories in the Western Pacific using machine learning.
Data is sourced from the IBTrACS dataset via NCICS.

## Project Structure

```
data/
  raw/               # Raw crawled CSV (git-ignored)
  processed/         # Preprocessed model-ready CSV (git-ignored)
notebooks/
  experiments.ipynb          # Exploratory analysis and modelling
  preprocessing_data.ipynb   # Preprocessing walkthrough
src/
  crawling_data/
    crawler.py       # Web scraper for NCICS IBTrACS
  data/
    loader.py        # CSV loader and column definitions
    preprocessor.py  # Full preprocessing pipeline
    validator.py     # Data validation utilities
  db/
    schema.sql       # PostgreSQL + PostGIS DDL
    ingest.py        # Load processed CSV into PostgreSQL
setup_db.bat         # One-click database setup (Windows)
requirements.txt
```

## Requirements

- Python 3.10+
- PostgreSQL with PostGIS extension installed
- Dependencies: `pip install -r requirements.txt`

---

## Step 1 - Crawl raw data

Fetches storm track data from [NCICS IBTrACS](https://ncics.org/ibtracs/) and saves it to `data/raw/storm_data.csv`.

```bash
python src/crawling_data/crawler.py
```

This runs a parallel crawl with 10 workers. Depending on network speed it may take several minutes.

---

## Step 2 - Preprocess

Cleans and transforms the raw data into a model-ready 18-column CSV at `data/processed/storm_data.csv`.

```bash
python -m src.data.preprocessor
```

---

## Step 3 - Set up the database

### Option A - Automated (Windows)

Run the provided batch file. It will ask for your PostgreSQL password and handle everything:

```
setup_db.bat
```

It will:
1. Create the `storm_db` database
2. Enable the PostGIS extension
3. Apply the schema (`src/db/schema.sql`)
4. Install `psycopg2-binary`
5. Ingest the processed CSV into PostgreSQL

### Option B - Manual

```bash
# Create database and enable PostGIS
psql -U postgres -c "CREATE DATABASE storm_db;"
psql -U postgres -d storm_db -c "CREATE EXTENSION postgis;"

# Apply schema
psql postgresql://postgres:yourpassword@localhost:5432/storm_db -f src/db/schema.sql

# Install driver
pip install psycopg2-binary

# Run ingest
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/storm_db python -m src.db.ingest
```

### Verify

```sql
SELECT COUNT(*) FROM storm_observations;  -- expect ~132,646
SELECT COUNT(*) FROM storms;              -- expect ~2,343
```

### Spatial query example (PostGIS)

```sql
SELECT atcf_id, iso_time, lat, lon,
       ST_Distance(geom, ST_MakePoint(120,15)::GEOGRAPHY) / 1000 AS dist_km
FROM storm_observations
WHERE ST_DWithin(geom, ST_MakePoint(120,15)::GEOGRAPHY, 500000)
ORDER BY dist_km
LIMIT 5;
```

---

## Notes

- Rows without a valid `USA ATCF_ID` are excluded from the database (35,231 rows).
- Running `setup_db.bat` or the ingest script multiple times is safe — duplicate rows are skipped via `ON CONFLICT DO NOTHING`.
- `data/raw/` and `data/processed/` are git-ignored. Re-generate them by running Steps 1 and 2.
