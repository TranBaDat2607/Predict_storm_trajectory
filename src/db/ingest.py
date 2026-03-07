"""
Load data/processed/storm_data.csv into PostgreSQL.

Usage:
    # Via environment variable:
    DATABASE_URL=postgresql://user:pass@host/db python -m src.db.ingest

    # Via CLI argument:
    python -m src.db.ingest --db postgresql://user:pass@host/db
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

PROJECT_ROOT = Path(__file__).parents[2]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "storm_data.csv"
CHUNK_SIZE = 5_000

COLUMN_RENAMES = {
    "USA ATCF_ID": "atcf_id",
    "ISO_TIME":    "iso_time",
    "STORM PRES":  "storm_pres",
    "WIND_SPEED":  "wind_speed",
    "STORM SPEED": "storm_speed",
    "STORM DIR":   "storm_dir",
    "USA SSHS":    "usa_sshs",
    "USA POCI":    "usa_poci",
    "USA ROCI":    "usa_roci",
    "USA RMW":     "usa_rmw",
    "DIST2LAND":   "dist2land",
    "LANDFALL":    "landfall",
    "SEASON":      "season",
    "NATURE":      "nature",
    "LAT":         "lat",
    "LON":         "lon",
    "BASIN":       "basin",
    "SUBBASIN":    "subbasin",
}


def _get_dsn() -> str:
    parser = argparse.ArgumentParser(description="Ingest storm CSV into PostgreSQL")
    parser.add_argument("--db", default=None, help="PostgreSQL DSN (overrides DATABASE_URL)")
    args = parser.parse_args()
    dsn = args.db or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: Supply a connection string via --db or DATABASE_URL env var.", file=sys.stderr)
        sys.exit(1)
    return dsn


def _upsert_storms(cur, df: pd.DataFrame) -> tuple[int, int]:
    storms = (
        df[["atcf_id", "season", "basin", "subbasin"]]
        .drop_duplicates(subset="atcf_id")
    )
    inserted = 0
    skipped = 0
    for row in storms.itertuples(index=False):
        cur.execute(
            """
            INSERT INTO storms (atcf_id, season, basin, subbasin)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (atcf_id) DO NOTHING
            """,
            (row.atcf_id, int(row.season), row.basin, row.subbasin or None),
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1
    return inserted, skipped


def _upsert_observations(cur, chunk: pd.DataFrame) -> tuple[int, int]:
    obs_cols = [
        "atcf_id", "iso_time", "lat", "lon", "nature",
        "dist2land", "landfall", "wind_speed", "storm_pres",
        "usa_sshs", "usa_poci", "usa_roci", "usa_rmw",
        "storm_speed", "storm_dir",
    ]
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in chunk[obs_cols].itertuples(index=False)
    ]
    before = cur.rowcount if cur.rowcount != -1 else 0
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO storm_observations
            (atcf_id, iso_time, lat, lon, nature,
             dist2land, landfall, wind_speed, storm_pres,
             usa_sshs, usa_poci, usa_roci, usa_rmw,
             storm_speed, storm_dir)
        VALUES %s
        ON CONFLICT (atcf_id, iso_time) DO NOTHING
        """,
        rows,
        page_size=len(rows),
    )
    inserted = cur.rowcount
    skipped = len(rows) - inserted
    return inserted, skipped


def ingest(dsn: str, csv_path: Path = PROCESSED_CSV):
    print(f"Reading {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns=COLUMN_RENAMES)

    # Ensure iso_time is parsed as datetime
    df["iso_time"] = pd.to_datetime(df["iso_time"], utc=True, errors="coerce")

    # Replace NaN in string cols with None
    for col in ("basin", "subbasin", "atcf_id"):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)

    total_rows = len(df)
    df = df[df["atcf_id"].notna()].copy()
    dropped = total_rows - len(df)
    print(f"Loaded {total_rows:,} rows, dropped {dropped:,} with null atcf_id.")
    print(f"Remaining {len(df):,} rows, {df['atcf_id'].nunique():,} unique storms.")

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                print("\nUpserting storms …")
                s_ins, s_skip = _upsert_storms(cur, df)
                print(f"  Storms  inserted={s_ins:,}  skipped={s_skip:,}")

                print("\nUpserting observations …")
                total_ins = total_skip = 0
                for start in range(0, len(df), CHUNK_SIZE):
                    chunk = df.iloc[start : start + CHUNK_SIZE]
                    ins, skip = _upsert_observations(cur, chunk)
                    total_ins += ins
                    total_skip += skip
                    end = min(start + CHUNK_SIZE, len(df))
                    print(f"  chunk {start:>7,}–{end:>7,}  inserted={ins:,}  skipped={skip:,}")

                print(f"\nDone.  Observations inserted={total_ins:,}  skipped={total_skip:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    ingest(_get_dsn())
