import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "storm_data.csv"

NUMERIC_COLS = [
    "LAT", "LON", "DIST2LAND", "LANDFALL",
    "USA WIND", "USA PRES", "STORM SPEED", "STORM DIR", "SEASON",
]


def _parse_iso_time(series: pd.Series) -> pd.Series:
    """Parse ISO_TIME, forward-filling the date for time-only rows.

    The raw CSV omits the date for intra-day rows, storing only a time
    string (e.g. '21:00:00').  We forward-fill the date from the last row
    that carried a full datetime before parsing.
    """
    # First pass: parse what pandas can handle directly
    parsed = pd.to_datetime(series, errors="coerce")

    # Build a date string from valid rows and forward-fill gaps
    date_str = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna())
    date_str = date_str.ffill()

    # Rows that are still NaT but have a raw time string → combine + reparse
    nat_mask = parsed.isna() & series.notna() & series.str.strip().ne("")
    if nat_mask.any():
        combined = date_str[nat_mask] + " " + series[nat_mask].str.strip()
        parsed = parsed.copy()
        parsed.loc[nat_mask] = pd.to_datetime(combined, errors="coerce")

    return parsed


def load_storm_data(path=DATA_PATH):
    """Load and clean storm_data.csv.

    Returns a DataFrame with:
    - Stripped column names (no trailing underscores/whitespace)
    - Units row removed
    - Core numeric columns cast to float
    - ISO_TIME parsed as datetime (date forward-filled for time-only rows)
    - LON normalised to [-180, 180]
    """
    df = pd.read_csv(path, low_memory=False, skiprows=[1])

    # Normalise column names: strip surrounding whitespace and underscores
    df.columns = [c.strip().strip("_") for c in df.columns]

    # Guard: drop any residual units row (SEASON == "Year")
    df = df[df["SEASON"] != "Year"].copy()

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise LON from 0-360 → [-180, 180]
    if "LON" in df.columns:
        df["LON"] = df["LON"].where(df["LON"] <= 180, df["LON"] - 360)

    # Parse timestamp — forward-fill date for time-only rows
    if "ISO_TIME" in df.columns:
        df["ISO_TIME"] = _parse_iso_time(df["ISO_TIME"])

    # Summary
    n_rows = len(df)
    date_range_str = (
        f"{df['ISO_TIME'].min()} → {df['ISO_TIME'].max()}"
        if "ISO_TIME" in df.columns else "N/A"
    )
    n_storms = df["USA ATCF_ID"].nunique() if "USA ATCF_ID" in df.columns else "N/A"

    print(f"Loaded        : {n_rows:,} rows")
    print(f"Date range    : {date_range_str}")
    print(f"Unique storms : {n_storms}")

    return df
