"""
Preprocess raw storm data into a model-ready CSV/Parquet.

Reproduces the full pipeline from notebooks/experiments.ipynb:
  1. Load raw CSV, fix column names, drop units row
  2. Fix ISO_TIME (forward-fill date for time-only rows)
  3. Normalise LON to [-180, 180]
  4. Cast numeric columns
  5. Multi-agency fill for STORM PRES (from TOKYO PRES) + mean impute
  6. Multi-agency fill for WIND_SPEED (from USA WIND) + per-NATURE mean impute
  7. LANDFALL null → -1
  8. NATURE: NR → TS, then label-encode
  9. KNN impute USA POCI (k=12), USA ROCI (k=6), USA RMW (k=21)
 10. ffill remaining STORM SPEED / STORM DIR nulls
 11. Select final 14 model columns
 12. Drop full-row duplicates
 13. Save to data/processed/

Usage:
    python -m src.data.preprocessor
    python src/data/preprocessor.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import NUMERIC_COLS, _parse_iso_time

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "storm_data.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

META_COLS = ["USA ATCF_ID", "ISO_TIME", "BASIN", "SUBBASIN"]

FINAL_COLS = [
    "SEASON", "NATURE", "LAT", "LON", "DIST2LAND", "LANDFALL",
    "WIND_SPEED", "STORM PRES", "USA SSHS", "USA POCI",
    "USA ROCI", "USA RMW", "STORM SPEED", "STORM DIR",
]

# Columns cast to numeric in addition to loader.NUMERIC_COLS
EXTRA_NUMERIC = ["USA SSHS", "USA POCI", "USA ROCI", "USA RMW", "DIST2LAND"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, label: str):
    print(f"\n{'─' * 60}")
    print(f"  {n} / {total}  {label}")
    print(f"{'─' * 60}")


def _count(label: str, n):
    print(f"  {label:<45} {int(n):>8,}")


def _knn_impute(df: pd.DataFrame, feature: str, target: str, k: int) -> pd.DataFrame:
    """Impute nulls in `target` using KNN on `feature`."""
    null_mask = df[target].isna()
    if not null_mask.any():
        return df

    non_null = df[~null_mask]
    X_train = non_null[[feature]].values
    y_train = non_null[target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pred_s = scaler.transform(df.loc[null_mask, [feature]].values)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    df.loc[null_mask, target] = knn.predict(X_pred_s)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    TOTAL = 13
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw CSV
    # ------------------------------------------------------------------
    _step(1, TOTAL, "Load raw CSV")
    raw = pd.read_csv(raw_path, low_memory=False, skiprows=[1])
    _count("Rows read", len(raw))

    # ------------------------------------------------------------------
    # 2. Normalise column names + drop units row
    # ------------------------------------------------------------------
    _step(2, TOTAL, "Normalise column names & drop units row")
    raw.columns = [c.strip().strip("_") for c in raw.columns]
    before = len(raw)
    raw = raw[raw["SEASON"] != "Year"].copy()
    _count("Units rows removed", before - len(raw))
    _count("Rows remaining", len(raw))

    # ------------------------------------------------------------------
    # 3. Cast numeric columns (on full raw df, needed for fill chains)
    # ------------------------------------------------------------------
    _step(3, TOTAL, "Cast numeric columns")
    for col in set(NUMERIC_COLS) | set(EXTRA_NUMERIC):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Also cast agency wind/pres columns used in fill chains
    agency_cols = [
        "WMO WIND", "TOKYO WIND", "CMA WIND", "HKO WIND",
        "NEWDELHI WIND", "TD9636 WIND", "DS824 WIND", "TD9635 WIND",
        "CMA PRES", "NEWDELHI PRES", "DS824 PRES", "TD9635 PRES",
        "WMO PRES",
    ]
    for col in agency_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ------------------------------------------------------------------
    # 4. Fix ISO_TIME — forward-fill date for time-only rows
    # ------------------------------------------------------------------
    _step(4, TOTAL, "Fix ISO_TIME")
    if "ISO_TIME" in raw.columns:
        before_nat = raw["ISO_TIME"].isna().sum()
        raw["ISO_TIME"] = _parse_iso_time(raw["ISO_TIME"].astype(str))
        after_nat = raw["ISO_TIME"].isna().sum()
        _count("Time-only rows repaired", int(before_nat) - int(after_nat))
        _count("Still unparseable (NaT)", int(after_nat))

    # ------------------------------------------------------------------
    # 5. Normalise LON to [-180, 180]
    # ------------------------------------------------------------------
    _step(5, TOTAL, "Normalise LON")
    if "LON" in raw.columns:
        shifted = (raw["LON"] > 180).sum()
        raw["LON"] = raw["LON"].where(raw["LON"] <= 180, raw["LON"] - 360)
        _count("LON values shifted (0-360 → ±180)", int(shifted))

    # ------------------------------------------------------------------
    # 6. STORM PRES — multi-agency fill then mean impute
    #    Source priority: TOKYO PRES → USA PRES → CMA PRES →
    #                     NEWDELHI PRES → DS824 PRES → TD9635 PRES → mean
    # ------------------------------------------------------------------
    _step(6, TOTAL, "Build STORM PRES (multi-agency fill)")
    raw["STORM PRES"] = raw["TOKYO PRES"].copy() if "TOKYO PRES" in raw.columns else np.nan
    for donor in ["USA PRES", "CMA PRES", "NEWDELHI PRES", "DS824 PRES", "TD9635 PRES"]:
        if donor in raw.columns:
            raw["STORM PRES"] = raw["STORM PRES"].fillna(raw[donor])
    raw["STORM PRES"] = pd.to_numeric(raw["STORM PRES"], errors="coerce")
    remaining = raw["STORM PRES"].isna().sum()
    _count("Nulls after agency fill", int(remaining))
    raw["STORM PRES"] = raw["STORM PRES"].fillna(raw["STORM PRES"].mean())
    _count("Nulls after mean impute", int(raw["STORM PRES"].isna().sum()))

    # ------------------------------------------------------------------
    # 7. WIND_SPEED — multi-agency fill then per-NATURE mean impute
    #    Source priority: USA WIND → WMO WIND → TOKYO WIND → CMA WIND →
    #                     HKO WIND → NEWDELHI WIND → TD9636 WIND →
    #                     DS824 WIND → TD9635 WIND → mean by NATURE
    # ------------------------------------------------------------------
    _step(7, TOTAL, "Build WIND_SPEED (multi-agency fill)")
    raw["WIND_SPEED"] = raw["USA WIND"].copy() if "USA WIND" in raw.columns else np.nan
    for donor in [
        "WMO WIND", "TOKYO WIND", "CMA WIND", "HKO WIND",
        "NEWDELHI WIND", "TD9636 WIND", "DS824 WIND", "TD9635 WIND",
    ]:
        if donor in raw.columns:
            raw["WIND_SPEED"] = raw["WIND_SPEED"].fillna(raw[donor])
    raw["WIND_SPEED"] = pd.to_numeric(raw["WIND_SPEED"], errors="coerce")
    _count("Nulls after agency fill", int(raw["WIND_SPEED"].isna().sum()))

    # Per-NATURE mean impute for remaining nulls
    for nature_val in raw["NATURE"].dropna().unique():
        mask_null = raw["WIND_SPEED"].isna() & (raw["NATURE"] == nature_val)
        if mask_null.any():
            nature_mean = raw.loc[raw["NATURE"] == nature_val, "WIND_SPEED"].mean()
            if not pd.isna(nature_mean):
                raw.loc[mask_null, "WIND_SPEED"] = nature_mean
    _count("Nulls after per-NATURE mean impute", int(raw["WIND_SPEED"].isna().sum()))

    # ------------------------------------------------------------------
    # 8. LANDFALL null → -1  (null means storm dissipated at sea)
    # ------------------------------------------------------------------
    _step(8, TOTAL, "LANDFALL null → -1")
    if "LANDFALL" in raw.columns:
        n_null = raw["LANDFALL"].isna().sum()
        raw["LANDFALL"] = raw["LANDFALL"].fillna(-1)
        _count("LANDFALL nulls filled with -1", int(n_null))

    # ------------------------------------------------------------------
    # 9. NATURE encoding: NR → TS, then LabelEncoder
    #    Mapping: DS=0, ET=1, MX=2, SS=3, TS=4
    # ------------------------------------------------------------------
    _step(9, TOTAL, "Encode NATURE")
    if "NATURE" in raw.columns:
        n_nr = (raw["NATURE"] == "NR").sum()
        raw["NATURE"] = raw["NATURE"].replace("NR", "TS")
        _count("NR rows remapped to TS", int(n_nr))
        le = LabelEncoder()
        raw["NATURE"] = le.fit_transform(raw["NATURE"].astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  Label mapping: {mapping}")

    # ------------------------------------------------------------------
    # 10. KNN impute USA POCI (k=12), USA ROCI (k=6), USA RMW (k=21)
    #     Feature: USA SSHS
    # ------------------------------------------------------------------
    _step(10, TOTAL, "KNN impute USA POCI / ROCI / RMW")
    for target, k in [("USA POCI", 12), ("USA ROCI", 6), ("USA RMW", 21)]:
        if target in raw.columns and "USA SSHS" in raw.columns:
            before_null = raw[target].isna().sum()
            # Drop rows where the feature itself is null (can't impute)
            valid = raw[raw["USA SSHS"].notna()].copy()
            raw = _knn_impute(raw, feature="USA SSHS", target=target, k=k)
            after_null = raw[target].isna().sum()
            _count(f"{target} nulls imputed (k={k})", int(before_null) - int(after_null))

    # ------------------------------------------------------------------
    # 11. ffill STORM SPEED and STORM DIR (remaining nulls within storm)
    # ------------------------------------------------------------------
    _step(11, TOTAL, "ffill STORM SPEED / STORM DIR")
    for col in ["STORM SPEED", "STORM DIR"]:
        if col in raw.columns:
            before_null = raw[col].isna().sum()
            raw[col] = raw[col].ffill()
            _count(f"{col} nulls filled", int(before_null) - int(raw[col].isna().sum()))

    # ------------------------------------------------------------------
    # 12. Select final columns and drop full-row duplicates
    # ------------------------------------------------------------------
    _step(12, TOTAL, "Select final columns & drop duplicates")
    present = [c for c in META_COLS + FINAL_COLS if c in raw.columns]
    missing = [c for c in META_COLS + FINAL_COLS if c not in raw.columns]
    if missing:
        print(f"  WARNING — columns not found, skipped: {missing}")
    df = raw[present].copy()
    before_dedup = len(df)
    df = df[~df.duplicated(keep="first")].reset_index(drop=True)
    _count("Duplicate rows removed", before_dedup - len(df))
    _count("Final row count", len(df))

    # ------------------------------------------------------------------
    # 13. Save
    # ------------------------------------------------------------------
    _step(13, TOTAL, "Save")
    csv_path = out_dir / "storm_data.csv"
    parquet_path = out_dir / "storm_data.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"\n  Saved → {csv_path.relative_to(PROJECT_ROOT)}")
        print(f"  Saved → {parquet_path.relative_to(PROJECT_ROOT)}")
    except Exception:
        print(f"\n  Saved → {csv_path.relative_to(PROJECT_ROOT)}")
        print("  (Parquet skipped — install pyarrow to enable)")

    print(f"\n{'=' * 60}")
    print(f"  Done.  Final shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null counts:\n{df.isnull().sum().to_string()}")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    preprocess()
