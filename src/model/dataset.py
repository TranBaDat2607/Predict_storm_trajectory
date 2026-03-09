"""
Feature engineering, normalization, sliding-window Dataset for storm trajectory prediction.

Data is loaded from PostgreSQL via DATABASE_URL env var.
16 input features, 3 targets (d_lat, d_lon, wind_speed at t+1).
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

SEQ_LEN = 8
N_FEATURES = 16
N_TARGETS = 3
N_ROLLOUT_STEPS = 2

MODELS_DIR = Path(__file__).parents[2] / "models"

# Nature label encoding (consistent with existing preprocessor)
NATURE_MAP = {"DS": 0, "ET": 1, "MX": 2, "NR": 3, "SS": 4, "TS": 5}

# Basin label encoding
BASIN_MAP = {"EP": 0, "NA": 1, "NI": 2, "SA": 3, "SI": 4, "SP": 5, "WP": 6}
N_BASINS = 7

SQL = """
SELECT
    s.atcf_id, s.season, s.basin,
    o.iso_time, o.lat, o.lon, o.nature,
    o.dist2land, o.landfall, o.wind_speed, o.storm_pres,
    o.usa_sshs, o.usa_poci, o.usa_roci, o.usa_rmw,
    o.storm_speed, o.storm_dir
FROM storm_observations o
JOIN storms s ON o.atcf_id = s.atcf_id
ORDER BY o.atcf_id, o.iso_time
"""


def _load_from_db() -> pd.DataFrame:
    dsn = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(dsn)
    df = pd.read_sql(SQL, conn)
    conn.close()
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add d_lat, d_lon, cyclical storm_dir, basin_id, season_norm."""
    df = df.copy()
    df = df.sort_values(["atcf_id", "iso_time"]).reset_index(drop=True)

    # Label-encode nature
    df["nature"] = df["nature"].map(NATURE_MAP).fillna(0).astype(int)

    # Basin and season context
    df["basin_id"] = df["basin"].map(BASIN_MAP).fillna(0).astype(int)
    df["season_norm"] = ((df["season"] - 1980) / 40.0).astype(np.float32)

    # Delta lat/lon within each storm (0 at storm start)
    df["d_lat"] = df.groupby("atcf_id")["lat"].diff().fillna(0.0)
    df["d_lon"] = df.groupby("atcf_id")["lon"].diff().fillna(0.0)

    # Cyclical storm_dir
    rad = np.radians(df["storm_dir"].fillna(0.0))
    df["storm_dir_sin"] = np.sin(rad)
    df["storm_dir_cos"] = np.cos(rad)

    # Fill remaining NaNs with 0
    numeric_cols = [
        "lat", "lon", "d_lat", "d_lon", "nature",
        "dist2land", "landfall", "wind_speed", "storm_pres",
        "usa_sshs", "usa_poci", "usa_roci", "usa_rmw",
        "storm_speed", "storm_dir_sin", "storm_dir_cos",
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


FEATURE_COLS = [
    "lat", "lon", "d_lat", "d_lon", "nature",
    "dist2land", "landfall", "wind_speed", "storm_pres",
    "usa_sshs", "usa_poci", "usa_roci", "usa_rmw",
    "storm_speed", "storm_dir_sin", "storm_dir_cos",
]
TARGET_COLS = ["d_lat", "d_lon", "wind_speed"]


def _make_windows(df: pd.DataFrame, storm_ids):
    """
    Build sliding windows of length SEQ_LEN.

    Returns:
        X   [n_windows, SEQ_LEN, N_FEATURES]
        y   [n_windows, N_ROLLOUT_STEPS, N_TARGETS]
        ctx [n_windows, 2]  — (basin_id, season_norm)
    """
    X_list, y_list, ctx_list = [], [], []
    for sid in storm_ids:
        storm = df[df["atcf_id"] == sid]
        if len(storm) <= SEQ_LEN + N_ROLLOUT_STEPS - 1:
            continue
        feats = storm[FEATURE_COLS].values.astype(np.float32)
        targets = storm[TARGET_COLS].values.astype(np.float32)
        basin_id = int(storm["basin_id"].iloc[0])
        season_norm = float(storm["season_norm"].iloc[0])
        for i in range(len(storm) - SEQ_LEN - N_ROLLOUT_STEPS + 1):
            X_list.append(feats[i : i + SEQ_LEN])
            y_list.append(targets[i + SEQ_LEN : i + SEQ_LEN + N_ROLLOUT_STEPS])  # [K, 3]
            ctx_list.append([basin_id, season_norm])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    ctx = np.array(ctx_list, dtype=np.float32)
    return X, y, ctx


class StormWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ctx: np.ndarray):
        self.X = torch.from_numpy(X)      # [N, SEQ_LEN, N_FEATURES]
        self.y = torch.from_numpy(y)      # [N, N_ROLLOUT_STEPS, N_TARGETS]
        self.ctx = torch.from_numpy(ctx)  # [N, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ctx[idx]


def build_datasets(save_scalers: bool = True):
    """
    Load data from DB, engineer features, split by season, fit scalers on train only.

    Returns:
        train_ds, val_ds, test_ds : StormWindowDataset
        scaler_X, scaler_y        : fitted StandardScaler objects
    """
    print("Loading data from database…")
    df = _load_from_db()
    print(f"  Loaded {len(df):,} rows, {df['atcf_id'].nunique():,} unique storms.")

    df = _engineer_features(df)

    # Season-based splits
    train_mask = df["season"] <= 2014
    val_mask = (df["season"] >= 2015) & (df["season"] <= 2019)
    test_mask = df["season"] >= 2020

    train_ids = df.loc[train_mask, "atcf_id"].unique()
    val_ids = df.loc[val_mask, "atcf_id"].unique()
    test_ids = df.loc[test_mask, "atcf_id"].unique()

    print(f"  Train storms: {len(train_ids):,} | Val: {len(val_ids):,} | Test: {len(test_ids):,}")

    X_train, y_train, ctx_train = _make_windows(df, train_ids)
    X_val, y_val, ctx_val = _make_windows(df, val_ids)
    X_test, y_test, ctx_test = _make_windows(df, test_ids)

    print(f"  Windows — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Log basin distribution
    for split_name, ctx_arr in [("Train", ctx_train), ("Val", ctx_val), ("Test", ctx_test)]:
        counts = np.bincount(ctx_arr[:, 0].astype(int), minlength=N_BASINS)
        inv_map = {v: k for k, v in BASIN_MAP.items()}
        dist = {inv_map[i]: int(counts[i]) for i in range(N_BASINS) if counts[i] > 0}
        print(f"  {split_name} basin distribution: {dist}")

    # Fit scalers on train only
    n_train, seq, n_feat = X_train.shape
    scaler_X = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, n_feat))

    # Fit y scaler on step-1 train targets only
    scaler_y = StandardScaler()
    scaler_y.fit(y_train[:, 0, :])

    def scale_X(X):
        shape = X.shape
        return scaler_X.transform(X.reshape(-1, n_feat)).reshape(shape).astype(np.float32)

    def scale_y(y):
        # y: [N, N_ROLLOUT_STEPS, 3]
        n, k, t = y.shape
        return scaler_y.transform(y.reshape(-1, t)).reshape(n, k, t).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)
    y_test_s = scale_y(y_test)

    if save_scalers:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_X, MODELS_DIR / "scaler_X.pkl")
        joblib.dump(scaler_y, MODELS_DIR / "scaler_y.pkl")
        print(f"  Scalers saved to {MODELS_DIR}/")

    train_ds = StormWindowDataset(X_train_s, y_train_s, ctx_train)
    val_ds = StormWindowDataset(X_val_s, y_val_s, ctx_val)
    test_ds = StormWindowDataset(X_test_s, y_test_s, ctx_test)

    return train_ds, val_ds, test_ds, scaler_X, scaler_y


def predict_absolute(pred_norm: np.ndarray, X_last: np.ndarray, scaler_X, scaler_y):
    """
    Convert normalized predictions back to absolute lat/lon + wind.

    Args:
        pred_norm : [N, 3] normalized (d_lat, d_lon, wind)
        X_last    : [N, N_FEATURES] the last timestep of each window (normalized)
        scaler_X  : fitted StandardScaler for features
        scaler_y  : fitted StandardScaler for targets

    Returns:
        lat_pred  : [N]
        lon_pred  : [N]
        wind_pred : [N]
    """
    pred = scaler_y.inverse_transform(pred_norm)  # [N, 3]
    X_raw = scaler_X.inverse_transform(X_last)    # [N, N_FEATURES]

    lat_t = X_raw[:, 0]  # col 0 = lat
    lon_t = X_raw[:, 1]  # col 1 = lon

    lat_pred = lat_t + pred[:, 0]
    lon_pred = lon_t + pred[:, 1]
    wind_pred = pred[:, 2]

    return lat_pred, lon_pred, wind_pred


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
