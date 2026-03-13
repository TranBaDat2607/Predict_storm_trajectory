"""
Evaluate StormTransformer on the test set.

Usage:
    DATABASE_URL=postgresql://... python -m src.model.evaluate

Outputs:
    - Mean haversine distance (km)
    - Wind speed MAE (knots)
    - Trajectory plots for 3 test storms
    - Training/validation loss curve
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    SEQ_LEN, N_FEATURES, FEATURE_COLS, TARGET_COLS,
    BASIN_MAP, N_BASINS, N_ROLLOUT_STEPS,
    build_datasets, haversine_km, predict_absolute,
    _load_from_db, _engineer_features,
)
from .transformer import StormTransformer

MODELS_DIR = Path(__file__).parents[2] / "models"
CHECKPOINT = MODELS_DIR / "storm_transformer.pt"
BATCH_SIZE = 512


def load_model(device):
    model = StormTransformer(n_basins=N_BASINS).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_test(model, test_ds, scaler_X, scaler_y, device):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_pred_norm, all_X_last, all_y_true_norm = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, ctx_batch, mask_batch in loader:
            X_batch    = X_batch.to(device)
            ctx_batch  = ctx_batch.to(device)
            mask_batch = mask_batch.to(device)
            pred = model(X_batch, ctx_batch, mask=mask_batch)   # [B, 8, 3]
            all_pred_norm.append(pred[:, 0, :].cpu().numpy())   # extract step-1
            all_X_last.append(X_batch[:, -1, :].cpu().numpy())
            all_y_true_norm.append(y_batch[:, 0, :].numpy())    # step-1 target

    pred_norm  = np.concatenate(all_pred_norm, axis=0)
    X_last     = np.concatenate(all_X_last, axis=0)
    y_true_norm = np.concatenate(all_y_true_norm, axis=0)

    y_true     = scaler_y.inverse_transform(y_true_norm)
    X_last_raw = scaler_X.inverse_transform(X_last)

    lat_t    = X_last_raw[:, 0]
    lon_t    = X_last_raw[:, 1]
    lat_true = lat_t + y_true[:, 0]
    lon_true = lon_t + y_true[:, 1]

    lat_pred, lon_pred, wind_pred = predict_absolute(pred_norm, X_last, scaler_X, scaler_y)

    hav_km   = haversine_km(lat_true, lon_true, lat_pred, lon_pred)
    wind_mae = np.abs(wind_pred - y_true[:, 2]).mean()

    print("\n=== Test Set Metrics ===")
    print(f"  Mean Haversine distance : {hav_km.mean():.1f} km")
    print(f"  Median Haversine        : {np.median(hav_km):.1f} km")
    print(f"  Wind speed MAE          : {wind_mae:.2f} knots")

    return hav_km, wind_mae


def plot_trajectories(scaler_X, scaler_y, device):
    """Plot predicted vs actual trajectories for 3 test storms."""
    df = _load_from_db()
    df = _engineer_features(df)
    test_storms = df[df["season"] >= 2020]["atcf_id"].unique()

    # Pick short, medium, long storms
    storm_lengths = {
        sid: len(df[df["atcf_id"] == sid])
        for sid in test_storms
        if len(df[df["atcf_id"] == sid]) > 1
    }
    lengths = sorted(storm_lengths.values())
    short_thr = np.percentile(lengths, 25)
    long_thr  = np.percentile(lengths, 75)

    chosen = []
    for label, condition in [
        ("short",  lambda l: l <= short_thr),
        ("medium", lambda l: short_thr < l < long_thr),
        ("long",   lambda l: l >= long_thr),
    ]:
        candidates = [sid for sid, l in storm_lengths.items() if condition(l)]
        if candidates:
            chosen.append((label, candidates[0]))
        if len(chosen) == 3:
            break

    model = load_model(device)
    fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, (label, sid) in zip(axes, chosen):
        storm = df[df["atcf_id"] == sid].reset_index(drop=True)
        feats      = storm[FEATURE_COLS].values.astype(np.float32)
        feats_norm = scaler_X.transform(feats).astype(np.float32)

        # Build context tensor for this storm
        basin_id    = int(storm["basin_id"].iloc[0])
        season_norm = float(storm["season_norm"].iloc[0])
        ctx_tensor  = torch.tensor([[basin_id, season_norm]], dtype=torch.float32).to(device)

        true_lats = storm["lat"].values
        true_lons = storm["lon"].values

        pred_lats, pred_lons = [], []
        for k in range(1, len(storm)):
            real_start = max(0, k - SEQ_LEN)
            window_raw = feats_norm[real_start:k]
            real_len   = len(window_raw)
            pad_len    = SEQ_LEN - real_len
            if pad_len > 0:
                pad = np.zeros((pad_len, N_FEATURES), dtype=np.float32)
                window_raw = np.concatenate([pad, window_raw], axis=0)
            mask_np = np.array([True]*pad_len + [False]*real_len, dtype=bool)
            mask_t  = torch.tensor(mask_np).unsqueeze(0).to(device)
            window  = torch.tensor(window_raw).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm_t = model(window, ctx_tensor, mask=mask_t)[:, 0, :].cpu().numpy()
            X_last = feats_norm[k-1 : k]
            lat_p, lon_p, _ = predict_absolute(pred_norm_t, X_last, scaler_X, scaler_y)
            pred_lats.append(lat_p[0])
            pred_lons.append(lon_p[0])

        ax.plot(true_lons, true_lats, "b-o", markersize=3, label="Actual",    linewidth=1.5)
        ax.plot(pred_lons, pred_lats, "r--s", markersize=3, label="Predicted", linewidth=1.5)
        ax.set_title(f"{sid} ({label})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Storm Trajectory: Predicted vs Actual", fontsize=13)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "trajectory_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Trajectory plot saved to {MODELS_DIR}/trajectory_plots.png")


def plot_trajectories_earth(scaler_X, scaler_y, device):
    """
    Plot predicted vs actual trajectories on a real Earth map background.
    Uses cartopy PlateCarree projection with NASA Blue Marble stock image.
    Saves to models/trajectory_plots_earth.png.
    Requires: pip install cartopy>=0.22.0
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:
        raise ImportError(
            "cartopy is required for plot_trajectories_earth(). "
            "Install it with: pip install cartopy>=0.22.0"
        ) from e

    df = _load_from_db()
    df = _engineer_features(df)
    test_storms = df[df["season"] >= 2020]["atcf_id"].unique()

    storm_lengths = {
        sid: len(df[df["atcf_id"] == sid])
        for sid in test_storms
        if len(df[df["atcf_id"] == sid]) > 1
    }
    lengths = sorted(storm_lengths.values())
    short_thr = np.percentile(lengths, 25)
    long_thr  = np.percentile(lengths, 75)

    chosen = []
    for label, condition in [
        ("short",  lambda l: l <= short_thr),
        ("medium", lambda l: short_thr < l < long_thr),
        ("long",   lambda l: l >= long_thr),
    ]:
        candidates = [sid for sid, l in storm_lengths.items() if condition(l)]
        if candidates:
            chosen.append((label, candidates[0]))
        if len(chosen) == 3:
            break

    model = load_model(device)
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, len(chosen),
        figsize=(8 * len(chosen), 6),
        subplot_kw={"projection": projection},
    )
    if len(chosen) == 1:
        axes = [axes]

    for ax, (label, sid) in zip(axes, chosen):
        storm = df[df["atcf_id"] == sid].reset_index(drop=True)
        feats      = storm[FEATURE_COLS].values.astype(np.float32)
        feats_norm = scaler_X.transform(feats).astype(np.float32)

        basin_id    = int(storm["basin_id"].iloc[0])
        season_norm = float(storm["season_norm"].iloc[0])
        ctx_tensor  = torch.tensor([[basin_id, season_norm]], dtype=torch.float32).to(device)

        true_lats = storm["lat"].values
        true_lons = storm["lon"].values

        pred_lats, pred_lons = [], []
        for k in range(1, len(storm)):
            real_start = max(0, k - SEQ_LEN)
            window_raw = feats_norm[real_start:k]
            real_len   = len(window_raw)
            pad_len    = SEQ_LEN - real_len
            if pad_len > 0:
                pad = np.zeros((pad_len, N_FEATURES), dtype=np.float32)
                window_raw = np.concatenate([pad, window_raw], axis=0)
            mask_np = np.array([True]*pad_len + [False]*real_len, dtype=bool)
            mask_t  = torch.tensor(mask_np).unsqueeze(0).to(device)
            window  = torch.tensor(window_raw).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm_t = model(window, ctx_tensor, mask=mask_t)[:, 0, :].cpu().numpy()
            X_last = feats_norm[k-1:k]
            lat_p, lon_p, _ = predict_absolute(pred_norm_t, X_last, scaler_X, scaler_y)
            pred_lats.append(lat_p[0])
            pred_lons.append(lon_p[0])

        # Auto-compute map extent with 5° padding, clamped to valid bounds
        all_lats = np.concatenate([true_lats, pred_lats])
        all_lons = np.concatenate([true_lons, pred_lons])
        pad_deg  = 5.0
        extent = [
            max(float(all_lons.min()) - pad_deg, -180.0),
            min(float(all_lons.max()) + pad_deg,  180.0),
            max(float(all_lats.min()) - pad_deg,  -90.0),
            min(float(all_lats.max()) + pad_deg,   90.0),
        ]
        ax.set_extent(extent, crs=projection)

        ax.stock_img()  # NASA Blue Marble raster background
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="white", linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.4, color="white", alpha=0.5, linestyle="--")

        ax.plot(true_lons, true_lats, "b-o",  markersize=3, linewidth=1.5,
                label="Actual",    transform=projection)
        ax.plot(pred_lons, pred_lats, "r--s", markersize=3, linewidth=1.5,
                label="Predicted", transform=projection)

        ax.set_title(f"{sid} ({label})", color="white", fontsize=11, pad=8)
        ax.legend(fontsize=8, loc="lower left",
                  facecolor="black", labelcolor="white", framealpha=0.6)

    plt.suptitle("Storm Trajectory: Predicted vs Actual (Earth Background)", fontsize=13)
    plt.tight_layout()
    out_path = MODELS_DIR / "trajectory_plots_earth.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Earth trajectory plot saved to {out_path}")


def plot_loss_curves():
    """Plot training and validation loss curves from training_log.json."""
    log_path = MODELS_DIR / "training_log.json"
    if not log_path.exists():
        print("No training_log.json found - skipping loss curve plot.")
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]

    # Support both old (MSE) and new (km) log formats
    if "train_loss_km" in log[0]:
        train_loss = [e["train_loss_km"] for e in log]
        val_loss   = [e["val_loss_km"] for e in log]
        loss_label = "Haversine loss (km)"
    else:
        train_loss = [e["train_mse"] for e in log]
        val_loss   = [e["val_mse"] for e in log]
        loss_label = "MSE (normalized)"

    best_epoch = epochs[int(np.argmin(val_loss))]
    best_val   = min(val_loss)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label=f"Train {loss_label}", color="steelblue")
    axes[0].plot(epochs, val_loss,   label=f"Val {loss_label}",   color="darkorange")
    axes[0].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch {best_epoch}")
    axes[0].scatter([best_epoch], [best_val], color="green", zorder=5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(loss_label)
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hav = [e["val_haversine_km"] for e in log]
    axes[1].plot(epochs, hav, color="purple", label="Val Haversine (km)")
    axes[1].axvline(best_epoch, color="green", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Haversine distance (km)")
    axes[1].set_title("Validation Position Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Loss curve saved to {MODELS_DIR}/loss_curves.png")


def evaluate():
    assert "DATABASE_URL" in os.environ, "Set DATABASE_URL env var first."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}. Run train.py first.")

    scaler_X = joblib.load(MODELS_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")

    _, _, test_ds, _, _ = build_datasets(save_scalers=False)

    model = load_model(device)
    evaluate_test(model, test_ds, scaler_X, scaler_y, device)
    plot_trajectories(scaler_X, scaler_y, device)
    plot_loss_curves()


if __name__ == "__main__":
    evaluate()
