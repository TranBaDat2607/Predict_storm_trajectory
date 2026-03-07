"""
Evaluate StormTransformer on the test set.

Usage:
    DATABASE_URL=postgresql://... python -m src.models.evaluate

Outputs:
    - Mean haversine distance (km) vs RF baseline ~875 km
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
    build_datasets, haversine_km, predict_absolute,
    _load_from_db, _engineer_features,
)
from .transformer import StormTransformer

MODELS_DIR = Path(__file__).parents[2] / "models"
CHECKPOINT = MODELS_DIR / "storm_transformer.pt"
BATCH_SIZE = 512


def load_model(device):
    model = StormTransformer().to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_test(model, test_ds, scaler_X, scaler_y, device):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_pred_norm, all_X_last, all_y_true_norm = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_pred_norm.append(pred.cpu().numpy())
            all_X_last.append(X_batch[:, -1, :].cpu().numpy())
            all_y_true_norm.append(y_batch.numpy())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    X_last = np.concatenate(all_X_last, axis=0)
    y_true_norm = np.concatenate(all_y_true_norm, axis=0)

    y_true = scaler_y.inverse_transform(y_true_norm)
    X_last_raw = scaler_X.inverse_transform(X_last)

    lat_t = X_last_raw[:, 0]
    lon_t = X_last_raw[:, 1]
    lat_true = lat_t + y_true[:, 0]
    lon_true = lon_t + y_true[:, 1]

    lat_pred, lon_pred, wind_pred = predict_absolute(pred_norm, X_last, scaler_X, scaler_y)

    hav_km = haversine_km(lat_true, lon_true, lat_pred, lon_pred)
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
        if len(df[df["atcf_id"] == sid]) > SEQ_LEN
    }
    lengths = sorted(storm_lengths.values())
    short_thr = np.percentile(lengths, 25)
    long_thr = np.percentile(lengths, 75)

    chosen = []
    for label, condition in [
        ("short", lambda l: l <= short_thr),
        ("medium", lambda l: short_thr < l < long_thr),
        ("long", lambda l: l >= long_thr),
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
        feats = storm[FEATURE_COLS].values.astype(np.float32)

        # Normalize features
        shape = feats.shape
        feats_norm = scaler_X.transform(feats).astype(np.float32)

        true_lats = storm["lat"].values
        true_lons = storm["lon"].values

        pred_lats, pred_lons = [], []
        for i in range(len(storm) - SEQ_LEN):
            window = torch.tensor(feats_norm[i : i + SEQ_LEN]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(window).cpu().numpy()
            X_last = feats_norm[i + SEQ_LEN - 1 : i + SEQ_LEN]
            lat_p, lon_p, _ = predict_absolute(pred_norm, X_last, scaler_X, scaler_y)
            pred_lats.append(lat_p[0])
            pred_lons.append(lon_p[0])

        # Align: predicted positions correspond to rows SEQ_LEN..end
        ax.plot(true_lons, true_lats, "b-o", markersize=3, label="Actual", linewidth=1.5)
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


def plot_loss_curves():
    """Plot training and validation loss curves from training_log.json."""
    log_path = MODELS_DIR / "training_log.json"
    if not log_path.exists():
        print("No training_log.json found — skipping loss curve plot.")
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    train_mse = [e["train_mse"] for e in log]
    val_mse = [e["val_mse"] for e in log]

    best_epoch = epochs[int(np.argmin(val_mse))]
    best_val = min(val_mse)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MSE curves
    axes[0].plot(epochs, train_mse, label="Train MSE", color="steelblue")
    axes[0].plot(epochs, val_mse, label="Val MSE", color="darkorange")
    axes[0].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch {best_epoch}")
    axes[0].scatter([best_epoch], [best_val], color="green", zorder=5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (normalized)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val haversine
    hav = [e["val_haversine_km"] for e in log]
    axes[1].plot(epochs, hav, color="purple", label="Val Haversine (km)")
    axes[1].axhline(875, color="red", linestyle="--", label="RF baseline 875 km")
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
