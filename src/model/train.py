"""
Training loop for StormTransformer.

Usage:
    DATABASE_URL=postgresql://... python -m src.models.train

Saves:
    models/storm_transformer.pt   — best checkpoint (lowest val MSE)
    models/training_log.json      — per-epoch metrics
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import build_datasets, haversine_km, predict_absolute
from .transformer import StormTransformer, count_parameters

MODELS_DIR = Path(__file__).parents[2] / "models"

# Hyperparameters
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
T_MAX = 100
ETA_MIN = 1e-5


def train():
    assert "DATABASE_URL" in os.environ, "Set DATABASE_URL env var first."

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    # AMP requires PyTorch compiled for the exact GPU arch (e.g. nightly for Blackwell sm_120).
    # Disable AMP if CUDA is available but arch support is uncertain.
    use_amp = use_cuda and torch.cuda.get_device_capability(0)[0] < 12
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {device} — {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    else:
        print(f"Device: {device} (no CUDA GPU found)")

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, scaler_X, scaler_y = build_datasets(save_scalers=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = StormTransformer().to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Automatic mixed precision — only enabled when GPU arch is fully supported
    scaler_amp = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"AMP enabled    : {use_amp}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_losses.append(loss.item())

        train_mse = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_pred_norm = []
        all_X_last = []
        all_y_true_norm = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                all_pred_norm.append(pred.cpu().numpy())
                all_X_last.append(X_batch[:, -1, :].cpu().numpy())
                all_y_true_norm.append(y_batch.cpu().numpy())

        val_mse = np.mean(val_losses)

        # Haversine on val set
        pred_norm = np.concatenate(all_pred_norm, axis=0)
        X_last = np.concatenate(all_X_last, axis=0)
        y_true_norm = np.concatenate(all_y_true_norm, axis=0)
        y_true = scaler_y.inverse_transform(y_true_norm)
        X_last_raw = scaler_X.inverse_transform(X_last)

        lat_t = X_last_raw[:, 0]
        lon_t = X_last_raw[:, 1]
        lat_true = lat_t + y_true[:, 0]
        lon_true = lon_t + y_true[:, 1]

        lat_pred, lon_pred, _ = predict_absolute(pred_norm, X_last, scaler_X, scaler_y)
        val_haversine = haversine_km(lat_true, lon_true, lat_pred, lon_pred).mean()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val Haversine: {val_haversine:.1f} km | "
            f"LR: {current_lr:.3e} | "
            f"Time: {elapsed:.1f}s"
        )

        log.append({
            "epoch": epoch,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "val_haversine_km": float(val_haversine),
            "lr": float(current_lr),
        })

        # Checkpoint
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "storm_transformer.pt")
            print(f"  --> Saved checkpoint (val MSE {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

        scheduler.step()

    # Save training log
    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f}")
    print(f"Log saved to {MODELS_DIR}/training_log.json")

    return model, log


if __name__ == "__main__":
    train()
