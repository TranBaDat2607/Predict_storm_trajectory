"""
Training loop for StormTransformer.

Usage:
    DATABASE_URL=postgresql://... python -m src.model.train

Saves:
    models/storm_transformer.pt   — best checkpoint (lowest val Haversine km)
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

# Loss weights
LAMBDA_WIND = 0.1      # wind MAE weight relative to Haversine km loss

# Autoregressive rollout
N_ROLLOUT_STEPS = 2    # must match dataset.N_ROLLOUT_STEPS
ROLLOUT_LAMBDA = 0.5   # max weight for step-2 loss; ramped from 0 over first 20 epochs


class HaversineLoss(nn.Module):
    """
    Differentiable Haversine loss in km + weighted wind MAE.

    Stores scaler mean/std as registered buffers so they move to device automatically.
    Denormalization is fully differentiable (affine ops + clamped asin).
    """

    R = 6371.0

    def __init__(self, scaler_X, scaler_y, lambda_wind: float = 0.1):
        super().__init__()
        self.register_buffer("y_mean", torch.tensor(scaler_y.mean_, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(scaler_y.scale_, dtype=torch.float32))
        self.register_buffer("X_mean", torch.tensor(scaler_X.mean_, dtype=torch.float32))
        self.register_buffer("X_std",  torch.tensor(scaler_X.scale_, dtype=torch.float32))
        self.lambda_wind = lambda_wind

    def _denorm_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean          # [B, 3]

    def _raw_lat(self, X_last_norm):
        return X_last_norm[:, 0] * self.X_std[0] + self.X_mean[0]  # [B]

    def forward(self, pred_norm, y_norm, X_last_norm):
        pred_raw = self._denorm_y(pred_norm)
        true_raw = self._denorm_y(y_norm)
        lat_t    = self._raw_lat(X_last_norm)

        lat2_pred = torch.deg2rad(lat_t + pred_raw[:, 0])
        lat2_true = torch.deg2rad(lat_t + true_raw[:, 0])
        dlon_err  = torch.deg2rad(pred_raw[:, 1] - true_raw[:, 1])
        dlat      = lat2_pred - lat2_true

        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat2_true) * torch.cos(lat2_pred) * torch.sin(dlon_err / 2) ** 2)
        hav_km = 2 * self.R * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
        wind_mae = torch.abs(pred_raw[:, 2] - true_raw[:, 2]).mean()
        return hav_km.mean() + self.lambda_wind * wind_mae

    def build_next_row(self, X_last_norm, pred_norm):
        """Construct next normalized feature row from last row + prediction (differentiable)."""
        pred_raw = self._denorm_y(pred_norm)               # [B, 3]
        X_raw    = X_last_norm * self.X_std + self.X_mean  # [B, 16]
        new_row  = X_raw.clone()
        new_row[:, 0] = X_raw[:, 0] + pred_raw[:, 0]      # lat
        new_row[:, 1] = X_raw[:, 1] + pred_raw[:, 1]      # lon
        new_row[:, 2] = pred_raw[:, 0]                     # d_lat
        new_row[:, 3] = pred_raw[:, 1]                     # d_lon
        new_row[:, 7] = pred_raw[:, 2]                     # wind_speed
        return (new_row - self.X_mean) / self.X_std        # re-normalize [B, 16]


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
    criterion = HaversineLoss(scaler_X, scaler_y, lambda_wind=LAMBDA_WIND).to(device)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # Ramp rollout lambda: 0 for epochs 1–20, linearly up to ROLLOUT_LAMBDA by epoch 40
        rollout_lambda = min(ROLLOUT_LAMBDA, max(0.0, (epoch - 20) / 20.0) * ROLLOUT_LAMBDA)

        model.train()
        train_losses = []
        for X_batch, y_batch, ctx_batch in train_loader:
            X_batch  = X_batch.to(device)
            y_batch  = y_batch.to(device)
            ctx_batch = ctx_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                # Step 1
                pred1 = model(X_batch, ctx_batch)
                loss1 = criterion(pred1, y_batch[:, 0, :], X_batch[:, -1, :])

                # Step 2 (autoregressive rollout — fully differentiable)
                new_row = criterion.build_next_row(X_batch[:, -1, :], pred1)          # [B, 16]
                X_next  = torch.cat([X_batch[:, 1:, :], new_row.unsqueeze(1)], dim=1) # [B, 8, 16]
                pred2 = model(X_next, ctx_batch)
                loss2 = criterion(pred2, y_batch[:, 1, :], new_row)

                loss = loss1 + rollout_lambda * loss2

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_losses.append(loss1.item())  # log step-1 loss for comparability

        train_loss_km = np.mean(train_losses)

        # Validation — step-1 only for metrics
        model.eval()
        val_losses = []
        all_pred_norm = []
        all_X_last = []
        all_y_true_norm = []

        with torch.no_grad():
            for X_batch, y_batch, ctx_batch in val_loader:
                X_batch   = X_batch.to(device)
                y_batch   = y_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    pred = model(X_batch, ctx_batch)
                    loss = criterion(pred, y_batch[:, 0, :], X_batch[:, -1, :])
                val_losses.append(loss.item())
                all_pred_norm.append(pred.cpu().numpy())
                all_X_last.append(X_batch[:, -1, :].cpu().numpy())
                all_y_true_norm.append(y_batch[:, 0, :].cpu().numpy())

        val_loss_km = np.mean(val_losses)

        # Haversine sanity check on val set (numpy, from predict_absolute)
        pred_norm  = np.concatenate(all_pred_norm, axis=0)
        X_last     = np.concatenate(all_X_last, axis=0)
        y_true_norm = np.concatenate(all_y_true_norm, axis=0)
        y_true     = scaler_y.inverse_transform(y_true_norm)
        X_last_raw = scaler_X.inverse_transform(X_last)

        lat_t     = X_last_raw[:, 0]
        lon_t     = X_last_raw[:, 1]
        lat_true  = lat_t + y_true[:, 0]
        lon_true  = lon_t + y_true[:, 1]

        lat_pred, lon_pred, _ = predict_absolute(pred_norm, X_last, scaler_X, scaler_y)
        val_haversine = haversine_km(lat_true, lon_true, lat_pred, lon_pred).mean()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss_km:.2f} km | "
            f"Val loss: {val_loss_km:.2f} km | "
            f"Val Haversine: {val_haversine:.1f} km | "
            f"rollout_λ: {rollout_lambda:.2f} | "
            f"LR: {current_lr:.3e} | "
            f"Time: {elapsed:.1f}s"
        )

        log.append({
            "epoch": epoch,
            "train_loss_km": float(train_loss_km),
            "val_loss_km": float(val_loss_km),
            "val_haversine_km": float(val_haversine),
            "rollout_lambda": float(rollout_lambda),
            "lr": float(current_lr),
        })

        # Checkpoint on val Haversine loss (km units)
        if val_loss_km < best_val_loss:
            best_val_loss = val_loss_km
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "storm_transformer.pt")
            print(f"  --> Saved checkpoint (val loss {best_val_loss:.2f} km)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

        scheduler.step()

    # Save training log
    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.2f} km")
    print(f"Log saved to {MODELS_DIR}/training_log.json")

    return model, log


if __name__ == "__main__":
    train()
