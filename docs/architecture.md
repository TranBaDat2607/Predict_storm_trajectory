# StormTransformer — Model Architecture

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT TENSORS                                │
│                                                                         │
│   x: [batch, 8, 16]              ctx: [batch, 2]                        │
│   (sequence features)            (basin_id, season_norm)                │
└───────────────┬──────────────────────────┬──────────────────────────────┘
                │                          │
                ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────────────────────────┐
│      input_proj          │   │           CONTEXT EMBEDDING              │
│  Linear(16 → 64)         │   │                                          │
│                          │   │  ctx[:,0] ──► Embedding(7, 64)           │
│  [batch, 8, 16]          │   │               [batch, 64]    ┐           │
│       ↓                  │   │                              ├─► (+) ──► │
│  [batch, 8, 64]          │   │  ctx[:,1:] ──► Linear(1→64)  ┘           │
└──────────────┬───────────┘   │               [batch, 64]                │
               │               └───────────────────┬──────────────────────┘
               │                                   │ unsqueeze(1)
               │                                   │ [batch, 1, 64]
               ▼                                   │ broadcast →
┌──────────────────────────┐                       │ [batch, 8, 64]
│      pos_emb             │                       │
│  Embedding(8, 64)        │                       │
│  positions: [0,1,...,7]  │                       │
│  → [8, 64]               │                       │
│  broadcast → [batch,8,64]│                       │
└──────────────┬───────────┘                       │
               │                                   │
               └──────────────┬────────────────────┘
                              │  element-wise (+)
                              ▼
               ┌──────────────────────────┐
               │    EMBEDDED SEQUENCE     │
               │      [batch, 8, 64]      │
               └──────────────┬───────────┘
                              │
                ┌─────────────▼───────────────────────────────────────┐
                │           TRANSFORMER ENCODER LAYER × 3             │
                │                                                     │
                │  ┌───────────────────────────────────────────────┐  │
                │  │         Layer 1  [batch, 8, 64]               │  │
                │  │                                               │  │
                │  │  ┌─────────────────────────────────────────┐  │  │
                │  │  │  MultiHeadAttention                     │  │  │
                │  │  │  4 heads, d_head=16, d_model=64         │  │  │
                │  │  │  Q/K/V: [batch, 8, 64]                  │  │  │
                │  │  │  Attn:  [batch, 4, 8, 8]                │  │  │
                │  │  │  Out:   [batch, 8, 64]                  │  │  │
                │  │  └───────────────┬─────────────────────────┘  │  │
                │  │                  │ + residual + LayerNorm     │  │
                │  │  ┌───────────────▼────────────────────────┐   │  │
                │  │  │  FFN                                   │   │  │
                │  │  │  Linear(64 → 256) → GELU               │   │  │
                │  │  │  Linear(256 → 64) → Dropout(0.1)       │   │  │
                │  │  │  [batch, 8, 64]                        │   │  │
                │  │  └────────────────────────────────────────┘   │  │
                │  │                  + residual + LayerNorm       │  │
                │  └───────────────────────────────────────────────┘  │
                │                      ↓  (repeat ×2 more)            │
                │  ┌───────────────────────────────────────────────┐  │
                │  │              Layer 2  [batch, 8, 64]          │  │
                │  └───────────────────────────────────────────────┘  │
                │  ┌───────────────────────────────────────────────┐  │
                │  │              Layer 3  [batch, 8, 64]          │  │
                │  └───────────────────────────────────────────────┘  │
                └─────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │    LAST-TOKEN POOL       │
                       │    x[:, -1, :]           │
                       │    [batch, 8, 64]        │
                       │         ↓                │
                       │    [batch, 64]           │
                       └──────────────┬───────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │       MLP HEAD           │
                       │                          │
                       │  Linear(64 → 64)         │
                       │  [batch, 64]             │
                       │        ↓                 │
                       │  GELU                    │
                       │        ↓                 │
                       │  Dropout(0.1)            │
                       │        ↓                 │
                       │  Linear(64 → 3)          │
                       │  [batch, 3]              │
                       └──────────────┬───────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │         OUTPUT           │
                       │       [batch, 3]         │
                       │                          │
                       │  [0] d_lat   (normalized)│
                       │  [1] d_lon   (normalized)│
                       │  [2] wind_speed (norm.)  │
                       └──────────────────────────┘
```

---

## Tensor Shape Summary

| Stage | Shape |
|---|---|
| Raw input sequence | `[batch, 8, 16]` |
| After `input_proj` | `[batch, 8, 64]` |
| After `+pos_emb +ctx_emb` | `[batch, 8, 64]` |
| After each Transformer layer | `[batch, 8, 64]` |
| After last-token pool | `[batch, 64]` |
| After MLP head | `[batch, 3]` |

---

## Feature & Target Reference

### Input Features (16)

| # | Feature | Description |
|---|---|---|
| 0 | `lat` | Absolute latitude |
| 1 | `lon` | Absolute longitude |
| 2 | `d_lat` | Δlat from previous timestep |
| 3 | `d_lon` | Δlon from previous timestep |
| 4 | `nature` | Storm type, label-encoded 0–5 |
| 5 | `dist2land` | Distance to nearest land (km) |
| 6 | `landfall` | Landfall flag |
| 7 | `wind_speed` | Max sustained wind (knots) |
| 8 | `storm_pres` | Minimum central pressure (hPa) |
| 9 | `usa_sshs` | Saffir-Simpson category |
| 10 | `usa_poci` | Pressure of outermost closed isobar |
| 11 | `usa_roci` | Radius of outermost closed isobar |
| 12 | `usa_rmw` | Radius of maximum winds |
| 13 | `storm_speed` | Translation speed (kt) |
| 14 | `storm_dir_sin` | Heading — sine component |
| 15 | `storm_dir_cos` | Heading — cosine component |

### Context Vector (2)

| # | Feature | Description |
|---|---|---|
| 0 | `basin_id` | Basin, label-encoded 0–6 (EP/NA/NI/SA/SI/SP/WP) |
| 1 | `season_norm` | `(season − 1980) / 40.0` |

### Targets (3)

| # | Target | Description |
|---|---|---|
| 0 | `d_lat` | Latitude delta at next timestep |
| 1 | `d_lon` | Longitude delta at next timestep |
| 2 | `wind_speed` | Wind speed at next timestep |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=100, eta_min=1e-5) |
| Batch size | 512 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Gradient clip norm | 1.0 |
| Loss function | Haversine (km) + 0.1 × wind MAE |
| Rollout steps | 2 (step-2 loss ramped in epochs 20→40) |
| AMP | Enabled for GPU arch < sm_120 |

### Loss Formula

```
loss = haversine_km(pred_pos, true_pos).mean()
     + 0.1 × |pred_wind − true_wind|.mean()
     + rollout_λ × loss_step2
```

`rollout_λ` ramps from 0 (epoch 1–20) to 0.5 (epoch 40+), training the model to minimise compounding error over 2 autoregressive steps.
