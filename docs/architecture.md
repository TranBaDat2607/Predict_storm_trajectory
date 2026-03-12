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
                       │  Linear(64 → 24)         │
                       │  [batch, 24]             │
                       │        ↓                 │
                       │  .view(-1, 8, 3)         │
                       │  [batch, 8, 3]           │
                       └──────────────┬───────────┘
                                      │
                                      ▼
                       ┌──────────────────────────┐
                       │         OUTPUT           │
                       │     [batch, 8, 3]        │
                       │                          │
                       │  step 0 (+3h):           │
                       │    [0] d_lat  (norm.)    │
                       │    [1] d_lon  (norm.)    │
                       │    [2] wind   (norm.)    │
                       │  step 1 (+6h): ...       │
                       │  ...                     │
                       │  step 7 (+24h): ...      │
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
| After MLP head (flat) | `[batch, 24]` |
| After `.view(-1, 8, 3)` | `[batch, 8, 3]` |

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

### Targets (3 × 8 steps)

One forward pass outputs all 8 future 3h steps simultaneously:

| Axis | Dim | Meaning |
|---|---|---|
| `[:, j, 0]` | step j | d_lat at t+3h×(j+1) (normalized) |
| `[:, j, 1]` | step j | d_lon at t+3h×(j+1) (normalized) |
| `[:, j, 2]` | step j | wind_speed at t+3h×(j+1) (normalized) |

Steps 0–7 correspond to +3h, +6h, …, +24h. To recover absolute position, cumsum the predicted deltas from the last known lat/lon.

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
| Loss function | Multi-step Haversine (km) + 0.1 × wind MAE |
| Output steps | 8 (all predicted in one forward pass) |
| AMP | Enabled for GPU arch < sm_120 |

### Loss Formula

```
loss = mean over j in [0..7] of:
    haversine_km(anchor_j + pred_d_latlon_j, anchor_j + true_d_latlon_j).mean()
    + 0.1 × |pred_wind_j − true_wind_j|.mean()
```

`anchor_j` is the true accumulated position after j steps (teacher forcing on position). No autoregressive rollout — one forward pass, zero error compounding.

### Multi-Horizon Evaluation

| Horizon | Method |
|---|---|
| 3h (step-1) | `pred[:, 0, :]` direct |
| 24h (8-step) | 1 forward pass, cumsum 8 predicted deltas |
| 48h (16-step) | 2 forward passes chained; chunk 2 input = chunk 1 predicted rows |
