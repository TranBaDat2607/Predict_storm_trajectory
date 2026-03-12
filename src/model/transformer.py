"""
StormTransformer - Transformer encoder for storm trajectory + intensity prediction.

Architecture:
  Input [batch, SEQ_LEN, n_features]
  -> Linear(n_features, d_model)
  + learned positional embedding [SEQ_LEN, d_model]
  + basin/season context embedding (broadcast over sequence)
  -> TransformerEncoder (3 layers, 4 heads, d_ff=256, dropout=0.1)
  -> last-token pool -> Linear(d_model, d_model) -> GELU -> Dropout
  -> Linear(d_model, N_ROLLOUT_STEPS * 3)
  Output [batch, N_ROLLOUT_STEPS, 3]  - normalised (d_lat, d_lon, wind_speed) for each step
"""

import torch
import torch.nn as nn

from .dataset import N_FEATURES, N_TARGETS, SEQ_LEN, N_BASINS, N_ROLLOUT_STEPS


class StormTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        seq_len: int = SEQ_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_targets: int = N_TARGETS,
        n_basins: int = N_BASINS,
        use_season: bool = True,
        n_rollout_steps: int = N_ROLLOUT_STEPS,
    ):
        super().__init__()

        self.n_rollout_steps = n_rollout_steps
        self.n_targets = n_targets

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Storm-level context: basin (categorical) + season (continuous)
        self.basin_emb = nn.Embedding(n_basins, d_model)
        self.use_season = use_season
        if use_season:
            self.season_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_rollout_steps * n_targets),
        )

        self._register_positions(seq_len)

    def _register_positions(self, seq_len: int):
        pos = torch.arange(seq_len)
        self.register_buffer("positions", pos)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, mask=None) -> torch.Tensor:
        # x:    [batch, seq_len, n_features]
        # ctx:  [batch, 2]  - (basin_id as float, season_norm)
        # mask: [batch, seq_len] bool, True=padded position (ignored by attention)
        x = self.input_proj(x)                         # [batch, seq_len, d_model]
        x = x + self.pos_emb(self.positions)           # broadcast positional emb

        # Context embedding: basin (int lookup) + season (linear projection)
        ctx_emb = self.basin_emb(ctx[:, 0].long())     # [batch, d_model]
        if self.use_season:
            ctx_emb = ctx_emb + self.season_proj(ctx[:, 1:])  # [batch, d_model]
        x = x + ctx_emb.unsqueeze(1)                   # broadcast to [batch, seq_len, d_model]

        x = self.encoder(x, src_key_padding_mask=mask) # [batch, seq_len, d_model]
        x = x[:, -1, :]                                # last-token pool [batch, d_model]
        out = self.head(x)                             # [batch, n_rollout_steps * n_targets]
        return out.view(out.shape[0], self.n_rollout_steps, self.n_targets)  # [batch, N_ROLLOUT_STEPS, 3]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
