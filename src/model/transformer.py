"""
StormTransformer — Transformer encoder for storm trajectory + intensity prediction.

Architecture:
  Input [batch, SEQ_LEN, n_features]
  → Linear(n_features, d_model)
  + learned positional embedding [SEQ_LEN, d_model]
  → TransformerEncoder (3 layers, 4 heads, d_ff=256, dropout=0.1)
  → mean pool → Linear(d_model, d_model) → GELU → Dropout
  → Linear(d_model, 3)
  Output [batch, 3]  — normalised (d_lat, d_lon, wind_speed)
"""

import torch
import torch.nn as nn

from .dataset import N_FEATURES, N_TARGETS, SEQ_LEN


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
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

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
            nn.Linear(d_model, n_targets),
        )

        self._register_positions(seq_len)

    def _register_positions(self, seq_len: int):
        pos = torch.arange(seq_len)
        self.register_buffer("positions", pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features]
        x = self.input_proj(x)                         # [batch, seq_len, d_model]
        x = x + self.pos_emb(self.positions)           # broadcast positional emb
        x = self.encoder(x)                            # [batch, seq_len, d_model]
        x = x.mean(dim=1)                              # [batch, d_model]
        return self.head(x)                            # [batch, n_targets]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
