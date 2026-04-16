from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class _AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


@dataclass(eq=False)
class QaiAttentionModel(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.1
    forecast_horizons: int = 4
    num_classes: int = 3
    sequence_length: int = 4

    def __post_init__(self) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, self.hidden_dim)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.pool = _AttentionPooling(self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.price_head = nn.Linear(self.hidden_dim, self.forecast_horizons)
        self.class_head = nn.Linear(
            self.hidden_dim, self.forecast_horizons * self.num_classes
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.input_projection(features)
        hidden_states = hidden_states + self.position_embedding[:, : features.size(1), :]
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=padding_mask)
        pooled = self.pool(hidden_states, attention_mask=attention_mask)
        pooled = self.dropout_layer(self.norm(pooled))
        price_preds = self.price_head(pooled)
        class_logits = self.class_head(pooled).view(
            features.size(0), self.forecast_horizons, self.num_classes
        )
        return {
            "price_preds": price_preds,
            "class_logits": class_logits,
        }
