from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class _MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=-2)


@dataclass(eq=False)
class QaiPortfolioAttentionModel(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    sequence_length: int = 4

    def __post_init__(self) -> None:
        super().__init__()
        history_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        ticker_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, self.hidden_dim))
        self.history_encoder = nn.TransformerEncoder(
            history_encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.history_pool = _MaskedAttentionPooling(self.hidden_dim)
        self.ticker_encoder = nn.TransformerEncoder(
            ticker_encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.score_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        history_attention_mask: torch.Tensor | None = None,
        ticker_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, ticker_count, seq_len, _ = features.shape
        flat_features = features.view(batch_size * ticker_count, seq_len, -1)
        hidden_states = self.input_projection(flat_features)
        hidden_states = hidden_states + self.position_embedding[:, :seq_len, :]

        history_padding_mask = None
        if history_attention_mask is not None:
            flat_history_mask = history_attention_mask.view(batch_size * ticker_count, seq_len).bool()
            valid_history_rows = flat_history_mask.any(dim=-1)
            safe_history_mask = flat_history_mask.clone()
            safe_history_mask[~valid_history_rows, 0] = True
            history_padding_mask = ~safe_history_mask
        else:
            flat_history_mask = None
            safe_history_mask = None
            valid_history_rows = None

        hidden_states = self.history_encoder(hidden_states, src_key_padding_mask=history_padding_mask)
        ticker_embeddings = self.history_pool(hidden_states, attention_mask=safe_history_mask)
        if valid_history_rows is not None:
            ticker_embeddings = ticker_embeddings.masked_fill(~valid_history_rows.unsqueeze(-1), 0.0)
        ticker_embeddings = ticker_embeddings.view(batch_size, ticker_count, self.hidden_dim)

        ticker_padding_mask = None
        if ticker_attention_mask is not None:
            ticker_padding_mask = ~ticker_attention_mask.bool()

        cross_ticker = self.ticker_encoder(ticker_embeddings, src_key_padding_mask=ticker_padding_mask)
        cross_ticker = self.dropout_layer(self.norm(cross_ticker))
        scores = self.score_head(cross_ticker).squeeze(-1)
        if ticker_attention_mask is not None:
            scores = scores.masked_fill(~ticker_attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if ticker_attention_mask is not None:
            weights = weights * ticker_attention_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {
            "weights": weights,
            "scores": scores,
            "ticker_embeddings": cross_ticker,
        }
