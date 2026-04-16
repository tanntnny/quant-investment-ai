from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(eq=False)
class QaiPortfolioBiLSTMModel(nn.Module):
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    sequence_length: int = 4

    def __post_init__(self) -> None:
        super().__init__()
        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        embedding_dim = self.hidden_dim * 2
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.score_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        history_attention_mask: torch.Tensor | None = None,
        ticker_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, ticker_count, seq_len, feature_dim = features.shape
        flat_features = features.reshape(batch_size * ticker_count, seq_len, feature_dim)
        encoded, _ = self.encoder(flat_features)

        if history_attention_mask is not None:
            flat_mask = history_attention_mask.reshape(batch_size * ticker_count, seq_len).bool()
            valid_rows = flat_mask.any(dim=-1)
            lengths = flat_mask.sum(dim=-1).clamp_min(1)
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
            ticker_embeddings = encoded.gather(dim=1, index=gather_index).squeeze(1)
            ticker_embeddings = ticker_embeddings.masked_fill(~valid_rows.unsqueeze(-1), 0.0)
        else:
            ticker_embeddings = encoded[:, -1, :]

        ticker_embeddings = ticker_embeddings.view(batch_size, ticker_count, -1)
        ticker_embeddings = self.dropout_layer(self.norm(ticker_embeddings))
        scores = self.score_head(ticker_embeddings).squeeze(-1)

        if ticker_attention_mask is not None:
            scores = scores.masked_fill(~ticker_attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if ticker_attention_mask is not None:
            weights = weights * ticker_attention_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {
            "weights": weights,
            "scores": scores,
            "ticker_embeddings": ticker_embeddings,
        }
