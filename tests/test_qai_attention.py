from __future__ import annotations

import torch

from src.models.qai_attention import QaiAttentionModel
from src.models.qai_portfolio_bilstm import QaiPortfolioBiLSTMModel


def test_qai_attention_forward_shapes() -> None:
    model = QaiAttentionModel(
        input_dim=6,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
        forecast_horizons=4,
        num_classes=3,
    )
    features = torch.randn(3, 4, 6)

    outputs = model(features)

    assert outputs["price_preds"].shape == (3, 4)
    assert outputs["class_logits"].shape == (3, 4, 3)


def test_qai_portfolio_bilstm_forward_shapes() -> None:
    model = QaiPortfolioBiLSTMModel(
        input_dim=6,
        hidden_dim=8,
        num_layers=1,
        sequence_length=4,
    )
    features = torch.randn(2, 5, 4, 6)
    history_attention_mask = torch.ones(2, 5, 4, dtype=torch.bool)
    ticker_attention_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    outputs = model(
        features,
        history_attention_mask=history_attention_mask,
        ticker_attention_mask=ticker_attention_mask,
    )

    assert outputs["weights"].shape == (2, 5)
    assert outputs["scores"].shape == (2, 5)
    assert outputs["ticker_embeddings"].shape == (2, 5, 16)
    assert outputs["weights"][0, 3:].sum().item() == 0.0
    assert torch.allclose(outputs["weights"].sum(dim=-1), torch.ones(2))
