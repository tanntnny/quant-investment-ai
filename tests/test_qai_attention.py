from __future__ import annotations

import torch

from src.models.qai_attention import QaiAttentionModel


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
