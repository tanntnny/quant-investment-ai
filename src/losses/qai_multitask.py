from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class QaiMultiTaskLoss:
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    regression_loss: str = "mse"
    horizon_weights: Sequence[float] | None = None

    def __call__(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        price_preds = outputs["price_preds"]
        class_logits = outputs["class_logits"]
        price_targets = batch["price_targets"]
        class_targets = batch["class_targets"]

        horizon_weights = self._resolve_horizon_weights(price_targets)

        if self.regression_loss == "smooth_l1":
            regression_by_element = F.smooth_l1_loss(
                price_preds,
                price_targets,
                reduction="none",
            )
        else:
            regression_by_element = F.mse_loss(
                price_preds,
                price_targets,
                reduction="none",
            )
        regression = (regression_by_element * horizon_weights).sum(dim=1).mean()

        classification_by_element = F.cross_entropy(
            class_logits.view(-1, class_logits.size(-1)),
            class_targets.view(-1),
            reduction="none",
        ).view_as(class_targets)
        classification = (classification_by_element * horizon_weights).sum(dim=1).mean()
        total = (
            self.regression_weight * regression
            + self.classification_weight * classification
        )
        return {
            "loss": total,
            "regression_loss": regression.detach(),
            "classification_loss": classification.detach(),
        }

    def _resolve_horizon_weights(self, targets: torch.Tensor) -> torch.Tensor:
        if self.horizon_weights is None:
            return torch.full(
                (targets.size(1),),
                1.0 / targets.size(1),
                dtype=targets.dtype,
                device=targets.device,
            )

        weights = torch.tensor(
            list(self.horizon_weights),
            dtype=targets.dtype,
            device=targets.device,
        )
        if weights.numel() != targets.size(1):
            raise ValueError(
                "horizon_weights length must match the number of predicted horizons"
            )
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("horizon_weights must sum to a positive value")
        return weights / weight_sum
