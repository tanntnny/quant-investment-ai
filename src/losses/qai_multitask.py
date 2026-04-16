from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class QaiMultiTaskLoss:
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    regression_loss: str = "mse"

    def __call__(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        price_preds = outputs["price_preds"]
        class_logits = outputs["class_logits"]
        price_targets = batch["price_targets"]
        class_targets = batch["class_targets"]

        if self.regression_loss == "smooth_l1":
            regression = F.smooth_l1_loss(price_preds, price_targets)
        else:
            regression = F.mse_loss(price_preds, price_targets)

        classification = F.cross_entropy(
            class_logits.view(-1, class_logits.size(-1)),
            class_targets.view(-1),
        )
        total = (
            self.regression_weight * regression
            + self.classification_weight * classification
        )
        return {
            "loss": total,
            "regression_loss": regression.detach(),
            "classification_loss": classification.detach(),
        }
