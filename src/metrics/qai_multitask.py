from __future__ import annotations

import torch


class QaiMultiTaskMetrics:
    def __call__(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        price_preds = outputs["price_preds"]
        class_logits = outputs["class_logits"]
        price_targets = batch["price_targets"]
        class_targets = batch["class_targets"]

        mae = torch.mean(torch.abs(price_preds - price_targets))
        pred_classes = class_logits.argmax(dim=-1)
        accuracy = (pred_classes == class_targets).float().mean()

        metrics: dict[str, torch.Tensor] = {
            "price_mae": mae.detach(),
            "class_accuracy": accuracy.detach(),
        }
        for horizon_idx in range(class_targets.size(1)):
            metrics[f"class_accuracy_h{horizon_idx + 1}"] = (
                (pred_classes[:, horizon_idx] == class_targets[:, horizon_idx])
                .float()
                .mean()
                .detach()
            )
        return metrics
