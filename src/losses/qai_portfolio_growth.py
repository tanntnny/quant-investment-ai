from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QaiPortfolioGrowthLoss:
    use_log_growth: bool = True
    risk_penalty: float = 0.0
    concentration_penalty: float = 0.0
    eps: float = 1e-8

    def __call__(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        weights = outputs["weights"]
        current_prices = batch["current_prices"]
        future_prices = batch["future_prices"]
        ticker_attention_mask = batch.get("ticker_attention_mask")

        if ticker_attention_mask is not None:
            mask = ticker_attention_mask.float()
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # current_value = torch.sum(weights * current_prices, dim=-1).clamp_min(self.eps)
        # future_value = torch.sum(weights * future_prices, dim=-1).clamp_min(self.eps)
        # portfolio_growth = future_value / current_value
        # portfolio_return = portfolio_growth - 1.0
        asset_growth = future_prices / current_prices.clamp_min(self.eps)
        portfolio_growth = torch.sum(weights * asset_growth, dim=-1).clamp_min(self.eps)
        portfolio_return = portfolio_growth - 1.0

        if self.use_log_growth:
            base_loss = -torch.mean(torch.log(portfolio_growth.clamp_min(self.eps)))
        else:
            base_loss = -torch.mean(portfolio_return)

        drawdown = torch.clamp(1.0 - portfolio_growth, min=0.0)
        risk_penalty_loss = self.risk_penalty * torch.mean(drawdown)
        concentration = torch.sum(weights.square(), dim=-1)
        concentration_penalty_loss = self.concentration_penalty * torch.mean(concentration)
        loss = base_loss + risk_penalty_loss + concentration_penalty_loss

        return {
            "loss": loss,
            "regression_loss": loss.detach(),
            "base_loss": base_loss.detach(),
            "risk_penalty_loss": risk_penalty_loss.detach(),
            "concentration_penalty_loss": concentration_penalty_loss.detach(),
            "concentration": concentration.mean().detach(),
            "portfolio_growth": portfolio_growth.mean().detach(),
            "portfolio_return": portfolio_return.mean().detach(),
        }
