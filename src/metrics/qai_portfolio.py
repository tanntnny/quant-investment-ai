from __future__ import annotations

import torch


class QaiPortfolioMetrics:
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
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            valid_counts = mask.sum(dim=-1).clamp_min(1.0)
            equal_weights = mask / valid_counts.unsqueeze(-1)
        else:
            valid_counts = torch.full(
                (weights.size(0),),
                float(weights.size(-1)),
                dtype=weights.dtype,
                device=weights.device,
            )
            equal_weights = torch.full_like(weights, 1.0 / weights.size(-1))

        current_value = torch.sum(weights * current_prices, dim=-1).clamp_min(1e-8)
        future_value = torch.sum(weights * future_prices, dim=-1).clamp_min(1e-8)
        portfolio_growth = future_value / current_value
        portfolio_return = portfolio_growth - 1.0

        benchmark_current = torch.sum(equal_weights * current_prices, dim=-1).clamp_min(1e-8)
        benchmark_future = torch.sum(equal_weights * future_prices, dim=-1).clamp_min(1e-8)
        benchmark_growth = benchmark_future / benchmark_current
        benchmark_return = benchmark_growth - 1.0

        max_weight = weights.max(dim=-1).values
        effective_holdings = 1.0 / torch.sum(weights.square(), dim=-1).clamp_min(1e-8)

        return {
            "portfolio_growth": portfolio_growth.mean().detach(),
            "portfolio_return": portfolio_return.mean().detach(),
            "equal_weight_growth": benchmark_growth.mean().detach(),
            "equal_weight_return": benchmark_return.mean().detach(),
            "growth_alpha": (portfolio_growth - benchmark_growth).mean().detach(),
            "return_alpha": (portfolio_return - benchmark_return).mean().detach(),
            "max_weight": max_weight.mean().detach(),
            "effective_holdings": effective_holdings.mean().detach(),
        }
