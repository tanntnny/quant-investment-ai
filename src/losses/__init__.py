"""Loss implementations."""

from src.losses.qai_portfolio_growth import QaiPortfolioGrowthLoss
from src.losses.qai_multitask import QaiMultiTaskLoss

__all__ = ["QaiMultiTaskLoss", "QaiPortfolioGrowthLoss"]
