from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from src.data.providers.base import FinancialDataProvider
from src.data.providers.simfin_provider import SimFinProvider


def create_provider(provider_cfg: DictConfig, raw_data_dir: str) -> FinancialDataProvider:
    provider_type = provider_cfg.type
    if provider_type == "simfin":
        return SimFinProvider(
            api_key=provider_cfg.api_key,
            env_file=provider_cfg.get("env_file", ".env"),
            api_key_env=provider_cfg.get("api_key_env", "SIMFIN_API_KEY"),
            raw_data_dir=Path(raw_data_dir),
            cache_template=provider_cfg.cache_template,
            historical_price_years=provider_cfg.historical_price_years,
            market=provider_cfg.market,
            fundamentals_variant=provider_cfg.fundamentals_variant,
            prices_variant=provider_cfg.prices_variant,
        )
    raise ValueError(f"Unsupported provider type: {provider_type}")
