from __future__ import annotations

from pathlib import Path


def get_hydra_output_dir() -> Path | None:
    try:
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return None

    if not HydraConfig.initialized():
        return None

    output_dir = HydraConfig.get().runtime.output_dir
    if not output_dir:
        return None
    return Path(output_dir).resolve()
