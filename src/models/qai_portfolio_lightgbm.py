from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QaiPortfolioLightGBMModel:
    input_dim: int | None = None
    objective: str = "regression"
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 123
    temperature: float = 1.0
    _model: Any = field(default=None, init=False, repr=False)

    def fit(self, samples: list[dict[str, Any]]) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is required for QaiPortfolioLightGBMModel. "
                "Install the project's lightgbm extra before running this model."
            ) from exc

        x_train, y_train = self._samples_to_xy(samples)
        if x_train.size == 0:
            raise RuntimeError("No LightGBM training rows were generated from portfolio samples.")
        self._model = lgb.LGBMRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
        )
        self._model.fit(x_train, y_train)

    def predict_sample_weights(self, samples: list[dict[str, Any]]) -> list[np.ndarray]:
        if self._model is None:
            raise RuntimeError("QaiPortfolioLightGBMModel must be fit before prediction.")

        weights_by_sample: list[np.ndarray] = []
        for sample in samples:
            features = sample["features"].detach().cpu().numpy()
            ticker_count = int(sample["ticker_count"])
            flat_features = features[:ticker_count].reshape(ticker_count, -1)
            scores = np.asarray(self._model.predict(flat_features), dtype=np.float64)
            weights_by_sample.append(_softmax(scores / max(self.temperature, 1e-8)))
        return weights_by_sample

    def parameters(self) -> list[Any]:
        return []

    def named_children(self) -> list[tuple[str, Any]]:
        return []

    def to(self, *_args, **_kwargs) -> "QaiPortfolioLightGBMModel":
        return self

    @staticmethod
    def _samples_to_xy(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for sample in samples:
            features = sample["features"].detach().cpu().numpy()
            current_prices = sample["current_prices"].detach().cpu().numpy()
            future_prices = sample["future_prices"].detach().cpu().numpy()
            ticker_count = int(sample["ticker_count"])
            for ticker_idx in range(ticker_count):
                current_price = float(current_prices[ticker_idx])
                future_price = float(future_prices[ticker_idx])
                if current_price <= 0 or not np.isfinite(current_price + future_price):
                    continue
                rows.append(features[ticker_idx].reshape(-1))
                targets.append((future_price / current_price) - 1.0)
        if not rows:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.vstack(rows).astype(np.float32), np.asarray(targets, dtype=np.float32)


def _softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    shifted = scores - np.nanmax(scores)
    exp_scores = np.exp(shifted)
    denominator = exp_scores.sum()
    if not np.isfinite(denominator) or denominator <= 0:
        return np.full_like(scores, 1.0 / len(scores), dtype=np.float64)
    return exp_scores / denominator
