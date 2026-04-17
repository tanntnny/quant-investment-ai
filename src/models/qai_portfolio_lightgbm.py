from __future__ import annotations

import importlib.util
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QaiPortfolioLightGBMModel:
    input_dim: int | None = None
    sequence_length: int | None = None
    training_backend: str = "lightgbm"
    requires_optimizer: bool = False
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
    _backend: str = field(default="", init=False, repr=False)

    def fit(self, samples: list[dict[str, Any]]) -> None:
        x_train, y_train = self._samples_to_xy(samples)
        if x_train.size == 0:
            raise RuntimeError("No LightGBM training rows were generated from portfolio samples.")
        lightgbm_module = self._load_lightgbm_module()
        if lightgbm_module is not None:
            self._backend = "lightgbm"
            self._model = lightgbm_module.LGBMRegressor(
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
        else:
            self._backend = "sklearn"
            warnings.warn(
                "lightgbm is not installed; falling back to sklearn.HistGradientBoostingRegressor "
                "for QaiPortfolioLightGBMModel.",
                RuntimeWarning,
                stacklevel=2,
            )
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.learning_rate,
                max_iter=self.n_estimators,
                max_depth=None if self.max_depth < 0 else self.max_depth,
                min_samples_leaf=self.min_child_samples,
                random_state=self.random_state,
                early_stopping=False,
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
            if not np.isfinite(flat_features).all():
                raise RuntimeError("LightGBM prediction features contain NaN or infinite values.")
            scores = np.asarray(self._model.predict(flat_features), dtype=np.float64)
            if not np.isfinite(scores).all():
                raise RuntimeError("LightGBM prediction scores contain NaN or infinite values.")
            weights_by_sample.append(_softmax(scores / max(self.temperature, 1e-8)))
        return weights_by_sample

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted QaiPortfolioLightGBMModel.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "QaiPortfolioLightGBMModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, QaiPortfolioLightGBMModel):
            raise RuntimeError(f"Unexpected LightGBM model artifact type: {type(model)!r}")
        if model._model is None:
            raise RuntimeError("Loaded LightGBM model artifact is not fitted.")
        return model

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
                flat_features = features[ticker_idx].reshape(-1)
                if current_price <= 0:
                    raise RuntimeError("LightGBM training row has a non-positive current price.")
                if future_price <= 0:
                    raise RuntimeError("LightGBM training row has a non-positive future price.")
                if not np.isfinite(current_price + future_price):
                    raise RuntimeError("LightGBM training row has a non-finite price target.")
                if not np.isfinite(flat_features).all():
                    raise RuntimeError("LightGBM training features contain NaN or infinite values.")
                target = (future_price / current_price) - 1.0
                if not np.isfinite(target):
                    raise RuntimeError("LightGBM training target contains NaN or infinite values.")
                rows.append(flat_features)
                targets.append(target)
        if not rows:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.vstack(rows).astype(np.float32), np.asarray(targets, dtype=np.float32)

    @staticmethod
    def _load_lightgbm_module() -> Any:
        if importlib.util.find_spec("lightgbm") is None:
            return None
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        return lgb


def _softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    shifted = scores - np.nanmax(scores)
    exp_scores = np.exp(shifted)
    denominator = exp_scores.sum()
    if not np.isfinite(denominator) or denominator <= 0:
        return np.full_like(scores, 1.0 / len(scores), dtype=np.float64)
    return exp_scores / denominator
