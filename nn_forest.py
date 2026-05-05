#!/usr/bin/env python3
"""Lightweight bagging ensemble of MLPClassifiers ("MLP forest").

Defined in a standalone module so the resulting pickle can be loaded by any
script (train_nn_v2.py, all_detector_2.py, etc.) without code duplication.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class MLPForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (96, 48),
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        n_estimators: int = 5,
        max_samples_frac: float = 0.8,
        max_features_frac: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_samples_frac = max_samples_frac
        self.max_features_frac = max_features_frac
        self.random_state = random_state

    def _make_estimator(self, seed: int) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=128,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=False,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.feature_indices_ = []
        n_samples, n_features = X.shape
        sample_size = max(1, int(round(n_samples * self.max_samples_frac)))
        feature_size = max(1, int(round(n_features * self.max_features_frac)))

        for _ in range(int(self.n_estimators)):
            seed = int(rng.randint(0, 2**31 - 1))
            sample_idx = rng.choice(n_samples, size=sample_size, replace=True)
            if feature_size < n_features:
                feature_idx = rng.choice(n_features, size=feature_size, replace=False)
            else:
                feature_idx = np.arange(n_features)
            est = self._make_estimator(seed)
            est.fit(X[np.ix_(sample_idx, feature_idx)], y[sample_idx])
            self.estimators_.append(est)
            self.feature_indices_.append(feature_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        per_est = []
        for est, feat_idx in zip(self.estimators_, self.feature_indices_):
            proba = est.predict_proba(X[:, feat_idx])
            full = np.zeros((proba.shape[0], len(self.classes_)), dtype=float)
            for i, cls in enumerate(est.classes_):
                target_idx = int(np.where(self.classes_ == cls)[0][0])
                full[:, target_idx] = proba[:, i]
            per_est.append(full)
        return np.mean(per_est, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def loss_curves_(self) -> list[list[float]]:
        return [list(getattr(est, "loss_curve_", [])) for est in getattr(self, "estimators_", [])]
