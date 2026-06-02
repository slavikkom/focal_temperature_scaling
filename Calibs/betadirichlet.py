from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "BetaDirichletCalibrator requires SciPy. "
        "Install it with `pip install scipy`."
    ) from e


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute a numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def _nll_from_logits(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    """Mean negative log-likelihood from unnormalized logits."""
    logits = np.asarray(logits, dtype=float) / T
    labels = np.asarray(labels, dtype=int)

    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits array, got shape {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels array, got shape {labels.shape}")
    if len(labels) != len(logits):
        raise ValueError(
            f"Expected {len(logits)} labels, got {len(labels)}")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + shifted[:, 0] * 0
    log_probs = shifted - logsumexp[:, None]
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


@dataclass
class BetaDirichletCalibrator:
    """
    One-vs-rest beta-style multiclass calibrator.

    For each class k:
        score_k = a_k * log p_k + b_k * log(1 - p_k) + c_k

    Calibrated probabilities:
        q = softmax(score)

    This is not full Dirichlet calibration. Full Dirichlet uses a full matrix:
        score = W log(p) + b
    """
    maxiter: int = 100
    clip_eps: float = 1e-12
    lambda_l2: float = 0.0

    a_: Optional[np.ndarray] = None
    b_: Optional[np.ndarray] = None
    c_: Optional[np.ndarray] = None

    @staticmethod
    def _ensure_probs(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        If x looks like probabilities, use as probabilities. Otherwise treat
        x as logits and apply softmax.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array (N, C), got shape {x.shape}")

        if np.all(x >= -1e-8) and np.all(x <= 1.0 + 1e-8):
            row_sums = x.sum(axis=1)
            if np.allclose(row_sums, 1.0, atol=1e-3):
                x = np.clip(x, eps, 1.0)
                x /= x.sum(axis=1, keepdims=True)
                return x

        return _softmax(x, axis=1)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "BetaDirichletCalibrator":
        """Fit by minimizing NLL on validation data."""
        probs = self._ensure_probs(logits, eps=self.clip_eps)
        labels = np.asarray(labels, dtype=int)

        _, c = probs.shape

        p_clipped = np.clip(probs, self.clip_eps, 1.0 - self.clip_eps)
        log_p = np.log(p_clipped)
        log_1_minus_p = np.log(1.0 - p_clipped)

        a0 = np.ones(c, dtype=float)
        b0 = np.zeros(c, dtype=float)
        c0 = np.zeros(c, dtype=float)
        theta0 = np.concatenate([a0, b0, c0])

        def unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            a = theta[:c]
            b = theta[c:2 * c]
            cc = theta[2 * c:3 * c]
            return a, b, cc

        def objective(theta: np.ndarray) -> float:
            a, b, cc = unpack(theta)
            scores = (
                a[None, :] * log_p
                + b[None, :] * log_1_minus_p
                + cc[None, :]
            )
            nll = _nll_from_logits(scores, labels, T=1.0)
            if self.lambda_l2 > 0.0:
                nll += 0.5 * self.lambda_l2 * (
                    np.sum(a**2) + np.sum(b**2) + np.sum(cc**2)
                )
            return float(nll)

        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )

        self.a_, self.b_, self.c_ = unpack(res.x)
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated scores before the final softmax."""
        if self.a_ is None or self.b_ is None or self.c_ is None:
            raise RuntimeError("BetaDirichletCalibrator.fit() must be called first.")

        probs = self._ensure_probs(logits, eps=self.clip_eps)
        p_clipped = np.clip(probs, self.clip_eps, 1.0 - self.clip_eps)
        log_p = np.log(p_clipped)
        log_1_minus_p = np.log(1.0 - p_clipped)

        return (
            self.a_[None, :] * log_p
            + self.b_[None, :] * log_1_minus_p
            + self.c_[None, :]
        )

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return _softmax(self.transform_logits(logits), axis=1)

