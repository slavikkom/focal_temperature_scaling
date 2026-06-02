"""
Sanity checks for Dirichlet calibrators. These tests are not meant to be exhaustive, 
but rather to catch obvious bugs and ensure that the basic functionality is working as expected.
"""

import numpy as np
import torch
from sklearn.metrics import log_loss

from fulldirichlet import FullDirichletCalibrator
from diagdirichlet import DiagonalDirichletCalibrator
from betadirichlet import BetaDirichletCalibrator
from multinomial import MultinomialRegression
from utils import log_clipped


device = "cuda" if torch.cuda.is_available() else "cpu"


def make_probs(n=100, k=4, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(k), size=n)
    y = rng.integers(0, k, size=n)
    return probs, y


def calibrator_cases(reg_value=1e-3, max_iter=20, device=device):
    """Return all calibrator configurations used by these sanity checks."""
    return [
        (
            "full",
            FullDirichletCalibrator(
                reg_lambda=reg_value, reg_mu=None, max_iter=max_iter,
                device=device),
            False,
        ),
        (
            "odir",
            FullDirichletCalibrator(
                reg_lambda=reg_value, reg_mu=reg_value, max_iter=max_iter,
                device=device),
            False,
        ),
        (
            "diagonal",
            DiagonalDirichletCalibrator(
                reg_lambda=reg_value, reg_mu=None, max_iter=max_iter,
                device=device),
            False,
        ),
        (
            "beta",
            BetaDirichletCalibrator(lambda_l2=reg_value, maxiter=max_iter),
            False,
        ),
        (
            "fixdiag",
            MultinomialRegression(
                method="FixDiag", reg_lambda=reg_value, max_iter=max_iter,
                device=device),
            True,
        ),
    ]


def fit_case(case, probs, y):
    _, cal, needs_log_probs = case
    x = log_clipped(probs) if needs_log_probs else probs
    cal.fit(x, y)
    return cal


def predict_case(case, probs):
    _, cal, needs_log_probs = case
    x = log_clipped(probs) if needs_log_probs else probs
    return cal.predict_proba(x)


def assert_valid_probabilities(out, expected_shape):
    assert out.shape == expected_shape, "Output shape should match input shape"
    assert np.all(np.isfinite(out)), "Output should contain only finite values"
    assert np.all(out >= 0.0), "Output should contain only non-negative values"
    assert np.all(out <= 1.0), "Output should contain only values between 0 and 1"
    assert np.allclose(out.sum(axis=1), 1.0), "Output rows should sum to 1"


def test_calibrators_predict_valid_probabilities():
    """Test that all calibrators predict valid probabilities."""
    print("Testing that all calibrators predict valid probabilities...")
    probs, y = make_probs()

    for case in calibrator_cases(reg_value=1e-3, max_iter=20):
        name, _, _ = case
        fit_case(case, probs, y)
        out = predict_case(case, probs)
        assert_valid_probabilities(out, probs.shape)


def test_calibrators_do_not_worsen_training_nll_without_regularization():
    """Test that unregularized calibrators do not worsen the training NLL."""
    print("Testing that unregularized calibrators do not worsen training NLL...")
    probs, y = make_probs(n=200, k=5)

    base_loss = log_loss(y, probs)

    for case in calibrator_cases(reg_value=0.0, max_iter=50):
        name, _, _ = case
        fit_case(case, probs, y)
        cal_loss = log_loss(y, predict_case(case, probs))

        assert cal_loss <= base_loss + 1e-6, (
            "{} fitted NLL should not be worse than base NLL without "
            "regularization".format(name))


def test_torch_tensor_input_matches_numpy_input():
    """Test that fitting with torch tensors gives the same result as fitting with numpy arrays."""
    print("Testing torch tensor input matches numpy input...")
    probs, y = make_probs(n=80, k=3)

    probs_torch = torch.tensor(probs)
    y_torch = torch.tensor(y)

    for case_np, case_torch in zip(
            calibrator_cases(reg_value=1e-3, max_iter=20, device="cpu"),
            calibrator_cases(reg_value=1e-3, max_iter=20, device="cpu")):
        name, _, _ = case_np
        fit_case(case_np, probs, y)
        fit_case(case_torch, probs_torch, y_torch)

        out_np = predict_case(case_np, probs)
        out_torch = predict_case(case_torch, probs_torch)

        assert np.allclose(out_np, out_torch, atol=1e-5), (
            "{} output should be the same for numpy and torch inputs".format(
                name))


def test_one_hot_inputs_remain_nearly_one_hot():
    """Sanity check: exact one-hot probabilities stay nearly one-hot after calibration."""
    print("Testing one-hot input behavior...")
    n_classes = 4
    y = np.tile(np.arange(n_classes), 10)
    probs = np.eye(n_classes)[y]

    for case in calibrator_cases(reg_value=1e-3, max_iter=10, device="cpu"):
        name, _, _ = case
        fit_case(case, probs, y)
        out = predict_case(case, probs)

        assert_valid_probabilities(out, probs.shape)
        assert np.all(
            out[np.arange(len(y)), y] > 0.999
        ), "{} should preserve exact one-hot confidence".format(name)


if __name__ == "__main__":
    test_calibrators_predict_valid_probabilities()
    test_calibrators_do_not_worsen_training_nll_without_regularization()
    test_torch_tensor_input_matches_numpy_input()
    test_one_hot_inputs_remain_nearly_one_hot()
    print("All tests passed!")
