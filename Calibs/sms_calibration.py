"""Validation-only SMS selection using the unchanged, repository-local probmetrics."""

import importlib
import importlib.metadata
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold


DEFAULT_SMS_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
PACKAGE_ROOT = Path(__file__).resolve().parent / "probmetrics"


def add_sms_arguments(parser):
    parser.add_argument("--sms", action="store_true",
                        help="Include ridge Structured Matrix Scaling with validation-only CV.")
    parser.add_argument("--sms-reg-grid", nargs="+", type=float, default=list(DEFAULT_SMS_GRID),
                        help="Shared SMS bias/diagonal/off-diagonal strengths; smoke mode uses 1.")
    parser.add_argument("--sms-cv-folds", type=int, default=3,
                        help="Maximum stratified validation folds for SMS (unused in smoke mode).")


def load_sms_class():
    # """Import only on request; never silently use a different installed copy."""
    # if sys.version_info < (3, 10):
    #     raise ImportError("SMS requires Python >=3.10 and PyTorch >=2.0; see Calibs/SMS_CONFIGS.md.")
    import torch
    # if int(torch.__version__.split('.')[0]) < 2:
    #     raise ImportError("SMS requires PyTorch >=2.0; see Calibs/SMS_CONFIGS.md.")
    if not (PACKAGE_ROOT / "probmetrics" / "calibrators" / "logistic.py").is_file():
        raise ImportError("Missing repository-local Calibs/probmetrics implementation.")
    for name, module in list(sys.modules.items()):
        if name == "probmetrics" or name.startswith("probmetrics."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(PACKAGE_ROOT):
                raise ImportError("SMS requires the repository-local probmetrics; restart without another copy loaded.")
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        module = importlib.import_module("probmetrics.calibrators.logistic")
    except ImportError as exc:
        raise ImportError("Cannot load SMS dependencies; see Calibs/SMS_CONFIGS.md. " + str(exc)) from exc
    if not Path(module.__file__).resolve().is_relative_to(PACKAGE_ROOT):
        raise ImportError("SMS resolved to a different probmetrics checkout.")
    return module.SMSCalibrator


def sms_provenance():
    load_sms_class()
    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "numba", "pytorch-minimize", "torchmetrics"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    package = importlib.import_module("probmetrics.__about__")
    return {"implementation": "Calibs/probmetrics", "version": package.__version__,
            "python": sys.version.split()[0], "dependencies": versions}


def predict_sms_log_proba(estimator, probabilities):
    """Preserve upstream preprocessing, evaluating its fitted affine map in float64."""
    scaled = estimator.ts_.predict_proba(np.asarray(probabilities))
    design = estimator._get_logits(scaled).astype(np.float64)
    logits = design @ np.asarray(estimator.W_, dtype=np.float64).T
    if not np.isfinite(logits).all():
        raise ValueError("SMS produced nonfinite calibrated logits.")
    return logits - logsumexp(logits, axis=1, keepdims=True)


def sms_nll(log_probabilities, labels):
    return float(-np.mean(log_probabilities[np.arange(len(labels)), labels]))


def _fit_one(cls, probabilities, labels, strength, warning_messages):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator = cls(penalty="ridge", rho=1.0, tau=1.0, opt="bfgs",
                            lambda_intercept=strength, lambda_diagonal=strength,
                            lambda_off_diagonal=strength)
            estimator.fit(probabilities, labels)
            if not np.isfinite(estimator.W_).all():
                raise ValueError("SMS fitted nonfinite coefficients.")
            return estimator
        finally:
            warning_messages.extend(str(w.message) for w in caught)


def fit_sms(probabilities, labels, reg_grid=None, cv_folds=3, seed=1, smoke_test=False):
    """Return (selected estimator or None, JSON-ready selection diagnostics)."""
    cls = load_sms_class()  # Dependency errors are setup errors, not failed candidates.
    started = time.monotonic()
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    grid = list(DEFAULT_SMS_GRID if reg_grid is None else reg_grid)
    if not grid or any(not np.isfinite(s) or s <= 0 for s in grid):
        raise ValueError("SMS strengths must be finite positive numbers.")
    grid = list(dict.fromkeys(float(s) for s in grid))
    if probabilities.ndim != 2 or not len(probabilities) or labels.shape != (len(probabilities),):
        raise ValueError("SMS requires nonempty (n, K) probabilities and (n,) labels.")
    n, k = probabilities.shape
    if k < 2 or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("SMS labels must be in the probability column range, with K >= 2.")
    result = {"status": "ok", "smoke_test": bool(smoke_test),
              "selection_metric": "fixed_default_smoke_test" if smoke_test else "subset_CV_NLL",
              "configuration": {"penalty": "ridge", "rho": 1.0, "tau": 1.0, "opt": "bfgs"},
              "requested_grid": grid, "param_grid": {"shared_strength": [1.0] if smoke_test else grid},
              "requested_cv_folds": cv_folds, "cv_folds": 0, "seed": int(seed),
              "cv_results": [], "warnings": []}

    def unavailable(reason):
        result.update(status="unavailable", reason=reason, elapsed_seconds=time.monotonic() - started)
        return None, result

    if smoke_test:
        selected_strength = 1.0
        selected_score = None
    else:
        counts = np.bincount(labels, minlength=k)
        folds = min(cv_folds, int(counts.min()))
        if folds < 2:
            return unavailable("SMS CV requires every output class, at least two examples per class, and two folds.")
        result["cv_folds"] = folds
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(probabilities, labels))
        for strength in grid:
            candidate = {"shared_strength": strength, "fold_NLL": [], "warnings": [], "status": "ok"}
            candidate_start = time.monotonic()
            try:
                for train, held_out in splits:
                    fitted = _fit_one(cls, probabilities[train], labels[train], strength, candidate["warnings"])
                    loss = sms_nll(predict_sms_log_proba(fitted, probabilities[held_out]), labels[held_out])
                    if not np.isfinite(loss):
                        raise ValueError("Nonfinite held-out SMS NLL.")
                    candidate["fold_NLL"].append(loss)
                candidate["mean_NLL"] = float(np.mean(candidate["fold_NLL"]))
                candidate["std_NLL"] = float(np.std(candidate["fold_NLL"]))
            except Exception as exc:
                candidate.update(status="unavailable", reason="{}: {}".format(type(exc).__name__, exc))
            candidate["elapsed_seconds"] = time.monotonic() - candidate_start
            result["cv_results"].append(candidate)
        valid = [c for c in result["cv_results"] if c["status"] == "ok"]
        if not valid:
            return unavailable("No SMS candidate completed all folds with finite NLL.")
        best = min(valid, key=lambda c: c["mean_NLL"])
        selected_strength, selected_score = best["shared_strength"], best["mean_NLL"]

    result["best"] = {"shared_strength": selected_strength, "mean_cv_NLL": selected_score,
                      "grid_endpoint": None if smoke_test else selected_strength in (min(grid), max(grid)),
                      "effective_reg_intercept": selected_strength * k / n,
                      "effective_reg_diagonal": selected_strength * k / n,
                      "effective_reg_off_diagonal": selected_strength * k * (k - 1) / n}
    try:
        estimator = _fit_one(cls, probabilities, labels, selected_strength, result["warnings"])
    except Exception as exc:
        return unavailable("Selected SMS refit failed: {}: {}".format(type(exc).__name__, exc))
    result["best"]["temperature"] = float(1.0 / estimator.ts_.cal_.invtemp_)
    result["elapsed_seconds"] = time.monotonic() - started
    return estimator, result
