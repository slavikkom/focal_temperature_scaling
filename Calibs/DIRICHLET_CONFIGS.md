# Dirichlet Calibration Configurations

This directory contains several Dirichlet-style calibration options. They are
not all equivalent; the main implementation for paper-faithful Dirichlet
calibration is `FullDirichletCalibrator`.
The Dirichlet calibration implementation in `Calibs/fulldirichlet.py`,
`Calibs/diagdirichlet.py`, and `Calibs/multinomial.py` is adapted from
https://github.com/dirichletcal/dirichlet_python . The main adaptation is that
the optimization/autodiff backend was changed from JAX to PyTorch while keeping
the sklearn-style wrappers.

## Recommended Configurations

| Configuration | How to run | Score form | Notes |
| --- | --- | --- | --- |
| Full Dirichlet | `FullDirichletCalibrator(reg_lambda=..., reg_mu=None)` | `W log(p) + b` | Full class-interaction Dirichlet calibration. |
| ODIR | `FullDirichletCalibrator(reg_lambda=..., reg_mu=...)` | `W log(p) + b` | Full Dirichlet with off-diagonal and intercept regularization. |
| Diagonal Dirichlet | `DiagonalDirichletCalibrator(reg_lambda=..., reg_mu=None)` | `w_k log(p_k) + b_k` | Per-class scaling plus intercept. |
| Diagonal ODIR-style | `DiagonalDirichletCalibrator(reg_lambda=..., reg_mu=...)` | `w_k log(p_k) + b_k` | Mostly intercept regularization, since diagonal mode has no off-diagonal weights. |
| Fixed diagonal | `MultinomialRegression(method="FixDiag", ...)` | `alpha log(p_k)` | One shared scaling parameter; essentially temperature scaling on logits. |
| Restricted beta-style variant | `BetaDirichletCalibrator(lambda_l2=...)` | `a_k log(p_k) + b_k log(1 - p_k) + c_k` | One-vs-rest beta-style variant; not full Dirichlet and not ODIR. |

## Usage Examples

```python
from Calibs.fulldirichlet import FullDirichletCalibrator
from Calibs.diagdirichlet import DiagonalDirichletCalibrator
from Calibs.betadirichlet import BetaDirichletCalibrator
from Calibs.multinomial import MultinomialRegression


# Full Dirichlet calibration
cal = FullDirichletCalibrator(reg_lambda=1e-3, reg_mu=None)

# ODIR calibration
cal = FullDirichletCalibrator(reg_lambda=1e-3, reg_mu=1e-3)

# Diagonal Dirichlet calibration
cal = DiagonalDirichletCalibrator(reg_lambda=1e-3, reg_mu=None)

# Diagonal ODIR-style calibration
cal = DiagonalDirichletCalibrator(reg_lambda=1e-3, reg_mu=1e-3)

# Fixed diagonal / temperature-scaling-like calibration
cal = MultinomialRegression(method="FixDiag", reg_lambda=1e-3)

# Restricted beta-style variant
cal = BetaDirichletCalibrator(lambda_l2=1e-3)
```

## Regularization Knobs

### `reg_lambda`

When `reg_mu is None`, `reg_lambda` regularizes all calibration weights:

```text
loss = NLL + reg_lambda * ||weights - target||^2
```

By default, the target is zero. If `reg_format="identity"`, the target is the
identity calibration map.

When `reg_mu` is set, the implementation uses the ODIR-style branch:

```text
loss = NLL
     + reg_lambda * off_diagonal_weight_penalty
     + reg_mu * intercept_penalty
```

So in ODIR, `reg_lambda` penalizes off-diagonal class-interaction weights.

### `reg_mu`

`reg_mu=None` means ordinary Dirichlet regularization.

Setting `reg_mu` to a number enables the ODIR-style branch. In that branch,
`reg_mu` regularizes the intercept terms.

### `reg_norm`

If `True`, regularization strengths are normalized by the number of classes.
This can make regularization grids more comparable across datasets with
different class counts.

### `ref_row`

If `True`, the last row is subtracted from every row of the weight matrix. This
removes softmax redundancy and improves identifiability.

### `initializer`

The default is `initializer="identity"`, which starts optimization near the
identity calibration map. This is usually the safest default.

## Evaluation Runtime Controls

`evaluate.py --dirichlet` evaluates full ODIR calibration on softmax
probabilities. By default it searches a 5-value grid for both `reg_lambda` and
`reg_mu` with 3 validation folds:

```text
5 lambda values * 5 mu values * 3 folds = 75 fits
```

`GridSearchCV(refit=True)` then fits the best setting once more on the full
validation set. Each fit runs PyTorch LBFGS, so this can noticeably increase
evaluation time.

Use these command-line knobs to trade off runtime and search thoroughness:

| Option | Default | Effect |
| --- | --- | --- |
| `--dirichlet-reg-grid` | `1e-1 1e-2 1e-3 1e-4 1e-5` | Sets the shared grid used for both `reg_lambda` and `reg_mu`. Fewer values reduce fits quadratically. |
| `--dirichlet-cv-folds` | `3` | Sets validation folds. Fewer folds reduce fits linearly. |
| `--dirichlet-max-iter` | `1024` | Caps LBFGS iterations per fit. Lower values can speed up runs but may underfit the calibration map. |
| `--dirichlet-n-jobs` | `1` | Runs independent GridSearchCV fits in parallel. Useful on CPU; be careful when using GPU because multiple workers may contend for the same device. |

For a faster exploratory run:

```bash
python evaluate.py --dirichlet \
  --dirichlet-reg-grid 1e-2 1e-3 \
  --dirichlet-max-iter 100 \
  --dirichlet-n-jobs 4
```

This changes the search from 75 CV fits to:

```text
2 lambda values * 2 mu values * 3 folds = 12 fits
```

plus the final refit.

### Modification: LBFGS Regularization Cache

This modification was to improve performance of the code. The PyTorch LBFGS optimizer calls its closure many times during each fit,
especially when using the strong Wolfe line search. The regularization tensors
used by the objective are constant for a given fit: identity masks, zero
columns, and identity targets do not depend on the current LBFGS weights.

`Calibs/multinomial.py` therefore builds this regularization context once per
fit and passes it into the objective instead of recreating `torch.eye`,
`torch.zeros`, and `torch.hstack` on every closure call. This does not change
the calibration objective; it only removes repeated tensor allocation from the
inner optimization loop. The largest runtime savings still usually come from a
smaller grid, fewer iterations, or parallel GridSearchCV workers.

## Near One-Hot Probabilities

All of these calibrators have limited ability to change predictions that are
already exact or nearly exact one-hot probability vectors, such as:

```text
[1.0, 0.0, 0.0]
```

The Dirichlet-style methods operate on `log(p)`, so probabilities are clipped
before taking logs. A one-hot vector becomes approximately:

```text
[1 - eps, eps, eps]
```

and therefore:

```text
log(1 - eps) ~= 0
log(eps)     << 0
```

This creates extremely large score gaps. In practice, full, diagonal, ODIR,
fixed diagonal, and beta-style calibration usually leave these predictions
nearly one-hot after calibration.

This is expected behavior, not necessarily a bug. Once probabilities have been
hardened to zeros and ones, most uncertainty information has already been lost.
For calibration experiments, prefer saving and calibrating from raw logits or
non-hardened softmax probabilities.

## Practical Experiment Set

For most experiments, start with:

```python
# Full Dirichlet
FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)

# ODIR
FullDirichletCalibrator(reg_lambda=reg, reg_mu=reg)

# Diagonal Dirichlet
DiagonalDirichletCalibrator(reg_lambda=reg, reg_mu=None)
```

Treat `BetaDirichletCalibrator` from `betadirichlet.py` as a separate custom
baseline, not as the main full Dirichlet implementation. `DirichletCalibrator`
from `dirichlet.py` remains as a backward-compatible alias.
