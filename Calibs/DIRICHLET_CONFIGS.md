# Dirichlet Calibration Configurations

This directory contains several Dirichlet-style calibration options. They are
not all equivalent; the main implementation for paper-faithful Dirichlet
calibration is `FullDirichletCalibrator`.

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
from Calibs.beta_dirichlet import BetaDirichletCalibrator
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

Treat `BetaDirichletCalibrator` from `beta_dirichlet.py` as a separate custom
baseline, not as the main full Dirichlet implementation. `DirichletCalibrator`
from `dirichlet.py` remains as a backward-compatible alias.
