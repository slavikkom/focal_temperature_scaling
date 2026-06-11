# Analysis Scripts

This directory contains scripts for turning experiment JSON files into paper
tables and Excel workbooks. The two main scripts are:

- `results_to_table2.py`
- `cifar10c_results_to_excel.py`

## `results_to_table2.py`

Generates LaTeX tables and an Excel summary from 
evaluation results stored in JSON files.

This script has three types of outputs:
- summary of all results in terms of an excel file called `<dataset_name>_results.xlsx` which contains performance of each loss with each link function used during post-hoc recalibration different metrics over train, validation, and test sets. 
- trainability table which shows performance of each loss with its corresponding link function and the main metric of interest is accuracy.
- calibration table which shows best performing link function per each loss type and the metric of interest is ECE here.


The script is configured by editing constants near the top of the file:

- `dataset_name`: dataset to summarize, for example `CIFAR10`, `CIFAR100`,
  `TINYIMAGENET`, or supported MedMNIST variants.
- `epoch`: checkpoint/evaluation epoch used in result file names.
- `random_seeds`: seed directories to aggregate.
- `RESULTS_DIR`: root directory containing one subdirectory per seed.
- `params`: training hyperparameter values to include for focal, linear,
  exponential, minus-power, and log-power losses.
- `include_train_performance`: include train/test values in formatted table
  cells when enabled.
- `include_std`: include standard deviations across seeds.
- `bestparam_based_on_ece`: select evaluation-link parameters by validation
  ECE when enabled, otherwise by validation log loss.


The script reads each configured loss/result file, aggregates numeric metrics
across seeds, selects calibration/evaluation-link rows from validation metrics,
prints LaTeX tables to stdout, and writes a workbook named
`<dataset_name>_results.xlsx` in the current working directory.

Run from the `analysis_scripts/` directory so the relative `RESULTS_DIR`
defaults resolve as intended:

```bash
cd analysis_scripts
python results_to_table2.py
```

## `cifar10c_results_to_excel.py`

Creates an Excel workbook summarizing CIFAR-10-C corruption results. It scans
CIFAR-10-C result files, selects the best variant in each loss family by
validation log loss, aggregates metrics across seeds, and writes one summary
sheet plus one sheet per corruption type. By default, metrics are read from the
softmax link, but another evaluated link function can be selected with
`--metric-link-name`. To compare loss families using the best available
evaluation link for each family, use `--best-performing-link-per-loss`.

Expected corrupted-result layout:

```text
RESULTS/CIFAR10C/
  fog-1/
    42/
      resnet50_cross_entropy_350.json
      exp_1mp_alpha_0.5_350.json
      ...
  fog-2/
    42/
      ...
```

Severity `0` is treated as clean CIFAR-10 and is read from
`--clean-results-dir` instead of `--results-dir`.

The workbook contains:

- `Summary`: averages each metric across corruption types for each severity.
- One sheet per corruption, for example `fog`, `brightness`, or `gaussian_noise`.
- Optional detail sheets: `Aggregated Numeric`, `Raw Results`, and
  `Selected Rows`.
- `Warnings`, when files or metrics were missing.

Metrics include accuracy, log loss, ECE, smECE, smECE sigma, smECE with fixed
bandwidth `0.05`, and Brier score. By default cells are formatted as
`mean +/- std`; use `--separate-std-columns` to write mean and standard
deviation into separate columns.

Rows are selected by validation log loss within each loss family and
calibration state. With `--best-performing-link-per-loss`, the selection also
searches over every link function and link value available in the JSON files.
This produces one row per loss family for each requested calibration mode:
uncalibrated, temperature-calibrated, and Dirichlet-calibrated. The selected
link name and value are written to `Selected Rows` and `Raw Results` when
detail sheets are enabled.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--results-dir` | `RESULTS/CIFAR10C` | Directory containing CIFAR-10-C corruption result folders. |
| `--clean-results-dir` | `RESULTS/hpc_results_october/CIFAR10_epoch350` | Directory containing clean CIFAR-10 results used for severity `0`. |
| `--output` | `analysis_scripts/CIFAR10C_results.xlsx` | Path of the Excel workbook to write. |
| `--seeds` | `42 123 2023` | Seed directories to aggregate. |
| `--severities` | `0 1 2 3 4 5` | Severity levels to include. Severity `0` uses clean results. |
| `--splits` | `train val test` | Dataset splits to include. Choices: `train`, `val`, `test`. |
| `--calibrations` | `uncalibrated calibrated dirichlet` | Calibration states to include. Choices: `uncalibrated`, `calibrated`, `dirichlet`. |
| `--cal-criteria` | `ce` | Temperature selection criterion. Choices: `ce`, `ece`. |
| `--metric-link-name` | `softmax` | Link function to read metrics from in the result JSONs, for example `focal`, `focal_linear`, `exp_p`, `exp_1mp`, `one_minus_power`, or `log_power`. |
| `--metric-link-value` | inferred | Fixed link parameter value to read. If omitted, `softmax` uses `1.0`; non-softmax links select the evaluated value with the lowest validation Logloss. |
| `--best-performing-link-per-loss` | disabled | Search all available link functions and link values, then select the best validation-Logloss row per loss family and calibration state. Overrides `--metric-link-name` and `--metric-link-value` for metric collection. |
| `--skip-detail-sheets` | disabled | Skip raw/detail sheets for faster export and smaller workbooks. |
| `--separate-std-columns` | disabled | Write mean and std in separate columns instead of `mean +/- std` cells. |

### Examples

Run with defaults:

```bash
python analysis_scripts/cifar10c_results_to_excel.py
```

Write a workbook for the June 2026 CIFAR-10-C results:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --output analysis_scripts/CIFAR10C_limited_results_with_dirichlet.xlsx
```

Use only a subset of seeds and severities:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --seeds 42 \
  --severities 1 2 3
```

Export only test metrics for calibrated and uncalibrated rows:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --splits test \
  --calibrations uncalibrated calibrated
```

Read metrics from an evaluated `exp_p` link instead of softmax, selecting the
best `exp_p` value by validation Logloss:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --metric-link-name exp_p \
  --output analysis_scripts/CIFAR10C_results_exp_p.xlsx
```

Select the best available link function per loss family separately for
uncalibrated, temperature-calibrated, and Dirichlet-calibrated rows:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --best-performing-link-per-loss \
  --output analysis_scripts/CIFAR10C_results_best_link_per_loss.xlsx
```

Use ECE-selected temperatures and separate mean/std columns:

```bash
python analysis_scripts/cifar10c_results_to_excel.py \
  --results-dir RESULTS/hpc_results_june26/CIFAR10C_epoch350_with_dirichlet \
  --cal-criteria ece \
  --separate-std-columns \
  --output analysis_scripts/CIFAR10C_results_separate_mean_std.xlsx
```
