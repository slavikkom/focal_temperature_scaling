#!/usr/bin/env python
"""
Create an Excel workbook summarizing corruption/severity evaluation results.

The expected result layout is:

    RESULTS/CIFAR10C/<corruption>-<severity>/<seed>/<result-file>.json

Clean-only datasets can be exported with --severities 0 and a directory laid
out as:

    RESULTS/TINYIMAGENET_epoch100/<seed>/<result-file>.json

Each corruption worksheet contains metric tables with severities as columns and
model/loss variants as rows. The Summary worksheet averages each metric across
all corruption types for each severity, or shows the clean seed aggregate for
clean-only exports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "RESULTS" / "CIFAR10C"
DEFAULT_CLEAN_RESULTS_DIR = REPO_ROOT / "RESULTS" / "hpc_results_october" / "CIFAR10_epoch350"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "analysis_scripts" / "CIFAR10C_results.xlsx"
DEFAULT_SEVERITIES = (0, 1, 2, 3, 4, 5)
DEFAULT_SEEDS = (42, 123, 2023)
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_CALIBRATIONS = ("uncalibrated", "calibrated", "dirichlet")


@dataclass(frozen=True)
class Approach:
    sort_key: tuple
    approach: str
    loss: str
    file_name: str
    clean_file_name: str
    train_param: float | None
    metric_link_name: str = "softmax"
    metric_link_value: float | None = 1.0


APPROACH_PATTERNS = (
    {
        "loss": "cross_entropy",
        "display": "Cross-Entropy",
        "regex": re.compile(
            r"^(?P<prefix>resnet50(?:_ti)?_)?cross_entropy_(?P<epoch>\d+)\.json$"
        ),
        "clean_file": "{prefix}cross_entropy_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 0,
    },
    {
        "loss": "cross_entropy_label_smoothing",
        "display": "Cross-Entropy label_smoothing={param:g}",
        "regex": re.compile(
            r"^(?P<prefix>resnet50(?:_ti)?_)?cross_entropy_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"
        ),
        "clean_file": "{prefix}cross_entropy_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 0.1,
    },
    {
        "loss": "brier_score",
        "display": "Brier Score",
        "regex": re.compile(
            r"^(?P<prefix>resnet50(?:_ti)?_)?brier_score_(?P<epoch>\d+)\.json$"
        ),
        "clean_file": "{prefix}brier_score_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 0.2,
    },
    {
        "loss": "focal_loss",
        "display": "Focal gamma={param:g}",
        "regex": re.compile(r"^focal_loss_gamma_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "focal_loss_gamma_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 1,
    },
    {
        "loss": "proper_focal_loss",
        "display": "Proper Focal gamma={param:g}",
        "regex": re.compile(r"^proper_focal_loss_gamma_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "proper_focal_loss_gamma_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 1.1,
    },
    {
        "loss": "linear_loss",
        "display": "Linear beta={param:g}",
        "regex": re.compile(
            r"^(?P<prefix>resnet50_|ti_)linear_beta_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"
        ),
        "clean_file": "{prefix}linear_beta_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 2,
    },
    {
        "loss": "exp_p_loss",
        "display": "ExpP alpha={param:g}",
        "regex": re.compile(r"^exp_p_alpha_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "exp_p_alpha_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 3,
    },
    {
        "loss": "exp_1mp_loss",
        "display": "Exp1mp alpha={param:g}",
        "regex": re.compile(r"^exp_1mp_alpha_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "exp_1mp_alpha_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 4,
    },
    {
        "loss": "minus_power_loss",
        "display": "MinusPow beta={param:g}",
        "regex": re.compile(r"^minus_power_beta_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "minus_power_beta_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 5,
    },
    {
        "loss": "log_power_loss",
        "display": "LogPow kappa={param:g}",
        "regex": re.compile(r"^log_power_kappa_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "log_power_kappa_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 6,
    },
)

METRICS = (
    ("Accuracy (%)", "ACC", 1.0, "0.2f"),
    ("Logloss", "CE", 1.0, "0.4f"),
    ("ECE (%)", "ECE", 100.0, "0.2f"),
    ("smECE (%)", "smECE", 100.0, "0.2f"),
    ("smECE sigma", "smECE_sigma", 1.0, "0.4f"),
    ("smECE 0.05 (%)", "smECE_0.05", 100.0, "0.2f"),
    ("Brier", "Brier", 1.0, "0.4f"),
)
PARAMETER_FIELDS = ("Temperature", "Dirichlet lambda", "Dirichlet mu")
AGGREGATED_FIELDS = (
    *METRICS,
    ("Temperature", "Temperature", 1.0, "0.2f"),
    ("Dirichlet lambda", "Dirichlet lambda", 1.0, "0.2g"),
    ("Dirichlet mu", "Dirichlet mu", 1.0, "0.2g"),
)
_CORRUPTION_INDEX_CACHE = {}
_SUMMARY_METRIC_INDEX_CACHE = {}
_SUMMARY_TEMPERATURE_INDEX_CACHE = {}
CLEAN_RESULTS_LABEL = "Clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export corruption/severity or clean-only results to Excel."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing corrupted results, or clean results when "
            f"--severities 0 is used. Default: {DEFAULT_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--clean-results-dir",
        type=Path,
        default=DEFAULT_CLEAN_RESULTS_DIR,
        help=(
            "Directory containing non-corrupted results used as severity 0. "
            f"Default: {DEFAULT_CLEAN_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Workbook path to write. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seed directories to aggregate. Default: 42 123 2023",
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEVERITIES),
        help="Severity levels to include. Default: 0 1 2 3 4 5",
    )
    parser.add_argument(
        "--splits",
        choices=DEFAULT_SPLITS,
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to include in each table cell. Default: train val test",
    )
    parser.add_argument(
        "--calibrations",
        choices=DEFAULT_CALIBRATIONS,
        nargs="+",
        default=list(DEFAULT_CALIBRATIONS),
        help="Calibration states to include as rows. Default: uncalibrated calibrated dirichlet",
    )
    parser.add_argument(
        "--cal-criteria",
        choices=("ce", "ece"),
        default="ce",
        help="Temperature selection criterion for calibrated metrics. Default: ce",
    )
    parser.add_argument(
        "--metric-link-name",
        default="softmax",
        help=(
            "Link function to read metrics from in each result JSON, for example "
            "softmax, focal, focal_linear, exp_p, exp_1mp, one_minus_power, or "
            "log_power. Default: softmax"
        ),
    )
    parser.add_argument(
        "--metric-link-value",
        type=float,
        default=None,
        help=(
            "Fixed link parameter value to read for --metric-link-name. If omitted, "
            "softmax uses 1.0 for every loss; non-softmax links select the "
            "evaluated value with the lowest validation Logloss."
        ),
    )
    parser.add_argument(
        "--best-performing-link-per-loss",
        action="store_true",
        help=(
            "For each loss family and calibration state, select the best "
            "available metric link function and link value by validation Logloss. "
            "This overrides --metric-link-name and --metric-link-value for metric "
            "collection."
        ),
    )
    parser.add_argument(
        "--skip-detail-sheets",
        action="store_true",
        help="Skip Raw Results, Aggregated Numeric, and Selected Rows sheets for faster export.",
    )
    parser.add_argument(
        "--separate-std-columns",
        action="store_true",
        help=(
            "Write mean and std into separate numeric columns in the Summary and "
            "corruption sheets instead of single 'mean +/- std' text cells."
        ),
    )
    return parser.parse_args()


def resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path

    return (REPO_ROOT / path).resolve()


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = path.resolve()
    if cwd_path.parent.exists():
        return cwd_path

    return (REPO_ROOT / path).resolve()


def infer_dataset_label(*paths: Path) -> str:
    for path in paths:
        for part in reversed(path.parts):
            normalized = re.sub(r"[^A-Za-z0-9]", "", part)
            match = re.search(r"cifar(?P<num>\d+)c", normalized, flags=re.IGNORECASE)
            if match is not None:
                return f"CIFAR-{match.group('num')}-C"
            if "TINYIMAGENET" in normalized.upper():
                return "TinyImageNet"
    return "CIFAR-10-C"


def is_clean_only(severities: Iterable[int]) -> bool:
    return set(severities) == {0}


def parse_corruption_dir(path: Path) -> tuple[str, int] | None:
    match = re.match(r"^(?P<corruption>.+)-(?P<severity>[1-5])$", path.name)
    if match is None:
        return None
    return match.group("corruption"), int(match.group("severity"))


def discover_corruptions(results_dir: Path, severities: Iterable[int]) -> list[str]:
    wanted_severities = set(severities)
    corruptions = set()
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        parsed = parse_corruption_dir(path)
        if parsed is None:
            continue
        corruption, severity = parsed
        if severity in wanted_severities:
            corruptions.add(corruption)
    return sorted(corruptions)


def approach_metric_link_value(
    pattern: dict,
    param: float | None,
    metric_link_name: str,
    metric_link_value: float | None,
) -> float | None:
    if metric_link_value is not None:
        return metric_link_value
    if metric_link_name == pattern["metric_link_name"]:
        return float(pattern["metric_link_value"])
    return None


def discover_approaches(
    results_dir: Path,
    metric_link_name: str = "softmax",
    metric_link_value: float | None = None,
) -> list[Approach]:
    file_names = {path.name for path in results_dir.rglob("*.json")}
    approaches = []

    for file_name in sorted(file_names):
        for pattern in APPROACH_PATTERNS:
            match = pattern["regex"].match(file_name)
            if match is None:
                continue

            param_text = match.groupdict().get("param")
            prefix = match.groupdict().get("prefix") or ""
            epoch = match.groupdict()["epoch"]
            param = float(param_text) if param_text is not None else None
            display = pattern["display"].format(param=param if param is not None else 1.0)
            clean_file_name = pattern["clean_file"].format(
                prefix=prefix,
                param=param_text if param_text is not None else "",
                epoch=epoch,
            )
            approaches.append(
                Approach(
                    sort_key=(pattern["sort"], param if param is not None else 0.0, file_name),
                    approach=display,
                    loss=pattern["loss"],
                    file_name=file_name,
                    clean_file_name=clean_file_name,
                    train_param=param,
                    metric_link_name=metric_link_name,
                    metric_link_value=approach_metric_link_value(
                        pattern,
                        param,
                        metric_link_name,
                        metric_link_value,
                    ),
                )
            )
            break

    return sorted(approaches, key=lambda item: item.sort_key)


def link_value_keys(value: float) -> list[str]:
    candidates = [
        str(value),
        str(round(value, 2)),
        f"{value:g}",
        f"{value:.1f}",
        f"{value:.2f}",
    ]
    if float(value).is_integer():
        candidates.insert(0, str(int(value)))
    return list(dict.fromkeys(candidates))


def parse_link_value_key(key: str) -> float | None:
    try:
        return float(key)
    except ValueError:
        return None


def available_metric_link_names(
    data: dict,
    split: str,
    calibration: str,
    cal_criteria: str,
) -> list[str]:
    if calibration == "dirichlet":
        return sorted(data.get("dirichlet_calibrated", {}).keys())
    if calibration == "uncalibrated":
        return sorted(data[split]["uncalibrated"].keys())
    return sorted(data[split]["calibrated"][cal_criteria].keys())


def available_metric_link_values(
    data: dict,
    approach: Approach,
    split: str,
    calibration: str,
    cal_criteria: str,
) -> list[tuple[str | None, float]]:
    if calibration == "dirichlet":
        value = approach.metric_link_value if approach.metric_link_value is not None else 1.0
        return [(None, value)]

    if calibration == "uncalibrated":
        block = data[split]["uncalibrated"][approach.metric_link_name]
    else:
        block = data[split]["calibrated"][cal_criteria][approach.metric_link_name]

    if approach.metric_link_value is not None:
        for key in link_value_keys(approach.metric_link_value):
            if key in block:
                return [(key, approach.metric_link_value)]

        available = ", ".join(block.keys())
        raise KeyError(
            f"Missing link value {approach.metric_link_value:g} "
            f"for {approach.metric_link_name}; "
            f"available values: {available}"
        )

    values = []
    for key in block:
        value = parse_link_value_key(key)
        if value is not None:
            values.append((key, value))
    if not values:
        raise KeyError(f"No numeric values found for link {approach.metric_link_name}.")

    return sorted(values, key=lambda item: item[1])


def ece_value(raw_value) -> float:
    if isinstance(raw_value, dict):
        if "ece" in raw_value:
            return float(np.sum(raw_value["ece"]))
        raise KeyError("ECE dictionary does not contain an 'ece' entry.")
    if isinstance(raw_value, (list, tuple)):
        return float(np.sum(raw_value))
    return float(raw_value)


def calibration_label(calibration: str) -> str:
    if calibration == "uncalibrated":
        return "T=1"
    if calibration == "calibrated":
        return "T=optimal_T"
    if calibration == "dirichlet":
        return "lambda=optimal_lambda, mu=optimal_mu"
    return calibration


def metric_link_label(
    metric_link_name: str,
    metric_link_value: float,
    calibration: str,
) -> str:
    if calibration == "dirichlet":
        return ""
    if metric_link_name == "softmax" and metric_link_value == 1.0:
        return ""
    return f", {metric_link_name}={metric_link_value:g}"


def get_optimal_temperature(
    data: dict,
    approach: Approach,
    metric_link_value: float,
    calibration: str,
    cal_criteria: str,
) -> float:
    if calibration == "uncalibrated":
        return 1.0
    if calibration == "dirichlet":
        return np.nan

    values = data["T_dict"][approach.metric_link_name]
    for key in link_value_keys(metric_link_value):
        if key in values:
            return float(values[key][f" T_opt {cal_criteria}"])

    available = ", ".join(values.keys())
    raise KeyError(
        f"Missing T_dict value {metric_link_value:g} "
        f"for {approach.metric_link_name}; "
        f"available values: {available}"
    )


def get_dirichlet_block(data: dict, metric_link_name: str = "softmax") -> dict:
    return data["dirichlet_calibrated"][metric_link_name]["full_odir"]


def get_dirichlet_parameters(data: dict, metric_link_name: str = "softmax") -> tuple[float, float]:
    best = get_dirichlet_block(data, metric_link_name)["best"]
    return float(best["reg_lambda"]), float(best["reg_mu"])


def get_metric_block(
    data: dict,
    approach: Approach,
    metric_link_key: str | None,
    metric_link_value: float,
    split: str,
    calibration: str,
    cal_criteria: str,
) -> dict:
    if calibration == "uncalibrated":
        block = data[split]["uncalibrated"][approach.metric_link_name]
    elif calibration == "dirichlet":
        return get_dirichlet_block(data, approach.metric_link_name)[split]
    else:
        block = data[split]["calibrated"][cal_criteria][approach.metric_link_name]

    if metric_link_key is not None and metric_link_key in block:
        return block[metric_link_key]

    for key in link_value_keys(metric_link_value):
        if key in block:
            return block[key]

    available = ", ".join(block.keys())
    raise KeyError(
        f"Missing link value {metric_link_value:g} "
        f"for {approach.metric_link_name}; "
        f"available values: {available}"
    )


def collect_records(
    results_dir: Path,
    clean_results_dir: Path,
    approaches: list[Approach],
    corruptions: list[str],
    severities: list[int],
    seeds: list[int],
    splits: list[str],
    calibrations: list[str],
    cal_criteria: str,
    best_performing_link_per_loss: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    records = []
    warnings = []
    json_cache = {}

    for approach in approaches:
        for corruption in corruptions:
            for severity in severities:
                for seed in seeds:
                    if severity == 0:
                        path = clean_results_dir / str(seed) / approach.clean_file_name
                    else:
                        path = (
                            results_dir
                            / f"{corruption}-{severity}"
                            / str(seed)
                            / approach.file_name
                        )
                    if not path.exists():
                        warnings.append(f"Missing file: {path}")
                        continue

                    if path in json_cache:
                        data = json_cache[path]
                    else:
                        try:
                            with path.open("r", encoding="utf-8") as file:
                                data = json.load(file)
                            json_cache[path] = data
                        except Exception as exc:
                            warnings.append(f"Could not read {path}: {exc}")
                            continue

                    for calibration in calibrations:
                        if (
                            calibration == "dirichlet"
                            and "dirichlet_calibrated" not in data
                        ):
                            continue

                        for split in splits:
                            if best_performing_link_per_loss:
                                try:
                                    metric_link_names = available_metric_link_names(
                                        data,
                                        split=split,
                                        calibration=calibration,
                                        cal_criteria=cal_criteria,
                                    )
                                except Exception as exc:
                                    warnings.append(
                                        f"Could not read link names from {path} "
                                        f"({split}, {calibration}): {exc}"
                                    )
                                    continue
                            else:
                                metric_link_names = [approach.metric_link_name]

                            for metric_link_name in metric_link_names:
                                candidate_approach = (
                                    replace(
                                        approach,
                                        metric_link_name=metric_link_name,
                                        metric_link_value=None,
                                    )
                                    if best_performing_link_per_loss
                                    else approach
                                )
                                if calibration == "dirichlet":
                                    candidate_approach = replace(
                                        candidate_approach,
                                        metric_link_value=1.0,
                                    )

                                try:
                                    metric_link_values = available_metric_link_values(
                                        data,
                                        approach=candidate_approach,
                                        split=split,
                                        calibration=calibration,
                                        cal_criteria=cal_criteria,
                                    )
                                except Exception as exc:
                                    warnings.append(
                                        f"Could not read link values from {path} "
                                        f"({split}, {calibration}, {metric_link_name}): {exc}"
                                    )
                                    continue

                                for metric_link_key, metric_link_value in metric_link_values:
                                    try:
                                        temperature = get_optimal_temperature(
                                            data,
                                            approach=candidate_approach,
                                            metric_link_value=metric_link_value,
                                            calibration=calibration,
                                            cal_criteria=cal_criteria,
                                        )
                                        if calibration == "dirichlet":
                                            dirichlet_lambda, dirichlet_mu = (
                                                get_dirichlet_parameters(
                                                    data,
                                                    candidate_approach.metric_link_name,
                                                )
                                            )
                                        else:
                                            dirichlet_lambda, dirichlet_mu = np.nan, np.nan
                                        metric_block = get_metric_block(
                                            data,
                                            approach=candidate_approach,
                                            metric_link_key=metric_link_key,
                                            metric_link_value=metric_link_value,
                                            split=split,
                                            calibration=calibration,
                                            cal_criteria=cal_criteria,
                                        )
                                    except Exception as exc:
                                        warnings.append(
                                            f"Could not read {path} "
                                            f"({split}, {calibration}, "
                                            f"{candidate_approach.metric_link_name}={metric_link_value:g}): {exc}"
                                        )
                                        continue

                                    row = {
                                        "approach": approach.approach,
                                        "approach_calibration": (
                                            f"{approach.approach} ("
                                            f"{calibration_label(calibration)}"
                                            f"{metric_link_label(candidate_approach.metric_link_name, metric_link_value, calibration)}"
                                            ")"
                                        ),
                                        "calibration": calibration,
                                        "split": split,
                                        "loss": approach.loss,
                                        "file_name": approach.file_name,
                                        "train_param": approach.train_param,
                                        "metric_link_name": candidate_approach.metric_link_name,
                                        "metric_link_value": metric_link_value,
                                        "corruption": corruption,
                                        "severity": severity,
                                        "seed": seed,
                                        "Temperature": temperature,
                                        "Dirichlet lambda": dirichlet_lambda,
                                        "Dirichlet mu": dirichlet_mu,
                                    }
                                    for display_name, raw_name, scale, _ in METRICS:
                                        if raw_name not in metric_block:
                                            row[display_name] = np.nan
                                            continue
                                        if raw_name == "ECE":
                                            value = ece_value(metric_block[raw_name])
                                        else:
                                            value = float(metric_block[raw_name])
                                        row[display_name] = value * scale
                                    records.append(row)

    return pd.DataFrame(records), warnings


def selection_loss_name(loss_name: str) -> str:
    if loss_name == "cross_entropy":
        return "cross_entropy"
    return loss_name


def select_best_validation_logloss(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = raw_df[raw_df["split"] == "val"].copy()
    val_df["selection_loss"] = val_df["loss"].map(selection_loss_name)
    val_df = val_df.drop_duplicates(
        [
            "selection_loss",
            "calibration",
            "approach",
            "approach_calibration",
            "file_name",
            "metric_link_name",
            "metric_link_value",
            "seed",
        ]
    )

    candidates = (
        val_df.groupby(
            [
                "selection_loss",
                "calibration",
                "approach",
                "approach_calibration",
                "file_name",
                "metric_link_name",
                "metric_link_value",
            ],
            sort=False,
        )["Logloss"]
        .mean()
        .reset_index(name="validation_logloss")
    )

    selected_rows = []
    for _, group in candidates.groupby(["selection_loss", "calibration"], sort=False):
        selected_rows.append(group.sort_values(["validation_logloss", "approach"]).iloc[0])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    filtered_df = raw_df.merge(
        selected_df[["approach", "calibration", "metric_link_name", "metric_link_value"]],
        on=["approach", "calibration", "metric_link_name", "metric_link_value"],
        how="inner",
    )

    return filtered_df, selected_df


def aggregate_by_seed(raw_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = [metric[0] for metric in AGGREGATED_FIELDS]
    grouped = raw_df.groupby(
        ["approach_calibration", "approach", "calibration", "split", "corruption", "severity"],
        sort=False,
    )[metric_names]
    agg = grouped.agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns
    ]
    return agg


def summary_across_corruptions(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name, _, _, _ in AGGREGATED_FIELDS:
        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        for (approach_calibration, approach, calibration, split, severity), group in agg_df.groupby(
            ["approach_calibration", "approach", "calibration", "split", "severity"],
            sort=False,
        ):
            values = group[mean_col].dropna()
            if metric_name in PARAMETER_FIELDS or split != "test" or severity == 0:
                std_value = group[std_col].dropna().mean()
            else:
                std_value = values.std(ddof=1) if len(values) > 1 else np.nan
            rows.append(
                {
                    "approach_calibration": approach_calibration,
                    "approach": approach,
                    "calibration": calibration,
                    "split": split,
                    "severity": severity,
                    "metric": metric_name,
                    "mean": values.mean() if len(values) else np.nan,
                    "std": std_value,
                    "count": int(len(values)),
                }
            )
        for (approach_calibration, approach, calibration, split), group in agg_df.groupby(
            ["approach_calibration", "approach", "calibration", "split"],
            sort=False,
        ):
            overall_specs = [
                ("Overall Corrupted", group[group["severity"] != 0]),
                ("Overall", group),
            ]
            for label, label_group in overall_specs:
                values = label_group[mean_col].dropna()
                if metric_name in PARAMETER_FIELDS or split != "test":
                    std_value = label_group[std_col].dropna().mean()
                else:
                    std_value = values.std(ddof=1) if len(values) > 1 else np.nan
                rows.append(
                    {
                        "approach_calibration": approach_calibration,
                        "approach": approach,
                        "calibration": calibration,
                        "split": split,
                        "severity": label,
                        "metric": metric_name,
                        "mean": values.mean() if len(values) else np.nan,
                        "std": std_value,
                        "count": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def format_number(value: float, fmt: str) -> str:
    if value is None or pd.isna(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, fmt)


def format_cell(mean: float, std: float, count: int, fmt: str) -> str:
    mean_text = format_number(mean, fmt)
    if not mean_text:
        return ""
    if count <= 1 or pd.isna(std):
        return mean_text
    return f"{mean_text} +/- {format_number(std, fmt)}"


def split_label(split: str) -> str:
    return {"train": "train", "val": "val", "test": "test"}.get(split, split)


def format_metric_from_row(row: pd.Series, metric_name: str, fmt: str) -> str:
    return format_cell(
        row[f"{metric_name}_mean"],
        row[f"{metric_name}_std"],
        int(row[f"{metric_name}_count"]),
        fmt,
    )


def format_summary_metric_from_row(row: pd.Series, fmt: str) -> str:
    return format_cell(row["mean"], row["std"], int(row["count"]), fmt)


def export_std(std: float, count: int):
    if count <= 1 or pd.isna(std):
        return ""
    return std


def metric_values_from_row(row: pd.Series | None, metric_name: str) -> tuple[float | str, float | str]:
    if row is None:
        return "", ""
    count = int(row[f"{metric_name}_count"])
    mean = row[f"{metric_name}_mean"]
    if pd.isna(mean):
        return "", ""
    return mean, export_std(row[f"{metric_name}_std"], count)


def summary_metric_values_from_row(row: pd.Series | None) -> tuple[float | str, float | str]:
    if row is None:
        return "", ""
    count = int(row["count"])
    mean = row["mean"]
    if pd.isna(mean):
        return "", ""
    return mean, export_std(row["std"], count)


def format_parameter_from_agg_row(
    row: pd.Series | None,
    metric_name: str,
    fmt: str,
) -> str:
    if row is None:
        return ""
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    count_col = f"{metric_name}_count"
    if mean_col not in row or pd.isna(row[mean_col]):
        return ""
    return format_cell(
        row[mean_col],
        row[std_col],
        int(row[count_col]),
        fmt,
    )


def format_parameter_from_summary_row(row: pd.Series | None, fmt: str) -> str:
    if row is None:
        return ""
    if pd.isna(row["mean"]):
        return ""
    return format_summary_metric_from_row(row, fmt)


def approach_with_parameters(
    approach: str,
    temperature: str = "",
    dirichlet_lambda: str = "",
    dirichlet_mu: str = "",
) -> str:
    if "T=optimal_T" in approach and temperature:
        approach = approach.replace("T=optimal_T", f"T={temperature}")
    if "lambda=optimal_lambda" in approach and dirichlet_lambda:
        approach = approach.replace("lambda=optimal_lambda", f"lambda={dirichlet_lambda}")
    if "mu=optimal_mu" in approach and dirichlet_mu:
        approach = approach.replace("mu=optimal_mu", f"mu={dirichlet_mu}")
    return approach


def format_split_value(
    rows: pd.DataFrame,
    metric_name: str,
    fmt: str,
    split: str,
) -> str:
    split_rows = rows[rows["split"] == split]
    if split_rows.empty:
        return ""

    if "severity" in split_rows.columns:
        split_rows = split_rows.sort_values("severity", key=lambda col: col.astype(str))
    return format_metric_from_row(split_rows.iloc[0], metric_name, fmt)


def format_summary_split_value(
    rows: pd.DataFrame,
    metric_name: str,
    fmt: str,
    split: str,
) -> str:
    metric_rows = rows[rows["metric"] == metric_name]
    split_rows = metric_rows[metric_rows["split"] == split]
    if split_rows.empty:
        return ""

    split_rows = split_rows.sort_values("severity", key=lambda col: col.astype(str))
    return format_summary_metric_from_row(split_rows.iloc[0], fmt)


def first_table_severity(severities: list[int]) -> int:
    return sorted(severities)[0]


def index_lookup(indexed_df: pd.DataFrame, key: tuple):
    try:
        row = indexed_df.loc[key]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def format_overall_from_rows(
    rows: list[pd.Series],
    metric_name: str,
    fmt: str,
) -> str:
    values = [
        row[f"{metric_name}_mean"]
        for row in rows
        if row is not None and pd.notna(row[f"{metric_name}_mean"])
    ]
    if not values:
        return ""

    values = pd.Series(values)
    return format_cell(
        values.mean(),
        values.std(ddof=1) if len(values) > 1 else np.nan,
        int(len(values)),
        fmt,
    )


def overall_values_from_rows(
    rows: list[pd.Series],
    metric_name: str,
) -> tuple[float | str, float | str]:
    values = [
        row[f"{metric_name}_mean"]
        for row in rows
        if row is not None and pd.notna(row[f"{metric_name}_mean"])
    ]
    if not values:
        return "", ""

    values = pd.Series(values)
    std = values.std(ddof=1) if len(values) > 1 else np.nan
    return values.mean(), export_std(std, len(values))


def set_mean_std_columns(
    row: dict,
    column_name: str,
    mean: float | str,
    std: float | str,
) -> None:
    row[f"{column_name} Mean"] = mean
    row[f"{column_name} Std"] = std


def corruption_index(agg_df: pd.DataFrame, corruption: str) -> pd.DataFrame:
    key = (id(agg_df), corruption)
    if key not in _CORRUPTION_INDEX_CACHE:
        subset = agg_df[agg_df["corruption"] == corruption]
        _CORRUPTION_INDEX_CACHE[key] = subset.set_index(
            ["approach_calibration", "split", "severity"], drop=False
        )
    return _CORRUPTION_INDEX_CACHE[key]


def summary_metric_index(summary_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    key = (id(summary_df), metric_name)
    if key not in _SUMMARY_METRIC_INDEX_CACHE:
        metric_summary = summary_df[summary_df["metric"] == metric_name]
        _SUMMARY_METRIC_INDEX_CACHE[key] = metric_summary.set_index(
            ["approach_calibration", "split", "severity"], drop=False
        )
    return _SUMMARY_METRIC_INDEX_CACHE[key]


def summary_temperature_index(summary_df: pd.DataFrame) -> pd.DataFrame:
    key = id(summary_df)
    if key not in _SUMMARY_TEMPERATURE_INDEX_CACHE:
        temp_summary = summary_df[summary_df["metric"].isin(PARAMETER_FIELDS)]
        _SUMMARY_TEMPERATURE_INDEX_CACHE[key] = temp_summary.set_index(
            ["approach_calibration", "split", "severity", "metric"], drop=False
        )
    return _SUMMARY_TEMPERATURE_INDEX_CACHE[key]


def metric_table_for_corruption(
    agg_df: pd.DataFrame,
    metric_name: str,
    fmt: str,
    approaches: list[str],
    severities: list[int],
    splits: list[str],
    corruption: str,
    separate_std_columns: bool = False,
) -> pd.DataFrame:
    rows = []
    indexed = corruption_index(agg_df, corruption)
    first_severity = first_table_severity(severities)

    for approach in approaches:
        temp_row = index_lookup(indexed, (approach, "test", first_severity))
        train_row = index_lookup(indexed, (approach, "train", first_severity))
        val_row = index_lookup(indexed, (approach, "val", first_severity))
        row = {
            "Approach": approach_with_parameters(
                approach,
                temperature=format_parameter_from_agg_row(temp_row, "Temperature", "0.2f"),
                dirichlet_lambda=format_parameter_from_agg_row(
                    temp_row, "Dirichlet lambda", "0.2g"
                ),
                dirichlet_mu=format_parameter_from_agg_row(
                    temp_row, "Dirichlet mu", "0.2g"
                ),
            ),
        }
        if separate_std_columns:
            set_mean_std_columns(row, "Train", *metric_values_from_row(train_row, metric_name))
            set_mean_std_columns(
                row, "Validation", *metric_values_from_row(val_row, metric_name)
            )
        else:
            row["Train"] = (
                format_metric_from_row(train_row, metric_name, fmt)
                if train_row is not None
                else ""
            )
            row["Validation"] = (
                format_metric_from_row(val_row, metric_name, fmt)
                if val_row is not None
                else ""
            )

        for severity in severities:
            test_row = index_lookup(indexed, (approach, "test", severity))
            column_name = f"Severity {severity}"
            if separate_std_columns:
                set_mean_std_columns(
                    row, column_name, *metric_values_from_row(test_row, metric_name)
                )
            else:
                if test_row is None:
                    row[column_name] = ""
                    continue
                row[column_name] = format_metric_from_row(test_row, metric_name, fmt)

        if not is_clean_only(severities):
            corrupted_severities = [severity for severity in severities if severity != 0]
            corrupted_rows = [
                index_lookup(indexed, (approach, "test", severity))
                for severity in corrupted_severities
            ]
            overall_rows = [
                index_lookup(indexed, (approach, "test", severity))
                for severity in severities
            ]
            if separate_std_columns:
                set_mean_std_columns(
                    row,
                    "Overall Corrupted",
                    *overall_values_from_rows(corrupted_rows, metric_name),
                )
                set_mean_std_columns(
                    row, "Overall", *overall_values_from_rows(overall_rows, metric_name)
                )
            else:
                row["Overall Corrupted"] = format_overall_from_rows(
                    corrupted_rows, metric_name, fmt
                )
                row["Overall"] = format_overall_from_rows(overall_rows, metric_name, fmt)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_table_for_summary(
    summary_df: pd.DataFrame,
    metric_name: str,
    fmt: str,
    approaches: list[str],
    severities: list[int],
    splits: list[str],
    separate_std_columns: bool = False,
) -> pd.DataFrame:
    rows = []
    metric_index = summary_metric_index(summary_df, metric_name)
    temp_index = summary_temperature_index(summary_df)
    columns = [*severities]
    if not is_clean_only(severities):
        columns.extend(["Overall Corrupted", "Overall"])
    first_severity = first_table_severity(severities)

    for approach in approaches:
        temp_row = index_lookup(temp_index, (approach, "test", first_severity, "Temperature"))
        lambda_row = index_lookup(
            temp_index, (approach, "test", first_severity, "Dirichlet lambda")
        )
        mu_row = index_lookup(temp_index, (approach, "test", first_severity, "Dirichlet mu"))
        train_row = index_lookup(metric_index, (approach, "train", first_severity))
        val_row = index_lookup(metric_index, (approach, "val", first_severity))
        row = {
            "Approach": approach_with_parameters(
                approach,
                temperature=format_parameter_from_summary_row(temp_row, "0.2f"),
                dirichlet_lambda=format_parameter_from_summary_row(lambda_row, "0.2g"),
                dirichlet_mu=format_parameter_from_summary_row(mu_row, "0.2g"),
            ),
        }
        if separate_std_columns:
            set_mean_std_columns(row, "Train", *summary_metric_values_from_row(train_row))
            set_mean_std_columns(
                row, "Validation", *summary_metric_values_from_row(val_row)
            )
        else:
            row["Train"] = (
                format_summary_metric_from_row(train_row, fmt)
                if train_row is not None
                else ""
            )
            row["Validation"] = (
                format_summary_metric_from_row(val_row, fmt)
                if val_row is not None
                else ""
            )

        for severity in columns:
            column_name = f"Severity {severity}" if isinstance(severity, int) else severity
            item = index_lookup(metric_index, (approach, "test", severity))
            if separate_std_columns:
                set_mean_std_columns(row, column_name, *summary_metric_values_from_row(item))
            else:
                if item is None:
                    row[column_name] = ""
                    continue
                row[column_name] = format_cell(
                    item["mean"], item["std"], int(item["count"]), fmt
                )
        rows.append(row)
    return pd.DataFrame(rows)


def safe_sheet_name(name: str) -> str:
    clean = re.sub(r"[\[\]:*?/\\]", "_", name)
    return clean[:31]


def write_metric_blocks(
    writer: pd.ExcelWriter,
    sheet_name: str,
    tables: list[tuple[str, pd.DataFrame]],
    title: str,
    note: str | None = None,
) -> None:
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    title_format = workbook.add_format({"bold": True, "font_size": 14})
    note_format = workbook.add_format({"italic": True, "font_color": "#666666"})
    metric_format = workbook.add_format({"bold": True, "bg_color": "#D9EAF7"})
    header_format = workbook.add_format({"bold": True, "bg_color": "#F2F2F2"})
    cell_format = workbook.add_format({"text_wrap": True, "valign": "top"})

    row = 0
    worksheet.write(row, 0, title, title_format)
    row += 1
    if note:
        worksheet.write(row, 0, note, note_format)
        row += 2
    else:
        row += 1

    for metric_name, table in tables:
        worksheet.write(row, 0, metric_name, metric_format)
        row += 1

        worksheet.write_row(row, 0, list(table.columns), header_format)
        row += 1

        for _, values in table.iterrows():
            worksheet.write_row(row, 0, list(values), cell_format)
            worksheet.set_row(row, 36)
            row += 1
        row += 2

    worksheet.freeze_panes(4, 1)
    worksheet.set_column(0, 0, 28)
    worksheet.set_column(1, 2, 18)
    worksheet.set_column(3, 10, 18)


def write_workbook(
    output_path: Path,
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    approaches: list[Approach],
    selected_df: pd.DataFrame,
    corruptions: list[str],
    severities: list[int],
    args: argparse.Namespace,
    warnings: list[str],
    dataset_label: str,
    clean_only: bool,
) -> None:
    approach_order = selected_df["approach_calibration"].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        if args.best_performing_link_per_loss:
            metric_link_note = (
                "Metrics are read from the best available link function and link "
                "value per loss family and calibration state, selected by lowest "
                "validation Logloss."
            )
        elif args.metric_link_value is None:
            if args.metric_link_name == "softmax":
                metric_link_note = "Metrics are read from softmax with link value 1.0."
            else:
                metric_link_note = (
                    f"Metrics are read from link '{args.metric_link_name}'. "
                    "The link value is selected by lowest validation Logloss."
                )
        else:
            metric_link_note = (
                f"Metrics are read from link '{args.metric_link_name}' "
                f"with fixed value {args.metric_link_value:g}."
            )
        summary_mean_note = (
            "Values are mean +/- std across seeds. "
            if clean_only
            else "Values are mean +/- std across corruption-level means. "
        )
        test_columns_note = (
            "The severity 0 column shows test. "
            if clean_only
            else "Severity columns show test only. "
        )
        summary_note = (
            f"{summary_mean_note}"
            "smECE sigma is the adaptive bandwidth selected for smECE; "
            "smECE 0.05 uses the fixed bandwidth. "
            f"Train and validation are shown in separate columns; {test_columns_note}"
            "Rows are selected by lowest validation Logloss within each loss family and calibration state. "
            f"Calibrated rows use {args.cal_criteria.upper()}-selected temperature scaling. "
            f"{metric_link_note} "
            "Dirichlet rows use softmax full ODIR with selected lambda and mu."
        )
        corruption_note = (
            "Values are mean +/- std across seeds. "
            "smECE sigma is the adaptive bandwidth selected for smECE; "
            "smECE 0.05 uses the fixed bandwidth. "
            "Train and validation are shown in separate columns; severity columns show test only. "
            "Rows are selected by lowest validation Logloss within each loss family and calibration state. "
            f"Calibrated rows use {args.cal_criteria.upper()}-selected temperature scaling. "
            f"{metric_link_note} "
            "Dirichlet rows use softmax full ODIR with selected lambda and mu."
        )
        if clean_only:
            summary_tables = [
                (
                    metric_name,
                    metric_table_for_corruption(
                        agg_df,
                        metric_name,
                        fmt,
                        approach_order,
                        severities,
                        args.splits,
                        CLEAN_RESULTS_LABEL,
                        args.separate_std_columns,
                    ),
                )
                for metric_name, _, _, fmt in METRICS
            ]
            summary_title = f"{dataset_label} clean results"
        else:
            summary_tables = [
                (
                    metric_name,
                    metric_table_for_summary(
                        summary_df,
                        metric_name,
                        fmt,
                        approach_order,
                        severities,
                        args.splits,
                        args.separate_std_columns,
                    ),
                )
                for metric_name, _, _, fmt in METRICS
            ]
            summary_title = f"Average across {dataset_label} corruptions"
        write_metric_blocks(
            writer,
            "Summary",
            summary_tables,
            summary_title,
            summary_note,
        )

        for corruption in ([] if clean_only else corruptions):
            tables = [
                (
                    metric_name,
                    metric_table_for_corruption(
                        agg_df,
                        metric_name,
                        fmt,
                        approach_order,
                        severities,
                        args.splits,
                        corruption,
                        args.separate_std_columns,
                    ),
                )
                for metric_name, _, _, fmt in METRICS
            ]
            write_metric_blocks(
                writer,
                safe_sheet_name(corruption),
                tables,
                f"Corruption: {corruption}",
                corruption_note,
            )

        if not args.skip_detail_sheets:
            agg_df.to_excel(writer, sheet_name="Aggregated Numeric", index=False)
            raw_df.to_excel(writer, sheet_name="Raw Results", index=False)
            selected_df.to_excel(writer, sheet_name="Selected Rows", index=False)

        if warnings:
            pd.DataFrame({"warning": warnings}).to_excel(
                writer, sheet_name="Warnings", index=False
            )


def main() -> None:
    args = parse_args()
    results_dir = resolve_existing_path(args.results_dir)
    clean_results_dir = resolve_existing_path(args.clean_results_dir)
    output_path = resolve_output_path(args.output)
    severities = sorted(args.severities)
    clean_only = is_clean_only(severities)

    if (
        clean_only
        and args.clean_results_dir == DEFAULT_CLEAN_RESULTS_DIR
        and args.results_dir != DEFAULT_RESULTS_DIR
    ):
        clean_results_dir = results_dir

    dataset_label = infer_dataset_label(clean_results_dir, results_dir, output_path)

    if not clean_only and not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if 0 in args.severities and not clean_results_dir.exists():
        raise FileNotFoundError(f"Clean results directory does not exist: {clean_results_dir}")

    seeds = sorted(args.seeds)
    corruptions = [CLEAN_RESULTS_LABEL] if clean_only else discover_corruptions(
        results_dir, severities
    )
    approach_discovery_dir = clean_results_dir if clean_only else results_dir
    approaches = discover_approaches(
        approach_discovery_dir,
        metric_link_name=args.metric_link_name,
        metric_link_value=args.metric_link_value,
    )

    if not corruptions:
        raise RuntimeError(f"No {dataset_label} corruption directories found in {results_dir}")
    if not approaches:
        raise RuntimeError(
            f"No known {dataset_label} result JSON files found in {approach_discovery_dir}"
        )

    selection_severity = next((severity for severity in severities if severity != 0), severities[0])
    selection_severities = [selection_severity]
    selection_raw_df, selection_warnings = collect_records(
        results_dir=results_dir,
        clean_results_dir=clean_results_dir,
        approaches=approaches,
        corruptions=[corruptions[0]],
        severities=selection_severities,
        seeds=seeds,
        splits=["val"],
        calibrations=args.calibrations,
        cal_criteria=args.cal_criteria,
        best_performing_link_per_loss=args.best_performing_link_per_loss,
    )
    if selection_raw_df.empty:
        raise RuntimeError("No readable validation records were found for parameter selection.")

    _, selected_df = select_best_validation_logloss(selection_raw_df)
    selected_approach_names = set(selected_df["approach"])
    selected_approaches = [
        approach for approach in approaches if approach.approach in selected_approach_names
    ]

    raw_df, final_warnings = collect_records(
        results_dir=results_dir,
        clean_results_dir=clean_results_dir,
        approaches=selected_approaches,
        corruptions=corruptions,
        severities=severities,
        seeds=seeds,
        splits=args.splits,
        calibrations=args.calibrations,
        cal_criteria=args.cal_criteria,
        best_performing_link_per_loss=args.best_performing_link_per_loss,
    )
    raw_df = raw_df.merge(
        selected_df[["approach", "calibration", "metric_link_name", "metric_link_value"]],
        on=["approach", "calibration", "metric_link_name", "metric_link_value"],
        how="inner",
    )
    warnings = selection_warnings + final_warnings
    if raw_df.empty:
        raise RuntimeError("No result records remained after validation-Logloss selection.")

    agg_df = aggregate_by_seed(raw_df)
    summary_df = summary_across_corruptions(agg_df)
    write_workbook(
        output_path=output_path,
        raw_df=raw_df,
        agg_df=agg_df,
        summary_df=summary_df,
        approaches=approaches,
        selected_df=selected_df,
        corruptions=corruptions,
        severities=severities,
        args=args,
        warnings=warnings,
        dataset_label=dataset_label,
        clean_only=clean_only,
    )

    print(f"Wrote {output_path}")
    if args.best_performing_link_per_loss:
        print("Metric link: best available per loss family/calibration (validation Logloss)")
    elif args.metric_link_value is None:
        if args.metric_link_name == "softmax":
            print(f"Metric link: {args.metric_link_name} (value 1.0)")
        else:
            print(f"Metric link: {args.metric_link_name} (selected by validation Logloss)")
    else:
        print(f"Metric link: {args.metric_link_name} ({args.metric_link_value:g})")
    print(f"Approaches: {', '.join(approach.approach for approach in approaches)}")
    print(f"Selected rows: {len(selected_df)}")
    if clean_only:
        print("Mode: clean results only")
    else:
        print(f"Corruptions: {len(corruptions)}")
    print(f"Rows read: {len(raw_df)}")
    if warnings:
        print(f"Warnings: {len(warnings)} (also written to the Warnings sheet)")


if __name__ == "__main__":
    main()
