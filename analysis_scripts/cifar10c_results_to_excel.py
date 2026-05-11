#!/usr/bin/env python
"""
Create an Excel workbook summarizing CIFAR-10-C evaluation results.

The expected result layout is:

    RESULTS/CIFAR10C/<corruption>-<severity>/<seed>/<result-file>.json

Each corruption worksheet contains metric tables with severities as columns and
model/loss variants as rows. The Summary worksheet averages each metric across
all corruption types for each severity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
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
DEFAULT_CALIBRATIONS = ("uncalibrated", "calibrated")


@dataclass(frozen=True)
class Approach:
    sort_key: tuple
    approach: str
    loss: str
    file_name: str
    clean_file_name: str
    train_param: float | None
    metric_link_name: str = "softmax"
    metric_link_value: float = 1.0


APPROACH_PATTERNS = (
    {
        "loss": "cross_entropy",
        "display": "Cross-Entropy",
        "regex": re.compile(r"^resnet50_cross_entropy_(?P<epoch>\d+)\.json$"),
        "clean_file": "resnet50_cross_entropy_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 0,
    },
    {
        "loss": "cross_entropy_label_smoothing",
        "display": "Cross-Entropy label_smoothing={param:g}",
        "regex": re.compile(r"^resnet50_cross_entropy_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "cross_entropy_{param}_{epoch}.json",
        "metric_link_name": "softmax",
        "metric_link_value": 1.0,
        "sort": 0.1,
    },
    {
        "loss": "brier_score",
        "display": "Brier Score",
        "regex": re.compile(r"^resnet50_brier_score_(?P<epoch>\d+)\.json$"),
        "clean_file": "brier_score_{epoch}.json",
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
        "loss": "linear_loss",
        "display": "Linear beta={param:g}",
        "regex": re.compile(r"^resnet50_linear_beta_(?P<param>[0-9.]+)_(?P<epoch>\d+)\.json$"),
        "clean_file": "resnet50_linear_beta_{param}_{epoch}.json",
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
AGGREGATED_FIELDS = (*METRICS, ("Temperature", "Temperature", 1.0, "0.2f"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CIFAR-10-C corruption/severity results to Excel."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing CIFAR-10-C results. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--clean-results-dir",
        type=Path,
        default=DEFAULT_CLEAN_RESULTS_DIR,
        help=(
            "Directory containing non-corrupted CIFAR-10 results used as severity 0. "
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
        help="Calibration states to include as rows. Default: uncalibrated calibrated",
    )
    parser.add_argument(
        "--cal-criteria",
        choices=("ce", "ece"),
        default="ce",
        help="Temperature selection criterion for calibrated metrics. Default: ce",
    )
    return parser.parse_args()


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


def discover_approaches(results_dir: Path) -> list[Approach]:
    file_names = {path.name for path in results_dir.rglob("*.json")}
    approaches = []

    for file_name in sorted(file_names):
        for pattern in APPROACH_PATTERNS:
            match = pattern["regex"].match(file_name)
            if match is None:
                continue

            param_text = match.groupdict().get("param")
            epoch = match.groupdict()["epoch"]
            param = float(param_text) if param_text is not None else None
            display = pattern["display"].format(param=param if param is not None else 1.0)
            clean_file_name = pattern["clean_file"].format(
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
                    metric_link_name=pattern["metric_link_name"],
                    metric_link_value=pattern["metric_link_value"],
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
    return calibration


def get_optimal_temperature(
    data: dict,
    approach: Approach,
    calibration: str,
    cal_criteria: str,
) -> float:
    if calibration == "uncalibrated":
        return 1.0

    values = data["T_dict"][approach.metric_link_name]
    for key in link_value_keys(approach.metric_link_value):
        if key in values:
            return float(values[key][f" T_opt {cal_criteria}"])

    available = ", ".join(values.keys())
    raise KeyError(
        f"Missing T_dict value {approach.metric_link_value:g} "
        f"for {approach.metric_link_name}; "
        f"available values: {available}"
    )


def get_metric_block(
    data: dict,
    approach: Approach,
    split: str,
    calibration: str,
    cal_criteria: str,
) -> dict:
    if calibration == "uncalibrated":
        block = data[split]["uncalibrated"][approach.metric_link_name]
    else:
        block = data[split]["calibrated"][cal_criteria][approach.metric_link_name]

    for key in link_value_keys(approach.metric_link_value):
        if key in block:
            return block[key]

    available = ", ".join(block.keys())
    raise KeyError(
        f"Missing link value {approach.metric_link_value:g} "
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
) -> tuple[pd.DataFrame, list[str]]:
    records = []
    warnings = []

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

                    try:
                        with path.open("r", encoding="utf-8") as file:
                            data = json.load(file)
                    except Exception as exc:
                        warnings.append(f"Could not read {path}: {exc}")
                        continue

                    for calibration in calibrations:
                        try:
                            temperature = get_optimal_temperature(
                                data,
                                approach=approach,
                                calibration=calibration,
                                cal_criteria=cal_criteria,
                            )
                        except Exception as exc:
                            warnings.append(
                                f"Could not read temperature from {path} "
                                f"({calibration}): {exc}"
                            )
                            continue

                        for split in splits:
                            try:
                                metric_block = get_metric_block(
                                    data,
                                    approach=approach,
                                    split=split,
                                    calibration=calibration,
                                    cal_criteria=cal_criteria,
                                )
                            except Exception as exc:
                                warnings.append(
                                    f"Could not read {path} "
                                    f"({split}, {calibration}): {exc}"
                                )
                                continue

                            row = {
                                "approach": approach.approach,
                                "approach_calibration": (
                                    f"{approach.approach} ({calibration_label(calibration)})"
                                ),
                                "calibration": calibration,
                                "split": split,
                                "loss": approach.loss,
                                "file_name": approach.file_name,
                                "train_param": approach.train_param,
                                "metric_link_name": approach.metric_link_name,
                                "metric_link_value": approach.metric_link_value,
                                "corruption": corruption,
                                "severity": severity,
                                "seed": seed,
                                "Temperature": temperature,
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
    if loss_name in {"cross_entropy", "cross_entropy_label_smoothing"}:
        return "cross_entropy"
    return loss_name


def select_best_validation_logloss(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = raw_df[raw_df["split"] == "val"].copy()
    val_df["selection_loss"] = val_df["loss"].map(selection_loss_name)
    val_df = val_df.drop_duplicates(
        ["selection_loss", "calibration", "approach", "approach_calibration", "file_name", "seed"]
    )

    candidates = (
        val_df.groupby(
            [
                "selection_loss",
                "calibration",
                "approach",
                "approach_calibration",
                "file_name",
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
        selected_df[["approach", "calibration"]],
        on=["approach", "calibration"],
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
            if metric_name == "Temperature" or split != "test" or severity == 0:
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
                if metric_name == "Temperature" or split != "test":
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


def format_temperature_from_agg(rows: pd.DataFrame) -> str:
    if rows.empty or "Temperature_mean" not in rows.columns:
        return ""
    temp_rows = rows.dropna(subset=["Temperature_mean"])
    if temp_rows.empty:
        return ""
    if "split" in temp_rows.columns:
        temp_rows = temp_rows[temp_rows["split"] == "test"]
        if temp_rows.empty:
            temp_rows = rows.dropna(subset=["Temperature_mean"])
    if "severity" in temp_rows.columns:
        temp_rows = temp_rows.sort_values("severity", key=lambda col: col.astype(str))
    item = temp_rows.iloc[0]
    return format_cell(
        item["Temperature_mean"],
        item["Temperature_std"],
        int(item["Temperature_count"]),
        "0.2f",
    )


def format_temperature_from_summary(rows: pd.DataFrame) -> str:
    temp_rows = rows[(rows["metric"] == "Temperature") & (rows["split"] == "test")]
    if temp_rows.empty:
        temp_rows = rows[rows["metric"] == "Temperature"]
    if temp_rows.empty:
        return ""
    temp_rows = temp_rows.sort_values("severity", key=lambda col: col.astype(str))
    return format_summary_metric_from_row(temp_rows.iloc[0], "0.2f")


def approach_with_temperature(approach: str, temperature: str) -> str:
    if "T=optimal_T" in approach and temperature:
        return approach.replace("T=optimal_T", f"T={temperature}")
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


def format_overall_from_agg(
    rows: pd.DataFrame,
    metric_name: str,
    fmt: str,
    severities: list[int],
) -> str:
    test_rows = rows[
        (rows["split"] == "test")
        & (rows["severity"].isin(severities))
        & (rows[f"{metric_name}_mean"].notna())
    ]
    if test_rows.empty:
        return ""

    values = test_rows[f"{metric_name}_mean"].dropna()
    return format_cell(
        values.mean(),
        values.std(ddof=1) if len(values) > 1 else np.nan,
        int(len(values)),
        fmt,
    )


def metric_table_for_corruption(
    agg_df: pd.DataFrame,
    metric_name: str,
    fmt: str,
    approaches: list[str],
    severities: list[int],
    splits: list[str],
    corruption: str,
) -> pd.DataFrame:
    rows = []
    subset = agg_df[agg_df["corruption"] == corruption]
    for approach in approaches:
        approach_subset = subset[subset["approach_calibration"] == approach]
        row = {
            "Approach": approach_with_temperature(
                approach, format_temperature_from_agg(approach_subset)
            ),
            "Train": format_split_value(approach_subset, metric_name, fmt, "train"),
            "Validation": format_split_value(approach_subset, metric_name, fmt, "val"),
        }
        for severity in severities:
            cell = approach_subset[
                (approach_subset["split"] == "test")
                & (approach_subset["severity"] == severity)
            ]
            if cell.empty:
                row[f"Severity {severity}"] = ""
                continue
            row[f"Severity {severity}"] = format_metric_from_row(cell.iloc[0], metric_name, fmt)
        corrupted_severities = [severity for severity in severities if severity != 0]
        row["Overall Corrupted"] = format_overall_from_agg(
            approach_subset, metric_name, fmt, corrupted_severities
        )
        row["Overall"] = format_overall_from_agg(
            approach_subset, metric_name, fmt, severities
        )
        rows.append(row)
    return pd.DataFrame(rows)


def metric_table_for_summary(
    summary_df: pd.DataFrame,
    metric_name: str,
    fmt: str,
    approaches: list[str],
    severities: list[int],
    splits: list[str],
) -> pd.DataFrame:
    rows = []
    metric_summary = summary_df[summary_df["metric"] == metric_name]
    columns = [*severities, "Overall Corrupted", "Overall"]
    for approach in approaches:
        approach_all_summary = summary_df[summary_df["approach_calibration"] == approach]
        approach_summary = metric_summary[metric_summary["approach_calibration"] == approach]
        row = {
            "Approach": approach_with_temperature(
                approach, format_temperature_from_summary(approach_all_summary)
            ),
            "Train": format_summary_split_value(
                approach_all_summary, metric_name, fmt, "train"
            ),
            "Validation": format_summary_split_value(
                approach_all_summary, metric_name, fmt, "val"
            ),
        }
        for severity in columns:
            cell = approach_summary[
                (approach_summary["split"] == "test")
                & (approach_summary["severity"] == severity)
            ]
            column_name = f"Severity {severity}" if isinstance(severity, int) else severity
            if cell.empty:
                row[column_name] = ""
                continue
            item = cell.iloc[0]
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

        for col_idx, column in enumerate(table.columns):
            worksheet.write(row, col_idx, column, header_format)
        row += 1

        for _, values in table.iterrows():
            for col_idx, value in enumerate(values):
                worksheet.write(row, col_idx, value, cell_format)
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
) -> None:
    approach_order = selected_df["approach_calibration"].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_note = (
            "Values are mean +/- std across corruption-level means. "
            "smECE sigma is the adaptive bandwidth selected for smECE; "
            "smECE 0.05 uses the fixed bandwidth. "
            "Train and validation are shown in separate columns; severity columns show test only. "
            "Rows are selected by lowest validation Logloss within each loss family and calibration state. "
            "Calibrated rows use CE-selected temperature scaling."
        )
        corruption_note = (
            "Values are mean +/- std across seeds. "
            "smECE sigma is the adaptive bandwidth selected for smECE; "
            "smECE 0.05 uses the fixed bandwidth. "
            "Train and validation are shown in separate columns; severity columns show test only. "
            "Rows are selected by lowest validation Logloss within each loss family and calibration state. "
            "Calibrated rows use CE-selected temperature scaling."
        )
        summary_tables = [
            (
                metric_name,
                metric_table_for_summary(
                    summary_df, metric_name, fmt, approach_order, severities, args.splits
                ),
            )
            for metric_name, _, _, fmt in METRICS
        ]
        write_metric_blocks(
            writer,
            "Summary",
            summary_tables,
            "Average across CIFAR-10-C corruptions",
            summary_note,
        )

        for corruption in corruptions:
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

        agg_df.to_excel(writer, sheet_name="Aggregated Numeric", index=False)
        raw_df.to_excel(writer, sheet_name="Raw Results", index=False)
        selected_df.to_excel(writer, sheet_name="Selected Rows", index=False)

        if warnings:
            pd.DataFrame({"warning": warnings}).to_excel(
                writer, sheet_name="Warnings", index=False
            )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    clean_results_dir = args.clean_results_dir.resolve()
    output_path = args.output.resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if 0 in args.severities and not clean_results_dir.exists():
        raise FileNotFoundError(f"Clean results directory does not exist: {clean_results_dir}")

    severities = sorted(args.severities)
    seeds = sorted(args.seeds)
    corruptions = discover_corruptions(results_dir, severities)
    approaches = discover_approaches(results_dir)

    if not corruptions:
        raise RuntimeError(f"No CIFAR-10-C corruption directories found in {results_dir}")
    if not approaches:
        raise RuntimeError(f"No known CIFAR-10-C result JSON files found in {results_dir}")

    raw_df, warnings = collect_records(
        results_dir=results_dir,
        clean_results_dir=clean_results_dir,
        approaches=approaches,
        corruptions=corruptions,
        severities=severities,
        seeds=seeds,
        splits=args.splits,
        calibrations=args.calibrations,
        cal_criteria=args.cal_criteria,
    )
    if raw_df.empty:
        raise RuntimeError("No readable result records were found.")

    raw_df, selected_df = select_best_validation_logloss(raw_df)
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
    )

    print(f"Wrote {output_path}")
    print(f"Approaches: {', '.join(approach.approach for approach in approaches)}")
    print(f"Selected rows: {len(selected_df)}")
    print(f"Corruptions: {len(corruptions)}")
    print(f"Rows read: {len(raw_df)}")
    if warnings:
        print(f"Warnings: {len(warnings)} (also written to the Warnings sheet)")


if __name__ == "__main__":
    main()
