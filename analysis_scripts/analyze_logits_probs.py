#%% Load saved logits/probabilities and make simplex/link-shape plots.
#
# This script is intentionally usable in two ways:
#   1. import functions from it in a notebook; or
#   2. run it directly after editing DATASET_PATH / SEED_DIR below.

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.tri as mtri
import numpy as np
from scipy.stats import gaussian_kde
import torch
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_links import multi_link
from temperature_scaling import multi_focal_link


SPLITS = ("train", "val", "test")
PROB_FILE_RE = re.compile(r"^(?P<split>train|val|test)_probs_(?P<method>.+)\.npz$")


@dataclass
class LogitRecord:
    split: str
    path: Path
    logits: np.ndarray
    labels: np.ndarray
    indices: np.ndarray


@dataclass
class ProbabilityRecord:
    split: str
    method: str
    path: Path
    probs: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict


def scalar_or_array(value):
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value


def softmax(x, axis=1):
    x = np.asarray(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def brier_score(probs, labels):
    labels = labels.astype(int)
    y_true = np.zeros_like(probs)
    y_true[np.arange(len(labels)), labels] = 1
    return np.mean(np.sum((probs - y_true) ** 2, axis=1))


def compute_prob_metrics(probs, labels):
    labels = labels.astype(int)
    preds = np.argmax(probs, axis=1)
    n_classes = probs.shape[1]
    return {
        "acc": accuracy_score(labels, preds),
        "nll": log_loss(labels, probs, labels=list(range(n_classes))),
        "brier": brier_score(probs, labels),
    }


def load_logits_file(path, split):
    data = np.load(path)
    labels = data["labels"].reshape(-1)
    indices = data["indices"] if "indices" in data.files else np.arange(len(labels))
    return LogitRecord(
        split=split,
        path=Path(path),
        logits=data["logits"],
        labels=labels,
        indices=indices,
    )


def load_probability_file(path):
    path = Path(path)
    match = PROB_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a saved probability file: {path}")

    data = np.load(path, allow_pickle=True)
    labels = data["labels"].reshape(-1)
    indices = data["indices"] if "indices" in data.files else np.arange(len(labels))
    metadata = {
        key: scalar_or_array(data[key])
        for key in data.files
        if key not in {"probs", "labels", "indices"}
    }
    return ProbabilityRecord(
        split=match.group("split"),
        method=match.group("method"),
        path=path,
        probs=data["probs"],
        labels=labels,
        indices=indices,
        metadata=metadata,
    )


def load_seed_outputs(seed_dir, splits=SPLITS):
    seed_dir = Path(seed_dir)
    logits = {}
    probabilities = {split: {} for split in splits}

    for split in splits:
        logits_path = seed_dir / f"{split}_logits_labels_indices.npz"
        if logits_path.exists():
            logits[split] = load_logits_file(logits_path, split)

    for path in sorted(seed_dir.glob("*_probs_*.npz")):
        record = load_probability_file(path)
        if record.split in probabilities:
            probabilities[record.split][record.method] = record

    return {
        "seed_dir": seed_dir,
        "logits": logits,
        "probabilities": probabilities,
    }


def load_many_seeds(base_path_root, seed_dirs, splits=SPLITS):
    base_path_root = Path(base_path_root)
    return {
        str(seed): load_seed_outputs(base_path_root / str(seed), splits=splits)
        for seed in seed_dirs
    }


def list_probability_methods(seed_outputs, split=None):
    methods = set()
    for split_name, split_records in seed_outputs["probabilities"].items():
        if split is not None and split_name != split:
            continue
        methods.update(split_records.keys())
    return sorted(methods)


def print_seed_summary(seed_outputs):
    print(f"Seed directory: {seed_outputs['seed_dir']}")
    for split in SPLITS:
        logit_record = seed_outputs["logits"].get(split)
        if logit_record is not None:
            logits = logit_record.logits
            labels = logit_record.labels
            probs = softmax(logits)
            metrics = compute_prob_metrics(probs, labels)
            print(
                f"{split:>5} logits: {logits.shape}, "
                f"softmax acc={metrics['acc']:.4f}, nll={metrics['nll']:.4f}"
            )
        for method, record in seed_outputs["probabilities"][split].items():
            metrics = compute_prob_metrics(record.probs, record.labels)
            print(
                f"{split:>5} probs {method}: {record.probs.shape}, "
                f"acc={metrics['acc']:.4f}, nll={metrics['nll']:.4f}"
            )


def has_any_logits(seed_outputs):
    return any(split in seed_outputs["logits"] for split in SPLITS)


def has_any_probabilities(seed_outputs):
    return any(seed_outputs["probabilities"][split] for split in SPLITS)


def save_figure(fig, output_path, dpi=200):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved figure: {output_path.resolve()}")
    return output_path


def topk_indices(values, k=3, offset=0):
    if offset < 0:
        raise ValueError("top-k offset must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if values.shape[1] < offset + k:
        raise ValueError(
            "Need at least {} classes for k={} and offset={}, but got {}.".format(
                offset + k, k, offset, values.shape[1]
            )
        )
    ranked = np.argsort(values, axis=1)[:, ::-1]
    return ranked[:, offset:offset + k]


def row_normalize(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    denom = np.sum(values, axis=1, keepdims=True)
    return values / np.maximum(denom, eps)


def topk_simplex_from_probs(probs, topk=None, topk_offset=0):
    if topk is None:
        topk = topk_indices(probs, k=3, offset=topk_offset)
    top3 = np.take_along_axis(probs, topk, axis=1)
    return row_normalize(top3)


def topk_simplex_from_logits(logits, topk=None, topk_offset=0):
    if topk is None:
        topk = topk_indices(logits, k=3, offset=topk_offset)
    top3_logits = np.take_along_axis(logits, topk, axis=1)
    return softmax(top3_logits, axis=1)


def topk_rank_label(topk_offset):
    first = topk_offset + 1
    last = topk_offset + 3
    return "ranks {}-{}".format(first, last)


def topk_output_suffix(topk_offset):
    return "top{}_to_top{}".format(topk_offset + 1, topk_offset + 3)


def filename_token(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")


def barycentric_to_xy(simplex_probs):
    p = row_normalize(simplex_probs)
    x = p[:, 1] + 0.5 * p[:, 2]
    y = (np.sqrt(3.0) / 2.0) * p[:, 2]
    return x, y


def draw_simplex_outline(ax):
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]
    ax.plot(triangle_x, triangle_y, color="black", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, np.sqrt(3.0) / 2.0 + 0.04)
    ax.set_xticks([])
    ax.set_yticks([])


def simplex_xy_grid(grid_size=220):
    h = np.sqrt(3.0) / 2.0
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, h, grid_size)
    xx, yy = np.meshgrid(x, y)
    mask = (yy >= 0.0) & (yy <= np.sqrt(3.0) * xx) & (yy <= np.sqrt(3.0) * (1.0 - xx))
    return xx, yy, mask


def simplex_kde_surface(x, y, grid_size=220, max_points=8000, seed=0, bw_adjust=1.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    xx, yy, mask = simplex_xy_grid(grid_size=grid_size)
    zz = np.full_like(xx, np.nan, dtype=float)
    if len(x) < 2:
        return xx, yy, zz

    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        keep = rng.choice(len(x), size=max_points, replace=False)
        x = x[keep]
        y = y[keep]

    xy = np.vstack([x, y])
    if np.linalg.matrix_rank(np.cov(xy)) < 2:
        xy = xy + rng.normal(0.0, 1e-4, size=xy.shape)

    try:
        kde = gaussian_kde(xy)
        if bw_adjust != 1.0:
            kde.set_bandwidth(kde.factor * bw_adjust)
        values = kde(np.vstack([xx[mask], yy[mask]]))
    except np.linalg.LinAlgError:
        xy = xy + rng.normal(0.0, 1e-3, size=xy.shape)
        kde = gaussian_kde(xy)
        if bw_adjust != 1.0:
            kde.set_bandwidth(kde.factor * bw_adjust)
        values = kde(np.vstack([xx[mask], yy[mask]]))

    if values.size and values.max() > 0:
        values = values / values.max()
    zz[mask] = values
    return xx, yy, zz


def contourf_simplex_surface(ax, xx, yy, zz, title="", cmap="viridis", levels=36, vmin=None, vmax=None):
    contour = ax.contourf(
        xx,
        yy,
        zz,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    draw_simplex_outline(ax)
    ax.set_title(title)
    return contour


def simplex_histogram(x, y, bins=70, normalize=True):
    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0.0, 1.0], [0.0, np.sqrt(3.0) / 2.0]],
        density=False,
    )
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.T, xedges, yedges


def simplex_histogram_summary(hist):
    total_bins = hist.size
    occupied_bins = int(np.count_nonzero(hist))
    max_bin_mass = float(hist.max()) if total_bins else 0.0
    occupied_fraction = occupied_bins / total_bins if total_bins else 0.0
    return {
        "occupied_bins": occupied_bins,
        "occupied_fraction": occupied_fraction,
        "max_bin_mass": max_bin_mass,
    }


def print_simplex_histogram_summary(name, hist):
    summary = simplex_histogram_summary(hist)
    print(
        "{}: occupied_bins={} ({:.2%}), max_bin_mass={:.2%}".format(
            name,
            summary["occupied_bins"],
            summary["occupied_fraction"],
            summary["max_bin_mass"],
        )
    )


def format_simplex_mean(simplex_probs):
    mean_probs = np.mean(simplex_probs, axis=0)
    return "mean=({:.4f}, {:.4f}, {:.4f})".format(
        mean_probs[0],
        mean_probs[1],
        mean_probs[2],
    )


def ordered_rank_simplex_to_expanded(simplex_probs):
    """Affine-map the sorted-rank sector p1 >= p2 >= p3 onto the full simplex."""
    p = row_normalize(simplex_probs)
    expanded = np.column_stack(
        [
            1.0 - 2.0 * p[:, 1] - p[:, 2],
            2.0 * (p[:, 1] - p[:, 2]),
            3.0 * p[:, 2],
        ]
    )
    return row_normalize(np.clip(expanded, 0.0, None))


def is_mostly_ordered_rank_simplex(simplex_probs, tolerance=1e-10, min_fraction=0.995):
    p = row_normalize(simplex_probs)
    ordered = (p[:, 0] + tolerance >= p[:, 1]) & (p[:, 1] + tolerance >= p[:, 2])
    return np.mean(ordered) >= min_fraction


def rank_simplex_display_probs(simplex_probs):
    if is_mostly_ordered_rank_simplex(simplex_probs):
        return ordered_rank_simplex_to_expanded(simplex_probs), "ordered sector expanded"
    return simplex_probs, None


def topk_simplex_on_reference(record, reference_record=None, topk_offset=0):
    if reference_record is not None and len(reference_record.probs) == len(record.probs):
        topk = topk_indices(reference_record.probs, k=3, offset=topk_offset)
    else:
        topk = None
    return topk_simplex_from_probs(record.probs, topk=topk, topk_offset=topk_offset)


def plot_simplex_mean_marker(ax, simplex_probs, display_simplex_probs=None):
    mean_probs = np.mean(simplex_probs, axis=0, keepdims=True)
    if display_simplex_probs is None:
        marker_probs = mean_probs
    else:
        marker_probs = np.mean(display_simplex_probs, axis=0, keepdims=True)
    mean_x, mean_y = barycentric_to_xy(marker_probs)
    ax.scatter(
        mean_x,
        mean_y,
        s=95,
        marker="*",
        c="#f04e23",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
        label="dataset mean",
    )
    ax.text(
        0.02,
        0.98,
        format_simplex_mean(simplex_probs),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        zorder=7,
    )
    return mean_probs.squeeze(0)


def pcolormesh_positive_log(ax, xedges, yedges, hist, cmap="viridis"):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    masked_hist = np.ma.masked_less_equal(hist, 0.0)
    positive = hist[hist > 0.0]
    norm = None
    if positive.size > 0 and positive.max() > positive.min():
        norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    return ax.pcolormesh(
        xedges,
        yedges,
        masked_hist,
        shading="auto",
        cmap=cmap_obj,
        norm=norm,
    )


def plot_simplex_density_from_simplex_probs(
    ax,
    simplex_probs,
    title="",
    bins=35,
    cmap="viridis",
    scatter_overlay=True,
    display_simplex_probs=None,
    coordinate_note=None,
):
    density_simplex_probs = simplex_probs if display_simplex_probs is None else display_simplex_probs
    x, y = barycentric_to_xy(density_simplex_probs)
    hist, xedges, yedges = simplex_histogram(x, y, bins=bins)
    xx, yy, zz = simplex_kde_surface(x, y)
    mesh = contourf_simplex_surface(ax, xx, yy, zz, cmap=cmap)
    summary = simplex_histogram_summary(hist)
    if scatter_overlay:
        plot_x = x
        plot_y = y
        if summary["max_bin_mass"] > 0.5:
            rng = np.random.default_rng(0)
            x_width = (xedges[-1] - xedges[0]) / bins
            y_width = (yedges[-1] - yedges[0]) / bins
            plot_x = x + rng.uniform(-0.35 * x_width, 0.35 * x_width, size=len(x))
            plot_y = y + rng.uniform(-0.35 * y_width, 0.35 * y_width, size=len(y))
        ax.scatter(
            plot_x,
            plot_y,
            s=2.0,
            c="black",
            alpha=0.08,
            linewidths=0,
            rasterized=True,
        )
    mean_probs = plot_simplex_mean_marker(ax, simplex_probs, density_simplex_probs)
    summary_text = "max bin {:.1%}, occupied {:.1%}".format(
        summary["max_bin_mass"],
        summary["occupied_fraction"],
    )
    if coordinate_note:
        summary_text = "{}\n{}".format(summary_text, coordinate_note)
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        zorder=7,
    )
    ax.set_title(title)
    return mesh


def plot_simplex_density(ax, probs, title="", topk=None, topk_offset=0, bins=35, cmap="viridis"):
    simplex_probs = topk_simplex_from_probs(probs, topk=topk, topk_offset=topk_offset)
    display_simplex_probs, coordinate_note = rank_simplex_display_probs(simplex_probs)
    return plot_simplex_density_from_simplex_probs(
        ax,
        simplex_probs,
        title=title,
        bins=bins,
        cmap=cmap,
        display_simplex_probs=display_simplex_probs,
        coordinate_note=coordinate_note,
    )


def plot_logits_simplex_density_grid(seed_outputs, splits=SPLITS, topk_offset=0, bins=35, output_path=None):
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(4.0 * len(splits), 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, split in zip(axes[0], splits):
        record = seed_outputs["logits"].get(split)
        if record is None:
            ax.axis("off")
            ax.set_title(f"{split}: missing logits")
            continue
        simplex_probs = topk_simplex_from_logits(record.logits, topk_offset=topk_offset)
        display_simplex_probs, coordinate_note = rank_simplex_display_probs(simplex_probs)
        mesh = plot_simplex_density_from_simplex_probs(
            ax,
            simplex_probs,
            title=f"logits {topk_rank_label(topk_offset)} softmax\n{split}",
            bins=bins,
            display_simplex_probs=display_simplex_probs,
            coordinate_note=coordinate_note,
        )
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label("relative density")
        x, y = barycentric_to_xy(display_simplex_probs)
        hist, _, _ = simplex_histogram(x, y, bins=bins)
        print_simplex_histogram_summary(f"logits {topk_rank_label(topk_offset)} {split}", hist)
        print(f"logits {topk_rank_label(topk_offset)} {split}: {format_simplex_mean(simplex_probs)}")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


def plot_simplex_density_grid(
    seed_outputs,
    methods=None,
    splits=SPLITS,
    reference_method="softmax",
    topk_offset=0,
    bins=35,
    output_path=None,
):
    if methods is None:
        methods = list_probability_methods(seed_outputs)
    methods = [method for method in methods if any(method in seed_outputs["probabilities"][split] for split in splits)]

    ncols = len(splits) + 2
    fig, axes = plt.subplots(
        len(methods),
        ncols,
        figsize=(4.0 * ncols, 3.6 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, method in enumerate(methods):
        for col, split in enumerate(splits):
            ax = axes[row][col]
            record = seed_outputs["probabilities"].get(split, {}).get(method)
            if record is None:
                ax.axis("off")
                ax.set_title(f"{split}: missing")
                continue

            reference = seed_outputs["probabilities"].get(split, {}).get(reference_method)
            if reference is not None and len(reference.probs) == len(record.probs):
                topk = topk_indices(reference.probs, k=3, offset=topk_offset)
            else:
                topk = None

            simplex_probs = topk_simplex_from_probs(record.probs, topk=topk, topk_offset=topk_offset)
            display_simplex_probs, coordinate_note = rank_simplex_display_probs(simplex_probs)
            mesh = plot_simplex_density_from_simplex_probs(
                ax,
                simplex_probs,
                title=f"{method}\n{split}, {topk_rank_label(topk_offset)}",
                bins=bins,
                display_simplex_probs=display_simplex_probs,
                coordinate_note=coordinate_note,
            )
            colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
            colorbar.set_label("relative density")
            x, y = barycentric_to_xy(display_simplex_probs)
            hist, _, _ = simplex_histogram(x, y, bins=bins)
            print_simplex_histogram_summary(f"{method} {topk_rank_label(topk_offset)} {split}", hist)
            print(f"{method} {topk_rank_label(topk_offset)} {split}: {format_simplex_mean(simplex_probs)}")

        method_record = first_probability_record_for_method(seed_outputs, method, splits=splits)
        expanded_ax = axes[row][len(splits)]
        plot_calibration_displacement_panel(
            expanded_ax,
            method_record,
            method,
            expand_ordered_sector=True,
        )
        raw_ax = axes[row][len(splits) + 1]
        plot_calibration_displacement_panel(
            raw_ax,
            method_record,
            method,
            expand_ordered_sector=False,
        )

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


def density_on_reference_grid(record, reference_record=None, topk_offset=0, bins=35):
    simplex_probs = topk_simplex_on_reference(record, reference_record, topk_offset)
    display_simplex_probs, _ = rank_simplex_display_probs(simplex_probs)
    x, y = barycentric_to_xy(display_simplex_probs)
    return simplex_histogram(x, y, bins=bins)


def kde_surface_on_reference_grid(record, reference_record=None, topk_offset=0):
    simplex_probs = topk_simplex_on_reference(record, reference_record, topk_offset)
    display_simplex_probs, _ = rank_simplex_display_probs(simplex_probs)
    x, y = barycentric_to_xy(display_simplex_probs)
    return simplex_kde_surface(x, y)


def kde_surface_pair_on_reference_grid(
    record_a,
    record_b,
    reference_record_a=None,
    reference_record_b=None,
    topk_offset=0,
):
    simplex_a = topk_simplex_on_reference(record_a, reference_record_a, topk_offset)
    simplex_b = topk_simplex_on_reference(record_b, reference_record_b, topk_offset)
    if is_mostly_ordered_rank_simplex(simplex_a) and is_mostly_ordered_rank_simplex(simplex_b):
        simplex_a = ordered_rank_simplex_to_expanded(simplex_a)
        simplex_b = ordered_rank_simplex_to_expanded(simplex_b)
        coordinate_note = "ordered sector expanded"
    else:
        coordinate_note = None
    x_a, y_a = barycentric_to_xy(simplex_a)
    x_b, y_b = barycentric_to_xy(simplex_b)
    xx, yy, surface_a = simplex_kde_surface(x_a, y_a)
    _, _, surface_b = simplex_kde_surface(x_b, y_b)
    return xx, yy, surface_a, surface_b, coordinate_note


def plot_split_density_drift(
    seed_outputs,
    method,
    split_pairs=(("train", "val"), ("val", "test")),
    reference_method="softmax",
    topk_offset=0,
    bins=35,
    output_path=None,
):
    panels = []
    for split_a, split_b in split_pairs:
        record_a = seed_outputs["probabilities"].get(split_a, {}).get(method)
        record_b = seed_outputs["probabilities"].get(split_b, {}).get(method)
        if record_a is None or record_b is None:
            panels.append((split_a, split_b, None, None, None, None))
            continue

        ref_a = seed_outputs["probabilities"].get(split_a, {}).get(reference_method)
        ref_b = seed_outputs["probabilities"].get(split_b, {}).get(reference_method)
        xx, yy, surface_a, surface_b, coordinate_note = kde_surface_pair_on_reference_grid(
            record_a,
            record_b,
            ref_a,
            ref_b,
            topk_offset=topk_offset,
        )
        panels.append((split_a, split_b, surface_b - surface_a, xx, yy, coordinate_note))

    finite_panels = [panel[2] for panel in panels if panel[2] is not None]
    vmax = max(np.nanmax(np.abs(panel)) for panel in finite_panels) if finite_panels else 1.0

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.2 * len(panels), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (split_a, split_b, diff, xx, yy, coordinate_note) in zip(axes[0], panels):
        if diff is None:
            ax.axis("off")
            ax.set_title(f"{split_a} -> {split_b}: missing")
            continue
        mesh = ax.contourf(
            xx,
            yy,
            diff,
            levels=36,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        draw_simplex_outline(ax)
        ax.set_title(f"{method}\n{split_b} - {split_a}, {topk_rank_label(topk_offset)}")
        if coordinate_note:
            ax.text(
                0.02,
                0.02,
                coordinate_note,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
                zorder=7,
            )
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label("relative density difference")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


def infer_link_and_value(record):
    metadata = record.metadata
    link = metadata.get("link")
    link_value = metadata.get("link_value", 1.0)
    temperature = metadata.get("temperature", 1.0)
    if isinstance(link_value, np.ndarray):
        link_value = tuple(float(v) for v in link_value.ravel())
    else:
        link_value = float(link_value)
    return link, link_value, float(temperature)


def format_parameter_value(value):
    if isinstance(value, tuple):
        return "({})".format(", ".join(format_parameter_value(v) for v in value))
    return "{:.4g}".format(float(value))


def calibration_parameter_title(method, link, link_value, temperature, coordinate_title=None):
    if link in (None, "softmax"):
        link_text = "link=softmax"
    else:
        link_text = "link={}, a={}".format(link, format_parameter_value(link_value))
    title = "{}\nmap displacement\n{}, T={}".format(
        method,
        link_text,
        format_parameter_value(temperature),
    )
    if coordinate_title:
        title = "{}\n{}".format(title, coordinate_title)
    return title


def ordered_rank_sector_grid(grid_size=19):
    q = row_normalize(simplex_grid(grid_size))
    ordered = (q[:, 0] >= q[:, 1]) & (q[:, 1] >= q[:, 2])
    return q[ordered]


def calibration_displacement_coordinates(
    link,
    link_value,
    temperature,
    grid_size=19,
    expand_ordered_sector=True,
):
    q = ordered_rank_sector_grid(grid_size) if expand_ordered_sector else row_normalize(simplex_grid(grid_size))
    logits = np.log(q)
    p = link_probs_from_logits(
        logits,
        link=link,
        link_value=link_value,
        temperature=temperature,
    )

    ordered_output = (p[:, 0] >= p[:, 1] - 1e-10) & (p[:, 1] >= p[:, 2] - 1e-10)
    if expand_ordered_sector and np.any(ordered_output):
        q_display = ordered_rank_simplex_to_expanded(q[ordered_output])
        p_display = ordered_rank_simplex_to_expanded(p[ordered_output])
        kept_fraction = np.mean(ordered_output)
        coordinate_note = "ordered sector expanded"
        if kept_fraction < 0.995:
            coordinate_note = "{}\n{:.0%} stays ordered".format(coordinate_note, kept_fraction)
    else:
        q_display = q
        p_display = p
        coordinate_note = "raw simplex reference"

    x0, y0 = barycentric_to_xy(q_display)
    x1, y1 = barycentric_to_xy(p_display)
    dx = x1 - x0
    dy = y1 - y0
    mag = np.sqrt(dx ** 2 + dy ** 2)
    return x0, y0, dx, dy, mag, coordinate_note


def link_probs_from_logits(logits, link="softmax", link_value=1.0, temperature=1.0):
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    with torch.no_grad():
        if link == "softmax" or link is None:
            probs = torch.softmax(logits_t / temperature, dim=1)
        elif link == "focal":
            probs = multi_focal_link(logits_t / temperature, link_value)
        elif link == "generalized_focal":
            probs = multi_link(logits_t / temperature, link, link_value[0], link_value[1])
        else:
            probs = multi_link(logits_t / temperature, link, link_value)
    return probs.cpu().numpy()


def plot_calibration_displacement_panel(
    ax,
    record,
    method,
    grid_size=19,
    expand_ordered_sector=True,
):
    if record is None:
        ax.axis("off")
        ax.set_title(f"{method}\nmap displacement\nmissing")
        return None

    metadata_method = record.metadata.get("method", method)
    if metadata_method == "softmax+dirichlet" or record.metadata.get("dirichlet", False):
        draw_simplex_outline(ax)
        ax.text(
            0.5,
            0.5,
            "Dirichlet map\nparameters not saved",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3.0},
            zorder=7,
        )
        ax.set_title(f"{method}\nmap displacement")
        return None

    link, link_value, temperature = infer_link_and_value(record)
    x0, y0, dx, dy, mag, coordinate_note = calibration_displacement_coordinates(
        link=link,
        link_value=link_value,
        temperature=temperature,
        grid_size=grid_size,
        expand_ordered_sector=expand_ordered_sector,
    )
    coordinate_title = "expanded coords" if expand_ordered_sector else "raw coords"

    if np.nanmax(mag) <= 1e-7:
        ax.scatter(x0, y0, s=7, c="black", alpha=0.65, linewidths=0, rasterized=True)
        draw_simplex_outline(ax)
        identity_text = "identity"
        if coordinate_note:
            identity_text = "{}\n{}".format(identity_text, coordinate_note)
        ax.text(
            0.02,
            0.02,
            identity_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            zorder=7,
        )
        ax.set_title(calibration_parameter_title(method, link, link_value, temperature, coordinate_title))
        return None

    quiver = ax.quiver(
        x0,
        y0,
        dx,
        dy,
        mag,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        cmap="viridis",
    )
    draw_simplex_outline(ax)
    if coordinate_note:
        ax.text(
            0.02,
            0.02,
            coordinate_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            zorder=7,
        )
    ax.set_title(calibration_parameter_title(method, link, link_value, temperature, coordinate_title))
    ax.figure.colorbar(quiver, ax=ax, fraction=0.046, pad=0.02, label="simplex displacement")
    return quiver


def first_probability_record_for_method(seed_outputs, method, splits=SPLITS):
    for split in splits:
        record = seed_outputs["probabilities"].get(split, {}).get(method)
        if record is not None:
            return record
    return None


def plot_calibration_displacement_figure(
    record,
    method,
    output_path=None,
    grid_size=23,
    expand_ordered_sector=False,
):
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    plot_calibration_displacement_panel(
        ax,
        record,
        method,
        grid_size=grid_size,
        expand_ordered_sector=expand_ordered_sector,
    )
    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


def plot_calibration_displacement_figures(seed_outputs, methods, splits=SPLITS, output_dir=None):
    figures = {}
    for method in methods:
        record = first_probability_record_for_method(seed_outputs, method, splits=splits)
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir) / f"calibration_map_displacement_{filename_token(method)}.png"
        figures[method] = plot_calibration_displacement_figure(
            record,
            method,
            output_path=output_path,
        )
    return figures


def plot_link_slice(
    link="exp_1mp",
    link_value=1.5,
    temperature=1.0,
    z0=0.0,
    z1_range=(-2.0, 4.0),
    z2_range=(-2.0, 4.0),
    grid_size=160,
    class_index=0,
    output_path=None,
):
    z1 = np.linspace(z1_range[0], z1_range[1], grid_size)
    z2 = np.linspace(z2_range[0], z2_range[1], grid_size)
    zz1, zz2 = np.meshgrid(z1, z2)
    logits = np.stack(
        [
            np.full_like(zz1, z0),
            zz1,
            zz2,
        ],
        axis=-1,
    ).reshape(-1, 3)

    q = softmax(logits / temperature)
    p = link_probs_from_logits(logits, link=link, link_value=link_value, temperature=temperature)
    q_class = q[:, class_index].reshape(grid_size, grid_size)
    p_class = p[:, class_index].reshape(grid_size, grid_size)
    diff = p_class - q_class

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    panels = [
        (q_class, f"q_{class_index} (softmax)", "viridis", None),
        (p_class, f"phi(q)_{class_index} ({link}, a={link_value})", "viridis", None),
        (diff, f"phi(q)_{class_index} - q_{class_index}", "coolwarm", max(abs(diff.min()), abs(diff.max()))),
    ]
    for ax, (values, title, cmap, vmax) in zip(axes, panels):
        kwargs = {"levels": 18, "cmap": cmap}
        if vmax is not None:
            kwargs.update({"vmin": -vmax, "vmax": vmax})
        contour = ax.contourf(zz1, zz2, values, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        fig.colorbar(contour, ax=ax)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


def plot_link_barycentric_fields(
    link="exp_1mp",
    link_value=1.5,
    temperature=1.0,
    grid_size=120,
    class_index=0,
    output_path=None,
):
    q = simplex_grid(grid_size, eps=1e-5)
    logits = np.log(q)
    p = link_probs_from_logits(
        logits,
        link=link,
        link_value=link_value,
        temperature=temperature,
    )
    l1_distance = np.sum(np.abs(p - q), axis=1)
    x, y = barycentric_to_xy(q)
    triangulation = mtri.Triangulation(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    fig.suptitle(
        "3-class simplex: {} link (a={}) on barycentric q".format(link, link_value),
        fontsize=14,
    )
    panels = [
        (q[:, class_index], f"q_{class_index} (input barycentric)", "viridis"),
        (p[:, class_index], f"phi(q)_{class_index} after {link}", "viridis"),
        (l1_distance, r"L1 dist $||\phi(q)-q||_1$", "viridis"),
    ]
    for ax, (values, title, cmap) in zip(axes, panels):
        contour = ax.tricontourf(triangulation, values, levels=48, cmap=cmap)
        draw_simplex_outline(ax)
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


def simplex_grid(n=21, eps=1e-4):
    points = []
    for i in range(n):
        p1 = i / (n - 1)
        for j in range(n - i):
            p2 = j / (n - 1)
            p3 = 1.0 - p1 - p2
            if p3 >= -1e-12:
                points.append([p1, p2, p3])
    points = np.asarray(points, dtype=float)
    return np.clip(points, eps, 1.0 - eps)


def plot_link_displacement_simplex(
    link="exp_1mp",
    link_value=1.5,
    grid_size=23,
    output_path=None,
):
    q = simplex_grid(grid_size)
    logits = np.log(q)
    p = link_probs_from_logits(logits, link=link, link_value=link_value, temperature=1.0)

    x0, y0 = barycentric_to_xy(q)
    x1, y1 = barycentric_to_xy(p)
    dx = x1 - x0
    dy = y1 - y0
    mag = np.sqrt(dx ** 2 + dy ** 2)

    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    quiver = ax.quiver(x0, y0, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0, cmap="viridis")
    draw_simplex_outline(ax)
    ax.set_title(f"{link} displacement (a={link_value})")
    fig.colorbar(quiver, ax=ax, fraction=0.046, pad=0.02, label="simplex displacement")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


def suggested_next_plots(seed_outputs):
    return [
        "Reliability-on-simplex: color each bin by average correctness or calibration gap.",
        "Entropy drift: add contours of mean predictive entropy over the top-3 simplex.",
        "Margin drift: plot top1-top2 probability margin density for train/val/test.",
        "Correct-vs-incorrect simplex: overlay correct and incorrect densities to see where errors live.",
        "Link displacement vectors plus data density: put the method's vector field behind observed points.",
    ]


def parse_args():
    default_input_dir = PROJECT_ROOT / "RESULTS" / "CIFAR10_LOGITSLABELS" / "42"
    default_output_dir = PROJECT_ROOT / "analysis_scripts" / "logit_prob_visualizations"

    parser = argparse.ArgumentParser(
        description="Load saved logits/probabilities and write simplex/link visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing train/val/test logits and probability .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Probability methods to plot. Omit to auto-pick available methods.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=35,
        help="Number of bins per simplex axis. Use fewer bins when distributions look too concentrated.",
    )
    parser.add_argument(
        "--topk-offset",
        type=int,
        default=0,
        help=(
            "Offset into the sorted class ranks for the 3-coordinate simplex. "
            "0 plots ranks 1-3, 1 plots ranks 2-4, 2 plots ranks 3-5, etc."
        ),
    )
    return parser.parse_args()


def choose_link_shape_record(seed_outputs, methods_to_plot):
    for method in methods_to_plot:
        for split in SPLITS:
            record = seed_outputs["probabilities"][split].get(method)
            if record is None:
                continue
            link, _, _ = infer_link_and_value(record)
            if link not in (None, "softmax"):
                return record
    return None


def choose_default_methods(seed_outputs, max_methods=6):
    available = list_probability_methods(seed_outputs)
    selected = []

    def add_exact(method):
        if method in available and method not in selected:
            selected.append(method)

    def add_first_with_prefix(prefix):
        for method in available:
            if method.startswith(prefix) and method not in selected:
                selected.append(method)
                return

    add_exact("softmax")
    add_first_with_prefix("softmax+ts")
    add_first_with_prefix("softmax+exp1mp_")
    add_first_with_prefix("softmax+exp1mp+ts")
    add_first_with_prefix("softmax+exponp_")
    add_first_with_prefix("softmax+exponp+ts")
    add_exact("softmax+dirichlet")

    for method in available:
        if len(selected) >= max_methods:
            break
        if method not in selected:
            selected.append(method)

    return selected


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_outputs = load_seed_outputs(input_dir)
    topk_suffix = topk_output_suffix(args.topk_offset)
    print(f"Loading saved arrays from: {input_dir.resolve()}")
    print(f"Saving figures to: {output_dir.resolve()}")
    print(f"Simplex rank window: {topk_rank_label(args.topk_offset)}")
    print_seed_summary(seed_outputs)
    print("\nAvailable probability methods:")
    for method in list_probability_methods(seed_outputs):
        print(" ", method)

    if has_any_logits(seed_outputs):
        plot_logits_simplex_density_grid(
            seed_outputs,
            splits=SPLITS,
            topk_offset=args.topk_offset,
            bins=args.bins,
            output_path=output_dir / f"logits_simplex_density_grid_{topk_suffix}.png",
        )
    else:
        print("No logits files found, so logits simplex figure was not written.")

    if args.methods is not None:
        methods_to_plot = args.methods
    else:
        methods_to_plot = choose_default_methods(seed_outputs)

    if methods_to_plot:
        plot_simplex_density_grid(
            seed_outputs,
            methods=methods_to_plot,
            splits=SPLITS,
            topk_offset=args.topk_offset,
            bins=args.bins,
            output_path=output_dir / f"simplex_density_grid_{topk_suffix}.png",
        )
        plot_calibration_displacement_figures(
            seed_outputs,
            methods=methods_to_plot,
            splits=SPLITS,
            output_dir=output_dir,
        )
        plot_split_density_drift(
            seed_outputs,
            method=methods_to_plot[0],
            topk_offset=args.topk_offset,
            bins=args.bins,
            output_path=output_dir / f"simplex_density_drift_{topk_suffix}.png",
        )

        link_shape_record = choose_link_shape_record(seed_outputs, methods_to_plot)
        if link_shape_record is not None:
            link, link_value, temperature = infer_link_and_value(link_shape_record)
            plot_link_barycentric_fields(
                link=link,
                link_value=link_value,
                temperature=temperature,
                output_path=output_dir / "link_barycentric_fields.png",
            )
            plot_link_slice(
                link=link,
                link_value=link_value,
                temperature=temperature,
                output_path=output_dir / "link_slice.png",
            )
            plot_link_displacement_simplex(
                link=link,
                link_value=link_value,
                output_path=output_dir / "link_displacement.png",
            )
        else:
            print("No non-softmax probability method found, so link-shape figures were not written.")
    else:
        print("No saved probability files found, so probability simplex/drift figures were not written.")

    print("\nOther plot ideas:")
    for idea in suggested_next_plots(seed_outputs):
        print(" -", idea)

    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
