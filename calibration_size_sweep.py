"""Repeated calibration fits on cached validation logits, with a fixed test set.

Rows are local to their source artifact. A model seed and a subset seed identify
different levels of replication; repetitions are not independent trained models.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PureWindowsPath
import tempfile
import time

import numpy as np
import sklearn
import torch
from torch import nn
from Calibs.sms_calibration import add_sms_arguments, load_sms_class, sms_provenance

from evaluate import (
    evaluation_json_path, get_active_link_dict, normalize_links,
    link_value_to_key, method_name_for_link, round_floats_for_json, set_seed,
)
from evaluate_focal_calibration import (
    dirichlet_calibration_evaluation, focal_calibration_evaluation, sms_calibration_evaluation,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run calibration-set-size sweeps from cached validation/test logits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--logits-path", required=True,
                        help="Directory containing val/test_logits_labels_indices.npz and optional logits_metadata.json.")
    parser.add_argument("--save-eval-path", dest="save_eval_loc",
                        help="Output directory, including model seed; defaults to --logits-path.")
    parser.add_argument("--dataset", help="Dataset name; defaults to source metadata.")
    parser.add_argument("--model", help="Model architecture; defaults to source metadata.")
    parser.add_argument("--model-name", help="Model prefix for output filenames; defaults to source metadata.")
    parser.add_argument("--saved_model_name", help="Checkpoint filename for output naming; no checkpoint is loaded.")
    parser.add_argument("--seed", type=int, help="Trained model seed; defaults to metadata seed or 1.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--calibration-sizes", nargs="+", default=["100", "250", "500", "1000", "2500", "full"],
                        help="Positive example counts or 'full'; full size is always included.")
    parser.add_argument("--subset-seeds", nargs="+", type=int, default=[1001, 1002, 1003],
                        help="Calibration draw seeds, separate from the trained model seed.")
    parser.add_argument("--links", nargs="+", help="Calibration links; omit to evaluate all links.")
    parser.add_argument("--posthoc-order", choices=["ts_first", "ts_last"], default="ts_first")
    parser.add_argument("--dirichlet", action="store_true", help="Include full ODIR Dirichlet calibration.")
    parser.add_argument("--dirichlet-cv-folds", type=int, default=3)
    parser.add_argument("--dirichlet-reg-grid", nargs="+", type=float)
    parser.add_argument("--dirichlet-max-iter", type=int, default=1024)
    parser.add_argument("--dirichlet-n-jobs", type=int, default=1)
    add_sms_arguments(parser)
    parser.add_argument("--json-precision", type=int, default=5,
                        help="Precision for nested metrics; selected metrics/summaries retain full precision.")
    return parser.parse_args(argv)


def load_metadata(logits_path):
    path = Path(logits_path) / "logits_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fill_args_from_metadata(args, metadata):
    args.save_eval_loc = args.save_eval_loc or args.logits_path
    args.dataset = args.dataset or metadata.get("dataset") or "logits"
    args.model = args.model or metadata.get("model") or metadata.get("model_name") or "model"
    args.model_name = args.model_name or metadata.get("model_name") or args.model
    args.saved_model_name = args.saved_model_name or metadata.get("saved_model_name") or "{}_posthoc.model".format(args.model_name)
    args.seed = int(args.seed if args.seed is not None else metadata.get("seed", 1))
    args.links = normalize_links(args.links)
    return args


def load_split_npz(logits_path, split_name, device):
    if split_name not in ("val", "test"):
        raise ValueError("Sweeps load only validation and test logits.")
    path = Path(logits_path) / (split_name + "_logits_labels_indices.npz")
    with np.load(path, allow_pickle=False) as data:
        if "logits" not in data or "labels" not in data:
            raise ValueError("{} must contain logits and labels arrays.".format(path))
        logits = torch.as_tensor(data["logits"], dtype=torch.float32, device=device)
        labels = torch.as_tensor(data["labels"], dtype=torch.long, device=device).view(-1)
    if logits.ndim != 2 or len(labels) != logits.shape[0]:
        raise ValueError("{} requires 2D logits and one label per row.".format(path))
    return logits, labels


def main(argv=None):
    args = parse_args(argv)
    metadata = load_metadata(args.logits_path)
    args = fill_args_from_metadata(args, metadata)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    return run_calibration_size_sweep(args, metadata, device)


def resolve_sizes(values, pool_size):
    sizes = {pool_size}
    for value in values:
        if str(value).lower() == "full":
            sizes.add(pool_size)
        else:
            size = int(value)
            if size <= 0:
                raise ValueError("Calibration sizes must be positive counts or 'full'.")
            sizes.add(size)
    return sorted(sizes)


def nested_class_order(labels, seed):
    """Return a class-covering, proportion-preserving ordering without replacement."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    rows = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
    taken = np.ones(len(classes), dtype=int)
    order = [int(indices[0]) for indices in rows]
    proportions = counts / len(labels)
    for next_size in range(len(classes) + 1, len(labels) + 1):
        deficit = next_size * proportions - taken
        deficit[taken == counts] = -np.inf
        # np.unique sorts classes; argmax resolves ties by class label.
        c = int(np.argmax(deficit))
        order.append(int(rows[c][taken[c]]))
        taken[c] += 1
    return np.asarray(order, dtype=np.int64)


def class_counts(labels):
    classes, counts = np.unique(labels, return_counts=True)
    return {str(int(c)): int(n) for c, n in zip(classes, counts)}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Strict JSON, with nonfinite scores explicitly represented as null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Atomically publish compact, single-line JSON without reducing precision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = output.name
            json.dump(json_safe(value), output, separators=(",", ":"), allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def saved_path(value):
    """Represent paths without disclosing the workstation/HPC directory prefix."""
    project = Path(__file__).resolve().parent
    text = os.fspath(value).replace("\\", "/")
    # Imported metadata may refer to a different checkout on an HPC system.
    parts = text.split("/")
    if project.name in parts:
        project_index = len(parts) - 1 - parts[::-1].index(project.name)
        relative = Path(*parts[project_index + 1:])
        if ".." not in relative.parts:
            return relative.as_posix()
    if PureWindowsPath(text).is_absolute():
        return "<external>/" + PureWindowsPath(text).name
    absolute = Path(os.path.abspath(Path(text).expanduser()))
    try:
        return absolute.relative_to(project).as_posix()
    except ValueError:
        # Outside-project inputs cannot be represented relative to the project
        # without revealing parent directories. Content hashes still identify them.
        return "<external>/" + absolute.name


def sanitize_saved_paths(value, key=""):
    """Remove absolute paths even from nested, copied source metadata/CLI args."""
    if isinstance(value, dict):
        return {k: sanitize_saved_paths(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_saved_paths(v, key) for v in value]
    if isinstance(value, str):
        path_key = key in {"path", "manifest", "checkpoint", "filename", "saved_model_name"} or key.endswith(("_path", "_root", "_loc"))
        absolute = value.startswith(("/", "~", "\\")) or PureWindowsPath(value).is_absolute()
        if absolute or (path_key and ("/" in value or "\\" in value)):
            return saved_path(value)
    return value


def experiment_identity(args, metadata, device, sizes, active_links):
    root = Path(args.logits_path).resolve()
    sources = {}
    for split in ("val", "test"):
        path = root / (split + "_logits_labels_indices.npz")
        sources[split] = {"path": str(path), "sha256": sha256(path)}
    code_root = Path(__file__).resolve().parent
    code_files = ["calibration_size_sweep.py", "evaluate.py",
                  "evaluate_focal_calibration.py", "temperature_scaling.py",
                  "new_links.py", "Metrics/metrics.py"]
    code_files += [str(p.relative_to(code_root)) for p in
                   sorted((code_root / "Calibs").glob("*.py")) if not p.name.startswith("test_")]
    sms = None
    if getattr(args, "sms", False):
        sms = {"reg_grid": args.sms_reg_grid, "cv_folds": args.sms_cv_folds,
               "penalty": "ridge", "rho": 1.0, "tau": 1.0, "opt": "bfgs",
               "provenance": sms_provenance()}
        code_files += [str(p.relative_to(code_root)) for p in
                       sorted((code_root / "Calibs/probmetrics/probmetrics").rglob("*.py"))]
        code_files.append("Calibs/probmetrics/pyproject.toml")
    return json_safe(sanitize_saved_paths({
        "version": 2, "sources": sources, "source_metadata": metadata,
        "code_sha256": {name: sha256(code_root / name) for name in code_files},
        "dataset": args.dataset, "model": args.model, "model_name": args.model_name,
        "checkpoint": args.saved_model_name, "model_seed": args.seed,
        "sizes": sizes, "subset_seeds": args.subset_seeds,
        "links": active_links, "posthoc_order": args.posthoc_order,
        "metrics": ["CE", "ECE"], "json_precision": args.json_precision,
        "dirichlet": args.dirichlet,
        "dirichlet_reg_grid": args.dirichlet_reg_grid,
        "dirichlet_cv_folds": args.dirichlet_cv_folds,
        "dirichlet_max_iter": args.dirichlet_max_iter,
        "dirichlet_n_jobs": args.dirichlet_n_jobs,
        "sms": sms,
        "device": str(device), "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }))


def unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def metric_pair(metrics):
    return {"NLL": float(metrics["CE"]), "ECE": float(sum(metrics["ECE"]["ece"]))}


def selected_methods(stats, active_links, order, include_dirichlet, dirichlet_reason=None,
                     include_sms=False):
    """Select with validation NLL only, before attaching any test metrics."""
    selections = {}
    for link, values in active_links.items():
        for scaled in (False, True):
            name = method_name_for_link(link, scaled, order)
            branch = stats["val"]["calibrated"]["ce"] if scaled else stats["val"]["uncalibrated"]
            candidates = branch[link]
            keys = [link_value_to_key(v) for v in values]
            finite_keys = [key for key in keys if math.isfinite(candidates[key]["CE"])]
            if not finite_keys:
                selections[name] = unavailable("No finite validation NLL candidate.")
                continue
            key = min(finite_keys, key=lambda k: candidates[k]["CE"])
            value = values[keys.index(key)]
            selections[name] = {
                "status": "ok", "selection_metric": "subset_validation_NLL",
                "parameters": {"link": link, "link_value": value,
                               "temperature": stats["T_dict"][link][key][" T_opt ce"] if scaled else 1,
                               "key": key, "temperature_metric": "ce" if scaled else "none"},
                "val": metric_pair(candidates[key]),
            }
            test_branch = stats["test"]["calibrated"]["ce"] if scaled else stats["test"]["uncalibrated"]
            selections[name]["test"] = metric_pair(test_branch[link][key])
    if include_dirichlet:
        if dirichlet_reason:
            selections["softmax+dirichlet"] = unavailable(dirichlet_reason)
        else:
            result = stats["dirichlet_calibrated"]["softmax"]["full_odir"]
            selections["softmax+dirichlet"] = {
                "status": "ok", "selection_metric": "subset_CV_NLL",
                "parameters": result["best"],
                "val": metric_pair(result["val"]), "test": metric_pair(result["test"]),
            }
    if include_sms:
        result = stats["sms_calibrated"]["softmax"]["sms"]
        selected = {"status": result["status"], "selection_metric": result["selection_metric"]}
        if "best" in result:
            selected["parameters"] = result["best"]
        if "reason" in result:
            selected["reason"] = result["reason"]
        for split in ("val", "test"):
            if split in result:
                selected[split] = metric_pair(result[split])
        selections["softmax+sms"] = selected
    for result in selections.values():
        if result["status"] == "ok" and any(
                not math.isfinite(v) for split in ("val", "test") for v in result[split].values()):
            # Keep the chosen parameters; never choose a different candidate using test results.
            result.update(status="unavailable", reason="Selected map produced nonfinite metrics.")
    return json_safe(selections)


def method_names(active_links, order, include_dirichlet, include_sms=False):
    names = [method_name_for_link(link, scaled, order)
             for link in active_links for scaled in (False, True)]
    return names + (["softmax+dirichlet"] if include_dirichlet else []) + (["softmax+sms"] if include_sms else [])


def summarize(records, names, requested, full=False):
    result = {}
    for name in names:
        good = [r["selected_methods"][name] for r in records
                if r["selected_methods"][name]["status"] == "ok"]
        failures = [{"subset_seed": r["sweep"]["subset_seed"],
                     "reason": r["selected_methods"][name]["reason"]}
                    for r in records if r["selected_methods"][name]["status"] != "ok"]
        entry = {"successful_repetitions": len(good), "requested_repetitions": requested,
                 "unavailable_repetitions": failures,
                 "status": "ok" if len(good) == requested else "partial" if good else "unavailable"}
        for split in ("val", "test"):
            entry[split] = {}
            for metric in ("NLL", "ECE"):
                values = [r[split][metric] for r in good]
                entry[split][metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 and not full else None,
                }
        result[name] = entry
    return result


def run_calibration_size_sweep(args, metadata, device, load_split=None):
    load_split = load_split or load_split_npz
    if metadata.get("smoke_test", False):
        raise ValueError("Sweeps require regular calibration data, not smoke-test artifacts; use small cached fixtures without smoke-test metadata.")
    include_sms = getattr(args, "sms", False)
    if include_sms:
        load_sms_class()
    if (not args.subset_seeds or len(set(args.subset_seeds)) != len(args.subset_seeds)
            or any(s < 0 or s >= 2**32 for s in args.subset_seeds)):
        raise ValueError("Subset seeds must be unique integers in [0, 2**32).")
    active_links = get_active_link_dict(args.links)
    val_logits, val_labels = load_split(args.logits_path, "val", device)
    test_logits, test_labels = load_split(args.logits_path, "test", device)
    num_classes = val_logits.shape[1]
    if test_logits.shape[1] != num_classes or metadata.get("num_classes", num_classes) != num_classes:
        raise ValueError("Validation/test logits and metadata must agree on class count.")
    for split, logits, labels in (("val", val_logits, val_labels), ("test", test_logits, test_labels)):
        if not len(labels) or not torch.isfinite(logits).all() or labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError("{} requires nonempty, finite logits and labels in the class range.".format(split))
    labels_np = val_labels.cpu().numpy()
    pool_size = len(labels_np)
    counts = class_counts(labels_np)
    sizes = resolve_sizes(args.calibration_sizes, pool_size)
    identity = experiment_identity(args, metadata, device, sizes, active_links)
    filename = Path(evaluation_json_path(args.save_eval_loc, args.saved_model_name, args.model_name)).name
    manifest_path = Path(args.save_eval_loc) / (Path(filename).stem + "_sweep_manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError("Sweep provenance/settings mismatch at {}; use a separate output destination.".format(manifest_path))
    else:
        write_json(manifest_path, identity)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    orders = {seed: nested_class_order(labels_np, seed) for seed in args.subset_seeds}
    names = method_names(active_links, args.posthoc_order, args.dirichlet, include_sms)
    summaries = {}
    for size in sizes:
        full = size == pool_size
        size_dir = Path(args.save_eval_loc) / ("subset-{}".format(size))
        reason = ("Requested size exceeds validation pool." if size > pool_size else
                  "Requested size is smaller than validation class count." if size < len(counts) else None)
        records = []
        for seed in ([None] if full else args.subset_seeds):
            rows = np.arange(pool_size) if full else orders[seed][:size]
            if reason:
                rows = np.array([], dtype=np.int64)
            run_meta = {
                "experiment_sha256": fingerprint, "manifest": saved_path(manifest_path),
                "model_seed": args.seed, "subset_seed": seed,
                "fit_seed": args.seed if full else seed,
                "requested_size": size, "actual_size": len(rows), "pool_size": pool_size,
                "class_counts": class_counts(labels_np[rows]), "pool_class_counts": counts,
                "row_positions": rows.tolist(), "row_identity": "source-artifact-local",
                "full_pool": full,
            }
            path = size_dir / ("full" if full else "repeat-{}".format(seed)) / filename
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("sweep") != run_meta or set(record.get("selected_methods", {})) != set(names):
                    raise ValueError("Existing sweep result mismatch at {}; use a separate destination.".format(path))
                print("Reusing verified sweep result: {}".format(path))
            else:
                start = time.monotonic()
                if reason:
                    record = {"status": "unavailable", "reason": reason,
                              "selected_methods": {name: unavailable(reason) for name in names}}
                else:
                    set_seed(run_meta["fit_seed"])
                    idx = torch.as_tensor(rows, device=device)
                    subset_logits, subset_labels = val_logits[idx], val_labels[idx]
                    print("Calibration sweep: model_seed={} size={} subset_seed={}".format(args.seed, size, seed))
                    stats = focal_calibration_evaluation(
                        nn.Identity().to(device), subset_logits, subset_labels, test_logits, test_labels,
                        num_classes=num_classes, device=device, links=args.links,
                        seed=run_meta["fit_seed"], posthoc_order=args.posthoc_order, metrics=("CE", "ECE"))
                    dirichlet_reason = None
                    if args.dirichlet:
                        if len(counts) != num_classes:
                            dirichlet_reason = "Current Dirichlet implementation requires every output class in the validation pool."
                        elif min(args.dirichlet_cv_folds, min(run_meta["class_counts"].values())) < 2:
                            dirichlet_reason = "Dirichlet CV requires at least two examples per class and two folds."
                        else:
                            stats["dirichlet_calibrated"] = dirichlet_calibration_evaluation(
                                subset_logits, subset_labels, test_logits, test_labels,
                                num_classes=num_classes, device=device, reg_grid=args.dirichlet_reg_grid,
                                cv_folds=args.dirichlet_cv_folds, max_iter=args.dirichlet_max_iter,
                                n_jobs=args.dirichlet_n_jobs, seed=run_meta["fit_seed"], metrics=("CE", "ECE"))
                    if include_sms:
                        stats["sms_calibrated"] = sms_calibration_evaluation(
                            subset_logits, subset_labels, test_logits, test_labels,
                            num_classes=num_classes, device=device, reg_grid=args.sms_reg_grid,
                            cv_folds=args.sms_cv_folds, seed=run_meta["fit_seed"], metrics=("CE", "ECE"))
                    selected = selected_methods(stats, active_links, args.posthoc_order, args.dirichlet,
                                                dirichlet_reason, include_sms=include_sms)
                    record = round_floats_for_json(stats, args.json_precision)
                    # Retain full precision for selection/aggregation, including after resume.
                    record["selected_methods"] = selected
                    record["unavailable_methods"] = {n: r["reason"] for n, r in selected.items() if r["status"] != "ok"}
                record["sweep"] = run_meta
                record["elapsed_seconds"] = time.monotonic() - start
                write_json(path, record)
                print("Saved sweep result: {}".format(path))
            records.append(record)
        summary = {
            "experiment_sha256": fingerprint, "model_seed": args.seed, "size": size,
            "full_pool": full, "variability": "sample SD across calibration draws within this model; null for fewer than two draws",
            "aggregation_note": "Average draws within each model before aggregating model seeds. Draws are not independent trained models.",
            "methods": summarize(records, names, len(records), full),
        }
        write_json(size_dir / (Path(filename).stem + "_summary.json"), summary)
        summaries[size] = summary
    print("Calibration-size sweep complete: {}".format(manifest_path))
    return summaries


if __name__ == "__main__":
    main()
