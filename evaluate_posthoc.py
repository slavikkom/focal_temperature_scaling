import argparse
import json
import os

import numpy as np
import torch

from evaluate import (
    get_probability_export_splits,
    normalize_links,
    round_floats_for_json,
    run_posthoc_evaluation,
    set_seed,
)


SPLIT_FILENAMES = {
    "train": "train_logits_labels_indices.npz",
    "val": "val_logits_labels_indices.npz",
    "test": "test_logits_labels_indices.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run post-hoc calibration from saved logits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logits-path", required=True,
                        help="Directory containing split logits .npz files and optional logits_metadata.json.")
    parser.add_argument("--save-eval-path", default=None, dest="save_eval_loc",
                        help="Directory to save post-hoc evaluation JSON/probability outputs. Defaults to --logits-path.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name. Defaults to logits_metadata.json when present.")
    parser.add_argument("--model", default=None,
                        help="Model architecture name for metadata/logging only.")
    parser.add_argument("--model-name", default=None, dest="model_name",
                        help="Model name prefix used to derive the output JSON filename.")
    parser.add_argument("--saved_model_name", default=None, dest="saved_model_name",
                        help="Checkpoint filename used to derive the output JSON filename.")
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Accepted for command compatibility; posthoc uses CUDA when available.")
    parser.set_defaults(gpu=False)
    parser.add_argument("-log", action="store_true", dest="log",
                        help="Accepted for command compatibility.")
    parser.set_defaults(log=False)
    parser.add_argument("--links", nargs="+", default=None,
                        help="Calibration links to evaluate. Omit to evaluate all links.")
    parser.add_argument("--dirichlet", action="store_true", dest="dirichlet",
                        help="Evaluate full ODIR Dirichlet calibration on softmax probabilities.")
    parser.set_defaults(dirichlet=False)
    parser.add_argument("--dirichlet-cv-folds", type=int, default=3,
                        dest="dirichlet_cv_folds",
                        help="Number of validation-folds for Dirichlet GridSearchCV.")
    parser.add_argument("--dirichlet-reg-grid", nargs="+", type=float, default=None,
                        dest="dirichlet_reg_grid",
                        help="Regularization values for Dirichlet lambda and mu grid search.")
    parser.add_argument("--dirichlet-max-iter", type=int, default=1024,
                        dest="dirichlet_max_iter",
                        help="Maximum LBFGS iterations for each Dirichlet fit.")
    parser.add_argument("--dirichlet-n-jobs", type=int, default=1,
                        dest="dirichlet_n_jobs",
                        help="Parallel GridSearchCV workers for Dirichlet calibration.")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Use smoke-test settings for calibration helpers.")
    parser.set_defaults(smoke_test=False)
    parser.add_argument("--seed", type=int, default=None,
                        dest="seed", help="Random seed for reproducibility. Defaults to metadata seed or 1.")
    parser.add_argument("--save-probs", action="store_true", dest="save_probs",
                        help="Save train/val/test probabilities for the selected --links.")
    parser.set_defaults(save_probs=False)
    parser.add_argument("--save-train-probs", action="store_true", dest="save_train_probs",
                        help="Save train probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--save-val-probs", action="store_true", dest="save_val_probs",
                        help="Save validation probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--save-test-probs", action="store_true", dest="save_test_probs",
                        help="Save test probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--json-precision", type=int, default=5, dest="json_precision",
                        help="Number of decimal places to keep for floating point values saved to JSON.")
    parser.add_argument("--train-smece-sample-size", type=int, default=5000,
                        dest="train_smooth_ece_sample_size",
                        help=(
                            "Number of train examples to use when estimating smECE and smECE_0.05. "
                            "Use 0 or a negative value to evaluate the full train set."
                        ))
    return parser.parse_args()


def load_metadata(logits_path):
    metadata_path = os.path.join(logits_path, "logits_metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_split_npz(logits_path, split_name, device):
    filename = SPLIT_FILENAMES[split_name]
    path = os.path.join(logits_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("Missing {} logits file: {}".format(split_name, path))

    data = np.load(path, allow_pickle=False)
    if "logits" not in data or "labels" not in data:
        raise ValueError("{} must contain 'logits' and 'labels' arrays.".format(path))

    logits = torch.as_tensor(data["logits"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(data["labels"], dtype=torch.long, device=device).view(-1)
    if logits.ndim != 2:
        raise ValueError("{} logits must have shape (n_examples, num_classes).".format(path))
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            "{} labels length {} does not match logits rows {}.".format(
                path, labels.shape[0], logits.shape[0]
            )
        )
    return logits, labels


def fill_args_from_metadata(args, metadata):
    args.save_eval_loc = args.save_eval_loc or args.logits_path
    args.dataset = args.dataset or metadata.get("dataset") or "logits"
    args.model = args.model or metadata.get("model") or metadata.get("model_name") or "model"
    args.model_name = args.model_name or metadata.get("model_name") or args.model
    args.saved_model_name = (
        args.saved_model_name
        or metadata.get("saved_model_name")
        or "{}_posthoc.model".format(args.model_name)
    )
    args.seed = int(args.seed if args.seed is not None else metadata.get("seed", 1))
    if not args.smoke_test:
        args.smoke_test = bool(metadata.get("smoke_test", False))
    args.links = normalize_links(args.links)
    return args


def infer_num_classes(metadata, train_logits, val_logits, test_logits):
    if "num_classes" in metadata:
        return int(metadata["num_classes"])
    num_classes = int(train_logits.shape[1])
    for split_name, logits in [("val", val_logits), ("test", test_logits)]:
        if int(logits.shape[1]) != num_classes:
            raise ValueError(
                "{} logits have {} classes, expected {}.".format(
                    split_name, int(logits.shape[1]), num_classes
                )
            )
    return num_classes


def main():
    args = parse_args()
    metadata = load_metadata(args.logits_path)
    args = fill_args_from_metadata(args, metadata)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logits, train_labels = load_split_npz(args.logits_path, "train", device)
    val_logits, val_labels = load_split_npz(args.logits_path, "val", device)
    test_logits, test_labels = load_split_npz(args.logits_path, "test", device)
    num_classes = infer_num_classes(metadata, train_logits, val_logits, test_logits)

    os.makedirs(args.save_eval_loc, exist_ok=True)
    stats, json_path = run_posthoc_evaluation(
        args,
        train_logits,
        train_labels,
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        num_classes,
        device,
        net=None,
    )
    print("Post-hoc evaluation complete: {}".format(json_path))
    if get_probability_export_splits(args):
        print("Probability exports saved to: {}".format(args.save_eval_loc))
    return round_floats_for_json(stats, args.json_precision)


if __name__ == "__main__":
    main()
