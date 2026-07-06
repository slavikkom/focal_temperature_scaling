import os
import sys
import torch
import json
import random
import argparse
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar10_c as cifar10_c
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet
import Data.medmnist_loader as get_medmnist_data_loader
import medmnist

# Import network architectures
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.resnet import resnet18, resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature
from evaluate_focal_calibration import *

try:
    import fcntl
except ImportError:
    fcntl = None

medmnist_datasets = [
    'pathmnist', 
    'dermamnist', 
    'octmnist', 
    'pneumoniamnist', 
    'retinamnist', 
    'breastmnist', 
    'bloodmnist',
    'tissuemnist',
    'organamnist',
    'organcmnist',
    'organsmnist'
]


# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar10_c': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
}

for name in medmnist_datasets:
    dataset_num_classes[name] = len(medmnist.INFO[name]['label'])

dataset_loader = {
    'cifar10': cifar10,
    'cifar10_c': cifar10_c,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet,
}

for name in medmnist_datasets:
    dataset_loader[name] = get_medmnist_data_loader

# Mapping model name to model function
models = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = '../Data/datasets'
    model = 'resnet50'
    save_loc = './'
    save_eval_loc = './'
    saved_model_name = 'resnet50_cross_entropy_350.model'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--save-eval-path", type=str, default=save_eval_loc,
                        dest="save_eval_loc",
                        help='Path to save evaluations of the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=False)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("--links", nargs="+", default=None,
                        help="Calibration links to evaluate. Choices: all, {}. Omit to evaluate all links.".format(
                            ", ".join(link_dict.keys())
                        ))
    parser.add_argument("--dirichlet", action="store_true", dest="dirichlet",
                        help="Evaluate full ODIR Dirichlet calibration on softmax probabilities")
    parser.set_defaults(dirichlet=False)
    parser.add_argument("--dirichlet-cv-folds", type=int, default=3,
                        dest="dirichlet_cv_folds",
                        help="Number of validation-folds for Dirichlet GridSearchCV")
    parser.add_argument("--dirichlet-reg-grid", nargs="+", type=float, default=None,
                        dest="dirichlet_reg_grid",
                        help="Regularization values for Dirichlet lambda and mu grid search")
    parser.add_argument("--dirichlet-max-iter", type=int, default=1024,
                        dest="dirichlet_max_iter",
                        help="Maximum LBFGS iterations for each Dirichlet fit")
    parser.add_argument("--dirichlet-n-jobs", type=int, default=1,
                        dest="dirichlet_n_jobs",
                        help="Parallel GridSearchCV workers for Dirichlet calibration")
    parser.add_argument("--corruption", type=str, default="gaussian_noise",
                        dest="corruption",
                        help="CIFAR-10-C corruption to evaluate, or 'all'")
    parser.add_argument("--severity", type=str, default="1",
                        dest="severity",
                        help="CIFAR-10-C severity level 1-5, or 'all'")
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")

    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                    help="Run a lightweight smoke test")
    parser.set_defaults(smoke_test=False)

    parser.add_argument("--seed", type=int, default=1,
        dest="seed", help="random seed for reproducibility")

    parser.add_argument("--inference-only", action="store_true", dest="inference_only",
        help=(
            "Only run train/validation/test inference and save a complete logits artifact. "
            "Skips all post-hoc calibration, probability exports, and evaluation JSON writing."
        ))
    parser.set_defaults(inference_only=False)

    parser.add_argument("--save-train-logits", action="store_true", dest="save_train_logits",
        help="Save train logits, labels, and indices to file.")
    parser.add_argument("--save-val-logits", action="store_true", dest="save_val_logits",
        help="Save validation logits, labels, and indices to file.")
    parser.add_argument("--save-test-logits", action="store_true", dest="save_test_logits",
        help="Save test logits, labels, and indices to file.")
    parser.add_argument("--save-probs", action="store_true", dest="save_probs",
        help=(
            "Save train/val/test probabilities for the selected --links. Alias for "
            "--save-train-probs --save-val-probs --save-test-probs. "
            "For each link, save the uncalibrated and CE-selected temperature-scaled probabilities. "
            "If --dirichlet is set, also save softmax+dirichlet probabilities."
        ))
    parser.set_defaults(save_probs=False)
    parser.add_argument("--save-train-probs", action="store_true", dest="save_train_probs",
        help="Save train probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--save-val-probs", action="store_true", dest="save_val_probs",
        help="Save validation probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--save-test-probs", action="store_true", dest="save_test_probs",
        help="Save test probabilities for the selected --links and optional --dirichlet calibration.")
    parser.add_argument("--json-precision", type=int, default=5, dest="json_precision",
        help="Number of decimal places to keep for floating point values saved to JSON.")

    return parser.parse_args()


LINK_ALIASES = {
    "exp1mp": "exp_1mp",
    "exp_1mp": "exp_1mp",
    "expon1mp": "exp_1mp",
    "exp_on_1mp": "exp_1mp",
    "exponp": "exp_p",
    "exp_p": "exp_p",
    "exp_on_p": "exp_p",
}

LINK_DISPLAY_NAMES = {
    "softmax": "softmax",
    "focal": "focal",
    "focal_linear": "focal_linear",
    "exp_p": "exponp",
    "exp_1mp": "exp1mp",
    "one_minus_power": "one_minus_power",
    "generalized_focal": "generalized_focal",
    "log_power": "log_power",
}


def normalize_links(links):
    if links is None:
        return None

    normalized = []
    for link in links:
        link_key = link.strip().lower().replace("-", "_")
        link_key = LINK_ALIASES.get(link_key, link_key)
        if link_key == "all":
            return ["all"]
        if link_key not in link_dict:
            raise ValueError(
                "Unknown calibration link '{}'. Valid links are: {}".format(
                    link, ", ".join(link_dict.keys())
                )
            )
        if link_key not in normalized:
            normalized.append(link_key)
    return normalized


def get_active_link_dict(links):
    if links is None or "all" in links:
        return link_dict
    return {link: link_dict[link] for link in links}


def get_probability_export_splits(args):
    if args.save_probs:
        return ["train", "val", "test"]

    splits = []
    if args.save_train_probs:
        splits.append("train")
    if args.save_val_probs:
        splits.append("val")
    if args.save_test_probs:
        splits.append("test")
    return splits


def link_value_to_key(link_value):
    if isinstance(link_value, tuple):
        return "_".join(str(round(v, 2)) for v in link_value)
    return str(round(link_value, 2))


def key_to_filename_token(key):
    return key.replace("_", "-")


def format_number_for_filename(value):
    value = float(value)
    text = "{:.4f}".format(value).rstrip("0").rstrip(".")
    return text if text else "0"


def method_name_for_link(link_name, temperature_scaled=False):
    display_name = LINK_DISPLAY_NAMES.get(link_name, link_name)
    if link_name == "softmax":
        method_name = "softmax"
    else:
        method_name = "softmax+{}".format(display_name)
    if temperature_scaled:
        method_name += "+ts"
    return method_name


def best_link_value_by_val_ce(stats, active_link_dict, link_name, temperature_scaled=False):
    state_stats = stats["val"]["calibrated"]["ce"] if temperature_scaled else stats["val"]["uncalibrated"]
    link_stats = state_stats[link_name]
    best_key = None
    best_ce = None
    for key, metrics in link_stats.items():
        ce = metrics["CE"]
        if best_ce is None or ce < best_ce:
            best_ce = ce
            best_key = key

    key_to_value = {
        link_value_to_key(link_value): link_value
        for link_value in active_link_dict[link_name]
    }
    if best_key not in key_to_value:
        raise ValueError(
            "Could not map selected key '{}' back to a parameter value for link '{}'.".format(
                best_key, link_name
            )
        )

    return best_key, key_to_value[best_key], best_ce


def probability_export_specs_from_stats(stats, active_link_dict, include_dirichlet=False):
    specs = []
    for link_name in active_link_dict:
        best_key, best_link_value, best_ce = best_link_value_by_val_ce(
            stats, active_link_dict, link_name, temperature_scaled=False)
        method_name = method_name_for_link(link_name, temperature_scaled=False)
        suffix = "" if link_name == "softmax" else "_{}".format(key_to_filename_token(best_key))
        specs.append({
            "method_name": method_name,
            "filename_suffix": suffix,
            "link_name": link_name,
            "link_value": best_link_value,
            "temperature": 1,
            "temperature_metric": "none",
            "val_ce": best_ce,
            "calibrated_on": "none",
        })

        best_key, best_link_value, best_ce = best_link_value_by_val_ce(
            stats, active_link_dict, link_name, temperature_scaled=True)
        temperature = stats["T_dict"][link_name][best_key][" T_opt ce"]
        method_name = method_name_for_link(link_name, temperature_scaled=True)
        if link_name == "softmax":
            suffix_parts = [format_number_for_filename(temperature)]
        else:
            suffix_parts = [
                key_to_filename_token(best_key),
                format_number_for_filename(temperature),
            ]
        specs.append({
            "method_name": method_name,
            "filename_suffix": "_{}".format("+".join(suffix_parts)),
            "link_name": link_name,
            "link_value": best_link_value,
            "temperature": temperature,
            "temperature_metric": "ce",
            "val_ce": best_ce,
            "calibrated_on": "val",
        })

    if include_dirichlet:
        specs.append({
            "method_name": "softmax+dirichlet",
            "filename_suffix": "",
            "link_name": "softmax",
            "link_value": 1,
            "temperature": 1,
            "temperature_metric": "none",
            "val_ce": None,
            "calibrated_on": "val",
            "dirichlet": True,
        })
    return specs


def log_probability_export_specs(specs):
    print("Probability exports selected by --links/--dirichlet:")
    for spec in specs:
        detail = spec["method_name"] + spec["filename_suffix"]
        if spec.get("val_ce") is not None:
            detail += " (val CE {:.5f})".format(spec["val_ce"])
        print("  {}".format(detail))


def to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def save_logits_labels_indices_npz(filename, logits, labels, indices=None,
                                   split=None, num_classes=None):
    logits = to_numpy(logits)
    labels = to_numpy(labels)
    if indices is None:
        indices = np.arange(len(labels))
    arrays = {
        "logits": logits,
        "labels": labels,
        "indices": indices,
    }
    if split is not None:
        arrays["split"] = np.asarray(split)
    if num_classes is not None:
        arrays["num_classes"] = np.asarray(num_classes)
    np.savez(filename, **arrays)


def save_probs_labels_indices_npz(filename, probs, labels, indices=None, metadata=None):
    probs = to_numpy(probs)
    labels = to_numpy(labels)
    if indices is None:
        indices = np.arange(len(labels))

    arrays = {
        "probs": probs,
        "labels": labels,
        "indices": indices,
    }
    if metadata is not None:
        for key, value in metadata.items():
            arrays[key] = np.asarray(value)
    np.savez(filename, **arrays)


def save_probability_exports(save_eval_loc, split_logits, split_labels, specs,
                             dirichlet_probabilities=None):
    with torch.inference_mode():
        for spec in specs:
            metadata = {
                "method": spec["method_name"],
                "link": spec["link_name"],
                "link_value": spec["link_value"],
                "temperature": float(spec["temperature"]),
                "temperature_metric": spec["temperature_metric"],
                "calibrated_on": spec["calibrated_on"],
                "selection_metric": "val_ce",
            }
            if spec.get("val_ce") is not None:
                metadata["val_ce"] = float(spec["val_ce"])

            if spec.get("dirichlet"):
                if dirichlet_probabilities is None:
                    raise ValueError(
                        "Dirichlet probabilities were requested but were not computed."
                    )
                for split_name, labels in split_labels.items():
                    if split_name not in dirichlet_probabilities:
                        continue
                    filename = os.path.join(
                        save_eval_loc,
                        "{}_probs_{}{}.npz".format(
                            split_name, spec["method_name"], spec["filename_suffix"]
                        ),
                    )
                    save_probs_labels_indices_npz(
                        filename,
                        dirichlet_probabilities[split_name],
                        labels,
                        metadata=metadata,
                    )
                    print("Saved probabilities: {}".format(filename))
                continue

            for split_name, logits in split_logits.items():
                filename = os.path.join(
                    save_eval_loc,
                    "{}_probs_{}{}.npz".format(
                        split_name, spec["method_name"], spec["filename_suffix"]
                    ),
                )
                probs = get_probs(
                    logits,
                    T=spec["temperature"],
                    a=spec["link_value"],
                    link=spec["link_name"],
                )
                save_probs_labels_indices_npz(
                    filename,
                    probs,
                    split_labels[split_name],
                    metadata=metadata,
                )
                print("Saved probabilities: {}".format(filename))

def round_floats_for_json(obj, precision=5):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, np.floating):
        return round(float(obj), precision)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return round_floats_for_json(obj.tolist(), precision)
    if torch.is_tensor(obj):
        return round_floats_for_json(obj.detach().cpu().tolist(), precision)
    if isinstance(obj, dict):
        return {k: round_floats_for_json(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats_for_json(v, precision) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats_for_json(v, precision) for v in obj)
    return obj

def merge_evaluation_stats(existing_stats, new_stats, replace_keys=None):
    if replace_keys is None:
        replace_keys = set(link_dict.keys())

    if not isinstance(existing_stats, dict) or not isinstance(new_stats, dict):
        return new_stats

    merged_stats = dict(existing_stats)
    for key, value in new_stats.items():
        if key in merged_stats and key not in replace_keys:
            merged_stats[key] = merge_evaluation_stats(
                merged_stats[key], value, replace_keys=replace_keys
            )
        else:
            merged_stats[key] = value
    return merged_stats

def merge_with_existing_json(stats, json_path):
    if not os.path.exists(json_path):
        return stats, False

    with open(json_path, 'r') as f:
        existing_stats = json.load(f)

    if not isinstance(existing_stats, dict):
        raise ValueError(
            "Existing evaluation JSON must contain an object at the top level: {}".format(json_path)
        )

    return merge_evaluation_stats(existing_stats, stats), True

def save_evaluation_json(stats, json_path):
    lock_file = None
    lock_path = json_path + ".lock"
    try:
        if fcntl is not None:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        merged_stats, merged_existing_json = merge_with_existing_json(stats, json_path)
        if merged_existing_json:
            print("Updating existing evaluation JSON: {}".format(json_path))
        else:
            print("Writing new evaluation JSON: {}".format(json_path))

        with open(json_path, 'w') as f:
            json.dump(merged_stats, f)
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

def json_safe_args(args):
    safe_args = {}
    for key, value in vars(args).items():
        safe_args[key] = round_floats_for_json(value)
    return safe_args

def logits_metadata(args, split_logits, split_labels, split_filenames,
                    num_classes, num_channels):
    split_info = {}
    for split_name, logits in split_logits.items():
        split_info[split_name] = {
            "filename": split_filenames[split_name],
            "num_examples": int(len(split_labels[split_name])),
            "logits_shape": list(to_numpy(logits).shape),
        }

    metadata = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "model": args.model,
        "model_name": args.model_name,
        "saved_model_name": args.saved_model_name,
        "save_path": args.save_loc,
        "seed": int(args.seed),
        "smoke_test": bool(args.smoke_test),
        "num_classes": int(num_classes),
        "num_channels": int(num_channels),
        "train_batch_size": int(args.train_batch_size),
        "test_batch_size": int(args.test_batch_size),
        "data_aug": bool(args.data_aug),
        "splits": split_info,
        "cli_args": json_safe_args(args),
    }
    if args.dataset == "cifar10_c":
        metadata["corruption"] = args.corruption
        metadata["severity"] = args.severity
        metadata["train_val_source"] = "clean_cifar10"
    return metadata

def save_logits_artifact(save_eval_loc, args, split_logits, split_labels,
                         num_classes, num_channels):
    if not os.path.exists(save_eval_loc):
        os.makedirs(save_eval_loc)

    split_filenames = {
        "train": "train_logits_labels_indices.npz",
        "val": "val_logits_labels_indices.npz",
        "test": "test_logits_labels_indices.npz",
    }
    for split_name in ["train", "val", "test"]:
        filename = os.path.join(save_eval_loc, split_filenames[split_name])
        save_logits_labels_indices_npz(
            filename,
            split_logits[split_name],
            split_labels[split_name],
            split=split_name,
            num_classes=num_classes,
        )
        print("Saved {} logits: {}".format(split_name, filename))

    metadata = logits_metadata(
        args,
        split_logits,
        split_labels,
        split_filenames,
        num_classes,
        num_channels,
    )
    metadata_path = os.path.join(save_eval_loc, "logits_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved logits metadata: {}".format(metadata_path))

def evaluation_json_path(save_eval_loc, saved_model_name, model_name):
    model_stem = os.path.splitext(saved_model_name)[0]
    model_prefix = model_name + "_"
    if model_stem.startswith(model_prefix):
        saved_stats_name = model_stem[len(model_prefix):]
    else:
        saved_stats_name = model_stem
    save_stats_path = os.path.join(save_eval_loc, saved_stats_name)
    return save_stats_path + ".json"

def run_posthoc_evaluation(args, train_logits, train_labels, val_logits,
                           val_labels, test_logits, test_labels, num_classes,
                           device, net=None):
    active_link_dict = get_active_link_dict(args.links)
    probability_export_splits = get_probability_export_splits(args)
    save_any_probs = len(probability_export_splits) > 0

    if net is None:
        net = nn.Identity().to(device)

    stats = focal_calibration_evaluation(net, val_logits, val_labels, test_logits, test_labels,
                                          num_classes=num_classes, device=device,
                                          train_logits=train_logits,
                                          train_labels=train_labels,
                                          links=args.links)
    dirichlet_probabilities = None
    needs_dirichlet_probabilities = save_any_probs and args.dirichlet
    if args.dirichlet or needs_dirichlet_probabilities:
        dirichlet_result = dirichlet_calibration_evaluation(
            val_logits, val_labels, test_logits, test_labels,
            num_classes=num_classes, device=device,
            train_logits=train_logits,
            train_labels=train_labels,
            cv_folds=args.dirichlet_cv_folds,
            reg_grid=args.dirichlet_reg_grid,
            max_iter=args.dirichlet_max_iter,
            n_jobs=args.dirichlet_n_jobs,
            seed=args.seed,
            smoke_test=args.smoke_test,
            return_probabilities=needs_dirichlet_probabilities)
        if needs_dirichlet_probabilities:
            dirichlet_stats, dirichlet_probabilities = dirichlet_result
        else:
            dirichlet_stats = dirichlet_result
        stats["dirichlet_calibrated"] = dirichlet_stats

    if not os.path.exists(args.save_eval_loc):
        os.makedirs(args.save_eval_loc)

    save_stats_json_path = evaluation_json_path(
        args.save_eval_loc,
        args.saved_model_name,
        args.model_name,
    )
    rounded_stats = round_floats_for_json(stats, args.json_precision)
    save_evaluation_json(rounded_stats, save_stats_json_path)

    if save_any_probs:
        probability_export_specs = probability_export_specs_from_stats(
            stats,
            active_link_dict,
            include_dirichlet=args.dirichlet,
        )
        log_probability_export_specs(probability_export_specs)
        all_split_logits = {
            "train": train_logits,
            "val": val_logits,
            "test": test_logits,
        }
        all_split_labels = {
            "train": train_labels,
            "val": val_labels,
            "test": test_labels,
        }
        split_logits = {
            split: all_split_logits[split]
            for split in probability_export_splits
        }
        split_labels = {
            split: all_split_labels[split]
            for split in probability_export_splits
        }
        save_probability_exports(
            args.save_eval_loc,
            split_logits,
            split_labels,
            probability_export_specs,
            dirichlet_probabilities=dirichlet_probabilities,
        )

    return stats, save_stats_json_path

def get_logits_labels(data_loader, net, device):
    logits_list = []
    labels_list = []
    net.eval()
    # use inference mode to avoid extra book keeping overhead of autograd in eval mode
    # also make data loading non-blocking/asynchronous to speed up data transfer to GPU 
    with torch.inference_mode():
        for data, label in data_loader:
            # data = data.cuda()
            data = data.to(device, non_blocking=True)
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        # logits = torch.cat(logits_list).cuda()
        # labels = torch.cat(labels_list).cuda()
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    return logits, labels

def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed) # applicable to multi-GPU 

    # For full reproducibility avoid using any non-deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    args = parseArgs()
    args.links = normalize_links(args.links)

    # Setting additional parameters
    set_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    if args.smoke_test:
        print("Running in smoke test mode...")
        # args.gpu = False  
        args.test_batch_size = 16  # Small batch size

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    save_eval_loc = args.save_eval_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    num_channels = 3
    if args.dataset in medmnist_datasets:
        num_channels = medmnist.INFO[args.dataset]['n_channels']

    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)

        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='test',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)
    elif args.dataset in medmnist_datasets:
        train_loader = dataset_loader[args.dataset].get_medmnist_data_loader(
            dataset_name=args.dataset,
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)

        val_loader = dataset_loader[args.dataset].get_medmnist_data_loader(
            dataset_name=args.dataset,
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)

        test_loader = dataset_loader[args.dataset].get_medmnist_data_loader(
            dataset_name=args.dataset,
            root=args.dataset_root,
            split='test',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test)
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            data_dir=args.dataset_root,
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=args.seed,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test
        )

        test_loader_kwargs = {
            "data_dir": args.dataset_root,
            "batch_size": args.test_batch_size,
            "pin_memory": args.gpu,
            "smoke_test": args.smoke_test,
        }
        if args.dataset == 'cifar10_c':
            test_loader_kwargs["corruption"] = args.corruption
            test_loader_kwargs["severity"] = args.severity

        test_loader = dataset_loader[args.dataset].get_test_loader(**test_loader_kwargs)

    model = models[model_name]

    net = model(num_classes=num_classes, in_channels=num_channels, temp=1.0)
    if cuda:
        net.cuda()
        # net.to(device)
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    else:
        net.to(device)
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
    """
    from collections import OrderedDict

    # 1) Instantiate your model (no DataParallel yet)
    net = model(num_classes=num_classes, temp=1.0).to(device)

    # 2) Wrap in DataParallel
    if cuda:
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        cudnn.benchmark = True

    # 3) Load the raw checkpoint (saved without “module.” prefixes)
    raw_state = torch.load(args.save_loc + args.saved_model_name, map_location=device)

    # 4) Prepend “module.” to each key so it matches the DataParallel model’s keys
    new_state = OrderedDict()
    for key, val in raw_state.items():
        new_key = "module." + key
        new_state[new_key] = val

    # 5) Load into the DataParallel-wrapped model
    net.load_state_dict(new_state)
    """
    # nll_criterion = nn.CrossEntropyLoss().cuda()
    # ece_criterion = ECELoss().cuda()
    # adaece_criterion = AdaptiveECELoss().cuda()
    # cece_criterion = ClasswiseECELoss().cuda()

    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)
    adaece_criterion = AdaptiveECELoss().to(device)
    cece_criterion = ClasswiseECELoss().to(device)

    train_logits, train_labels = get_logits_labels(train_loader, net, device=device)
    val_logits, val_labels = get_logits_labels(val_loader, net, device=device)
    test_logits, test_labels = get_logits_labels(test_loader, net, device=device)

    if args.inference_only:
        split_logits = {
            "train": train_logits,
            "val": val_logits,
            "test": test_logits,
        }
        split_labels = {
            "train": train_labels,
            "val": val_labels,
            "test": test_labels,
        }
        save_logits_artifact(
            args.save_eval_loc,
            args,
            split_logits,
            split_labels,
            num_classes,
            num_channels,
        )
        print("Inference-only mode complete. Post-hoc calibration was skipped.")
        sys.exit(0)

    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(test_logits, test_labels)

    p_ece = ece_criterion(test_logits, test_labels).item()
    p_adaece = adaece_criterion(test_logits, test_labels).item()
    p_cece = cece_criterion(test_logits, test_labels).item()
    p_nll = nll_criterion(test_logits, test_labels).item()

    res_str = '{:s}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(saved_model_name,  1-p_accuracy,  p_nll,  p_ece,  p_adaece, p_cece)

    # Printing the required evaluation metrics
    if args.log:
        print (conf_matrix)
        print ('Test error: ' + str((1 - p_accuracy)))
        print ('Test NLL: ' + str(p_nll))
        print ('ECE: ' + str(p_ece))
        print ('AdaECE: ' + str(p_adaece))
        print ('Classwise ECE: ' + str(p_cece))


    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(val_logits, val_labels, cross_validate=cross_validation_error, device=device)
    T_opt = scaled_model.get_temperature()
    # logits, labels = get_logits_labels(test_loader, scaled_model, device=device)
    # save some inference time by directly scaling the test logits instead of doing a forward pass through the model again
    logits = test_logits / T_opt
    labels = test_labels
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()
    stats, save_stats_json_path = run_posthoc_evaluation(
        args,
        train_logits,
        train_labels,
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        num_classes,
        device,
        net=net,
    )

    if args.save_train_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'train_logits_labels_indices.npz'), train_logits, train_labels, split="train", num_classes=num_classes)
    if args.save_val_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'val_logits_labels_indices.npz'), val_logits, val_labels, split="val", num_classes=num_classes)
    if args.save_test_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'test_logits_labels_indices.npz'), test_logits, test_labels, split="test", num_classes=num_classes)

    res_str += '&{:.4f}({:.2f})&{:.4f}&{:.4f}&{:.4f}'.format(nll,  T_opt,  ece,  adaece, cece)

    if args.log:
        print ('Optimal temperature: ' + str(T_opt))
        print (conf_matrix)
        print ('Test error: ' + str((1 - accuracy)))
        print ('Test NLL: ' + str(nll))
        print ('ECE: ' + str(ece))
        print ('AdaECE: ' + str(adaece))
        print ('Classwise ECE: ' + str(cece))

    # Test NLL & ECE & AdaECE & Classwise ECE
    print(res_str)
