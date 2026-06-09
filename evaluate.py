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

    parser.add_argument("--save-train-logits", action="store_true", dest="save_train_logits",
        help="Save train logits, labels, and indices to file.")
    parser.add_argument("--save-val-logits", action="store_true", dest="save_val_logits",
        help="Save validation logits, labels, and indices to file.")
    parser.add_argument("--save-test-logits", action="store_true", dest="save_test_logits",
        help="Save test logits, labels, and indices to file.")
    parser.add_argument("--json-precision", type=int, default=5, dest="json_precision",
        help="Number of decimal places to keep for floating point values saved to JSON.")

    return parser.parse_args()

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
    

    stats = focal_calibration_evaluation(net, val_logits, val_labels, test_logits, test_labels,
                                          num_classes=num_classes, device=device,
                                          train_logits=train_logits,
                                          train_labels=train_labels,
                                          links=args.links)
    if args.dirichlet:
        stats["dirichlet_calibrated"] = dirichlet_calibration_evaluation(
            val_logits, val_labels, test_logits, test_labels,
            num_classes=num_classes, device=device,
            train_logits=train_logits,
            train_labels=train_labels,
            cv_folds=args.dirichlet_cv_folds,
            reg_grid=args.dirichlet_reg_grid,
            max_iter=args.dirichlet_max_iter,
            n_jobs=args.dirichlet_n_jobs,
            seed=args.seed,
            smoke_test=args.smoke_test)
    # stats = round_floats(stats)

    def save_logits_labels_indices_npz(filename, logits, labels, indices=None):
        import numpy as np
        logits = logits.cpu().numpy() if hasattr(logits, 'cpu') else logits
        labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
        if indices is None:
            indices = np.arange(len(labels))
        np.savez(filename, logits=logits, labels=labels, indices=indices)

    # Ensure the save directory exists
    if not os.path.exists(args.save_eval_loc):
        os.makedirs(args.save_eval_loc)

    model_stem = os.path.splitext(saved_model_name)[0]
    model_prefix = args.model_name + "_"
    if model_stem.startswith(model_prefix):
        saved_stats_name = model_stem[len(model_prefix):]
    else:
        saved_stats_name = model_stem
    save_stats_path = os.path.join(save_eval_loc, saved_stats_name)
    save_stats_json_path = save_stats_path + ".json"
    rounded_stats = round_floats_for_json(stats, args.json_precision)
    save_evaluation_json(rounded_stats, save_stats_json_path)

    if args.save_train_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'train_logits_labels_indices.npz'), train_logits, train_labels)
    if args.save_val_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'val_logits_labels_indices.npz'), val_logits, val_labels)
    if args.save_test_logits:
        save_logits_labels_indices_npz(os.path.join(args.save_eval_loc, 'test_logits_labels_indices.npz'), test_logits, test_labels)

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
