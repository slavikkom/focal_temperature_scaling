'''
Script for training models.
'''

from torch import optim
import torch
import torch.utils.data
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import GradScaler
import argparse
import torch.backends.cudnn as cudnn
import random
import json
import sys
import os
import glob
import re

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet
import Data.medmnist_loader as get_medmnist_data_loader
import medmnist

# Import network models
from Net.resnet import resnet18, resnet50, resnet110
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import loss functions
#from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
#from Losses.loss import mmce, mmce_weighted
#from Losses.loss import brier_score
from Losses.loss import *

# Import train and validation utilities
from train_utils import train_single_epoch, test_single_epoch

# Import validation metrics
from Metrics.metrics import test_classification_net

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

dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
}

for name in medmnist_datasets:
    dataset_num_classes[name] = len(medmnist.INFO[name]['label'])

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet,
}

for name in medmnist_datasets:
    dataset_loader[name] = get_medmnist_data_loader

models = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def loss_function_save_name(loss_function,
                            scheduled=False,
                            gamma=1.0,
                            gamma1=1.0,
                            gamma2=1.0,
                            gamma3=1.0,
                            lamda=1.0,
                            beta=1.0,
                            seed=0):
    res_dict = {
        'cross_entropy': 'cross_entropy',
        'focal_loss': 'focal_loss_gamma_' + str(gamma),
        'focal_loss_adaptive': 'focal_loss_adaptive_gamma_' + str(gamma),
        'mmce': 'mmce_lamda_' + str(lamda),
        'mmce_weighted': 'mmce_weighted_lamda_' + str(lamda),
        'brier_score': 'brier_score',
        'adafocal': 'adafocal',
        # new losses
        'linear':             'linear_beta_' + str(gamma),               # “β” is passed via --gamma 
        'exp_p':              'exp_p_alpha_' + str(gamma),               # α via --gamma
        'exp_1mp':            'exp_1mp_alpha_' + str(gamma),             # α via --gamma
        'one_minus_power':    'one_minus_power_beta_' + str(gamma),      # β via --gamma
        'generalized_focal':  'generalized_focal_beta_' + str(beta)    # β via --gamma2, γ via --gamma3
                            + '_gamma_' + str(gamma),
        'log_power':          'log_power_kappa_' + str(gamma),     
        # random loss
        'random_loss':             'random_loss_seed_' + str(seed)
    }
    if (loss_function == 'focal_loss' and scheduled == True):
        res_str = 'focal_loss_scheduled_gamma_' + str(gamma1) + '_' + str(gamma2) + '_' + str(gamma3)
    else:
        res_str = res_dict[loss_function]
    return res_str


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = '../Data/datasets/'
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 1.0
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = './'
    model_name = None
    saved_model_name = "resnet50_cross_entropy_350.model"
    load_loc = './'
    model = "resnet50"
    epoch = 350
    first_milestone = 150 #Milestone for change in lr
    second_milestone = 250 #Milestone for change in lr
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=False)
    parser.add_argument("--load", action="store_true", dest="load",
                        help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("--beta", type=float, default=1.0,
        dest="beta", help="Generalized focal loss beta parameter")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=False)
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--lamda", type=float, default=lamda,
                        dest="lamda", help="Regularization factor")
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma-schedule-step1", type=int, default=gamma_schedule_step1,
                        dest="gamma_schedule_step1", help="1st step for gamma schedule")
    parser.add_argument("--gamma-schedule-step2", type=int, default=gamma_schedule_step2,
                        dest="gamma_schedule_step2", help="2nd step for gamma schedule")

    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name",
                        help='name of the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")

    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run a lightweight smoke test")
    parser.set_defaults(smoke_test=False)

    parser.add_argument("--amp", action="store_true", dest="use_amp",
                    help="Enable mixed precision training")
    parser.set_defaults(use_amp=False)

    parser.add_argument("--seed", type=int, default=1,
        dest="seed", help="random seed for reproducibility")

    return parser.parse_args()


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

    args = parseArgs()
    set_seed(args.seed)

    if args.smoke_test:
        print("Running in smoke test mode...")
        args.epoch = 1  # Only 1 epoch
        # args.gpu = False
        args.save_interval = 1
        args.train_batch_size = 16  # Small batch size
        args.test_batch_size = 16  # Small batch size
        # args.dataset_root = './data/smoke_test/'  # Use a small dataset

    generalized_focal_beta = 1.0 if args.beta is None else args.beta

    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    scaler = GradScaler() if (cuda and args.use_amp) else None
    if scaler is not None:
        use_amp = True
        print("Using mixed precision training with GradScaler.")
    else:
        use_amp = False

    num_classes = dataset_num_classes[args.dataset]
    num_channels = 3
    if args.dataset in medmnist_datasets:
        num_channels = medmnist.INFO[args.dataset]['n_channels']

    # Choosing the model to train
    net = models[args.model](num_classes=num_classes, in_channels=num_channels)

    # Setting model name
    if args.model_name is None:
        args.model_name = args.model

    # Ensure the save directory exists upfront for checkpoints and logs.
    os.makedirs(args.save_loc, exist_ok=True)

    if args.gpu is True:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch
    best_save_after_epoch = max(1, min(50 if args.dataset == "tiny_imagenet" else 250, num_epochs))
    periodic_save_after_epoch = max(1, min(50 if args.dataset == "tiny_imagenet" else 100, num_epochs))

    run_prefix = os.path.join(
        args.save_loc,
        args.model_name + '_' +
        loss_function_save_name(
            args.loss_function,
            args.gamma_schedule,
            args.gamma,
            args.gamma,
            args.gamma2,
            args.gamma3,
            args.lamda,
            args.beta,
            args.seed
        )
    )
    if args.load:
        # net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
        # start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_')+1:args.saved_model_name.rfind('.model')])
        # print("load model. start epoch: ", start_epoch)

        # If saved_model_name is the default value, look for the latest saved model in the directory
        if args.saved_model_name == "resnet50_cross_entropy_350.model":
            # Define the pattern for the model name
            # TODO: NB! this model loading of the latest saved model found does not take care of the gamma schedule i.e. gamma_schedule=0
            model_loss_str = args.model_name + '_' + \
                             loss_function_save_name(args.loss_function, args.gamma_schedule, args.gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.beta, args.seed)
            print("string to match: ", model_loss_str)
            model_pattern = re.compile(rf"{model_loss_str}.*_(\d+)\.model$")
            # Search for all model files in the save location
            model_files = glob.glob(os.path.join(args.save_loc, "*.model"))
            # Filter files that match the pattern
            matching_files = [f for f in model_files if model_pattern.search(os.path.basename(f))]
            print(matching_files) 
            if matching_files:
                # Extract the epoch number from each matching file and find the latest one
                def extract_epoch(file_name):
                    match = model_pattern.search(os.path.basename(file_name))
                    return int(match.group(1)) if match else -1
                
                # Sort matching files by epoch number
                matching_files.sort(key=extract_epoch, reverse=True)
                args.saved_model_name = os.path.basename(matching_files[0])  # Use the latest model
                print(f"Latest model found: {args.saved_model_name}")
            else:
                raise FileNotFoundError(f"No saved models matching the pattern found in {args.save_loc}.")
    
        # Load the model from the specified location
        net.load_state_dict(torch.load(os.path.join(args.save_loc, args.saved_model_name)))
        start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_')+1:args.saved_model_name.rfind('.model')])
        print(f"Resuming training from epoch {start_epoch}.")

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)


    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
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
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=args.seed,
            pin_memory=args.gpu,
            smoke_test=args.smoke_test,
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}
    val_set_err = {}

    for epoch in range(0, start_epoch):
        scheduler.step()

    best_val_acc = 0
    for epoch in range(start_epoch, num_epochs):
        if (args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
            if (epoch < args.gamma_schedule_step1):
                gamma = args.gamma
            elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                gamma = args.gamma2
            else:
                gamma = args.gamma3
        else:
            gamma = args.gamma

        train_loss = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_function=args.loss_function,
                                        gamma=gamma,
                                        lamda=args.lamda,
                                        loss_mean=args.loss_mean,
                                        scaler=scaler,
                                        beta=generalized_focal_beta,
                                        seed=args.seed,)
        scheduler.step()
        val_loss = test_single_epoch(epoch,
                                     net,
                                     val_loader,
                                     device,
                                     loss_function=args.loss_function,
                                     gamma=gamma,
                                     lamda=args.lamda,
                                     beta=generalized_focal_beta,
                                     seed=args.seed)
        test_loss = test_single_epoch(epoch,
                                      net,
                                      test_loader,
                                      device,
                                      loss_function=args.loss_function,
                                      gamma=gamma,
                                      lamda=args.lamda,
                                      use_amp=use_amp,
                                      beta=generalized_focal_beta,
                                      seed=args.seed)
        _, val_acc, _, _, _ = test_classification_net(net, val_loader, device, use_amp=use_amp)
        _, test_acc, _, _, _ = test_classification_net(net, test_loader, device, use_amp=use_amp)

        print(f"===> validation accuracy: {val_acc:.4f}")
        print(f"===> test accuracy: {test_acc:.4f}")

        training_set_loss[epoch] = train_loss
        val_set_loss[epoch] = val_loss
        test_set_loss[epoch] = test_loss
        val_set_err[epoch] = 1 - val_acc


        if val_acc > best_val_acc and (epoch + 1) >= best_save_after_epoch:
            best_val_acc = val_acc
            print('New best error: %.4f' % (1 - best_val_acc))
            save_name = args.save_loc + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.beta, args.seed) + \
                        '_best_' + \
                        str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)

        if ((((epoch + 1) % args.save_interval == 0) and ((epoch + 1) >= periodic_save_after_epoch))
                or args.smoke_test
                or ((epoch + 1) == num_epochs)):
            save_name = args.save_loc + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.beta, args.seed) + \
                        '_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)


    with open(run_prefix + '_train_loss.json', 'w') as f:
        json.dump(training_set_loss, f)

    with open(run_prefix + '_val_loss.json', 'w') as fv:
        json.dump(val_set_loss, fv)

    with open(run_prefix + '_test_loss.json', 'w') as ft:
        json.dump(test_set_loss, ft)

    with open(run_prefix + '_val_error.json', 'w') as ft:
        json.dump(val_set_err, ft)
