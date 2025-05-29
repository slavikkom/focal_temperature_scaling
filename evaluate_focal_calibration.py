import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from temperature_scaling import ModelWithTemperature
from Metrics.metrics import AdaptiveECELoss
from temperature_scaling import multi_focal_link

gamma_link_list = [0.25, 0.5, 0.75, 1, 2, 3, 5, 100] # -1, -0.75, -0.5,-0.25, 0, 

def multi_acc(y_pred, y_test):
    y_pred = torch.argmax(y_pred, dim = 1)    
    
    correct_pred = (y_pred == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = acc.cpu().numpy() * 100
    
    return acc


def evaluate(y_true, probs, num_classes=10):
    
    epoch_loss = {}
        
    mse_loss = torch.nn.MSELoss()

    log_loss_criterion = nn.NLLLoss()
        
    calibration_criterion = AdaptiveECELoss(return_stats=True)
        
    epoch_loss['CE'] = log_loss_criterion(torch.log(probs), y_true.long()).item()
    epoch_loss['ECE'] = calibration_criterion(probs, y_true.long())
        
    epoch_loss['Brier'] = mse_loss(probs, torch.nn.functional.one_hot(y_true.long(), num_classes=num_classes).float()).item()
    epoch_loss['ACC'] = multi_acc(probs, y_true)

    return epoch_loss
    
def get_probs(logits, T=1, a=1, softmax=False):
    if softmax:
        probs = torch.nn.Softmax(dim=1)(logits / T)
    else:
        probs = multi_focal_link(logits / T, a)
    
    return probs

def focal_calibration_evaluation(net, valLoader, val_logits, val_labels, test_logits, test_labels, num_classes=10, device='cuda', train_logits=None, train_labels=None, trainLoader=None):
    calib_states = ('calibrated', 'uncalibrated')
    datasets = ['val', 'test']
    if train_logits is not None:
        datasets.insert(0, 'train')   # now ['train','val','test']

    evaluation_metrics = {
        phase: { state: {} for state in calib_states }
        for phase in datasets
    }


    gamma_dict = {}
    
    for data_set in datasets:
        for T_metric in ['ce', 'ece']:
            evaluation_metrics[data_set]['calibrated'][T_metric] = {}

    for g_counter, gamma in enumerate(gamma_link_list):

        scaled_model = ModelWithTemperature(net, gamma=gamma, softmax=int(gamma>50))
        scaled_model.set_temperature(valLoader, cross_validate='ece', device=device)
        T_opt_ce = scaled_model.get_temperature(metric='ce')
        T_opt_ece = scaled_model.get_temperature(metric='ece')
        
        gamma_dict[str(round(gamma, 2)) + "_CE"] = scaled_model.nll_vals
        gamma_dict[str(round(gamma, 2)) + "_ECE"] = scaled_model.ece_vals

        ### First uncalibrated
        val_pred = get_probs(val_logits, T=1, a=gamma, softmax=int(gamma>50))
        evaluation_metrics['val']['uncalibrated'][str(round(gamma, 2))] = evaluate(val_labels, val_pred, num_classes=num_classes)
        test_pred = get_probs(test_logits, T=1, a=gamma, softmax=int(gamma>50))
        evaluation_metrics['test']['uncalibrated'][str(round(gamma, 2))] = evaluate(test_labels, test_pred, num_classes=num_classes)
            
        if train_labels is not None:
            train_pred = get_probs(train_logits, T=1, a=gamma, softmax=int(gamma>50))
            evaluation_metrics['train']['uncalibrated'][str(round(gamma, 2))] = evaluate(train_labels, train_pred, num_classes=num_classes)
        
            scaled_model = ModelWithTemperature(net, gamma=gamma, softmax=int(gamma>50))
            scaled_model.set_temperature(trainLoader, cross_validate='ece', device=device)

            for T_metric in ['ce', 'ece']:
                train_T_opt = scaled_model.get_temperature(metric=T_metric)

                train_pred = get_probs(train_logits, T=train_T_opt, a=gamma, softmax=int(gamma>50))
                evaluation_metrics['train']['calibrated'][T_metric][str(round(gamma, 2))] = evaluate(train_labels, train_pred, num_classes=num_classes)

        for T_metric in ['ce', 'ece']:
            T_opt = T_opt_ce if T_metric == 'ce' else T_opt_ece
            val_pred = get_probs(val_logits, T=T_opt, a=gamma, softmax=int(gamma>50))
            evaluation_metrics['val']['calibrated'][T_metric][str(round(gamma, 2))] = evaluate(val_labels, val_pred, num_classes=num_classes)
            
            test_pred = get_probs(test_logits, T=T_opt, a=gamma, softmax=int(gamma>50))
            evaluation_metrics['test']['calibrated'][T_metric][str(round(gamma, 2))] = evaluate(test_labels, test_pred, num_classes=num_classes)
        
        for T_metric in ['ce', 'ece']:
            gamma_dict[str(round(gamma, 2)) + " T_opt" + " " + T_metric] = T_opt_ce if T_metric == 'ce' else T_opt_ece
    
    evaluation_metrics['T_dict'] = gamma_dict
    return evaluation_metrics