'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from Metrics.metrics import ECELoss, AdaptiveECELoss

def focal_link(x, a=2): # a - gamma of focal loss
    nominator = a*torch.log(torch.exp(x)+1)+torch.exp(x)
    denominator = nominator + torch.exp(-a*x)*(a*torch.exp(x)*torch.log(torch.exp(-x)+1) + 1)
    p = nominator / denominator
    return nominator

def focal_derivative(p, gamma=2):
    p = torch.clamp(p, min=1e-12, max=1-1e-12)
    return (1 - p)**gamma * (gamma * torch.log(p) / (1 - p) - 1 / p)

def focal_map(q, gamma=2):
    inverse_grad = 1 / focal_derivative(q, gamma=gamma)
    p = inverse_grad / torch.sum(inverse_grad, dim=-1, keepdim=True)
    return p

def check_overflow(tensor_x):
    return (tensor_x > torch.finfo(tensor_x.dtype).min) & (tensor_x < torch.finfo(tensor_x.dtype).max)

def multi_focal_link(x, a=2):
    q = F.softmax(x, dim=-1)
    p = focal_map(q, gamma=a)
    nr_classes = p.shape[1]
    nr_instances = p.shape[0]
    # Identify rows with overflow
    overflowed_rows = ~check_overflow(p)
    # Select overflowed rows
    overflowed_rows_indices = overflowed_rows.any(dim=1)

    overflowed_max_indices = torch.argmax(x, dim=1).unsqueeze(1)

    overflowed_rows_indices = torch.nonzero(overflowed_rows_indices).squeeze()

    p[overflowed_rows_indices] = 1e-5/nr_classes
    p[overflowed_rows_indices, overflowed_max_indices] = 1-1e-5
    return p


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True, gamma=1, softmax=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log
        self.gamma = gamma
        self.softmax = softmax


    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature


    def set_temperature(self,
                        valid_loader,
                        cross_validate='ce'):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        self.cuda()
        self.model.eval()
        nll_criterion = nn.NLLLoss().cuda()
        ece_criterion = AdaptiveECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        print("Current gamma is ", self.gamma)
        probs = None
        if self.new_method:
            probs = torch.nn.Softmax(dim=1)(torch.log(multi_focal_link(logits, self.gamma)) / 1)
        else:
            if self.softmax:
                probs = torch.nn.Softmax(dim=1)(logits)
            else:
                probs = multi_focal_link(logits, self.gamma)
        
        before_temperature_nll = nll_criterion(torch.log(probs), labels.long()).item()
        before_temperature_ece = ece_criterion(probs, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        
        self.nll_vals = []
        self.ece_vals = []
        
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.05
        
        for i in range(100):
            self.temperature = T
            self.cuda()
            probs = None
            if self.new_method:
                probs = torch.nn.Softmax(dim=1)(torch.log(multi_focal_link(logits, self.gamma)) / T)
            else:
                if self.softmax:
                    probs = torch.nn.Softmax(dim=1)(logits / T)
                else:
                    probs = multi_focal_link(logits / T, self.gamma)

            after_temperature_nll = nll_criterion(torch.log(probs), labels.long()).item()
            after_temperature_ece = ece_criterion(probs, labels).item()
            
            self.nll_vals.append(after_temperature_nll)
            self.ece_vals.append(after_temperature_ece)
            
            if (nll_val > after_temperature_nll) and not (np.isnan(after_temperature_nll)):
                T_opt_nll = T
                nll_val = after_temperature_nll

            if (ece_val > after_temperature_ece) and not (np.isnan(after_temperature_ece)):
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.05

        if cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.cuda()

        # Calculate NLL and ECE after temperature scaling
        probs = None
        if self.softmax:
            probs = torch.nn.Softmax(dim=1)(logits / T)
        else:
            probs = multi_focal_link(logits / T, self.gamma)

        after_temperature_nll = nll_criterion(torch.log(probs), labels.long()).item()
        after_temperature_ece = ece_criterion(probs, labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % self.temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


    def get_temperature(self):
        return self.temperature