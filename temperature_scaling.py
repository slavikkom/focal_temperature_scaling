'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from Metrics.metrics import ECELoss, AdaptiveECELoss
from new_links import linear_invlink, multi_link

def focal_link(x, a=2): # a - gamma of focal loss
    nominator = a*torch.log(torch.exp(x)+1)+torch.exp(x)
    denominator = nominator + torch.exp(-a*x)*(a*torch.exp(x)*torch.log(torch.exp(-x)+1) + 1)
    p = nominator / denominator
    return nominator

"""
def focal_derivative(p, gamma=2):
    p = torch.clamp(p, min=1e-12, max=1-1e-12)
    return (1 - p)**gamma * (gamma * torch.log(p) / (1 - p) - 1 / p)
"""
def focal_derivative(p, gamma=2):
    eps = 1e-12
    p = p.clamp(min=eps, max=1.0 - eps)
    one_minus_p = 1.0 - p

    # gamma * log(p) / (1-p)
    term1 = gamma * torch.log(p).div(one_minus_p)
    # 1/p
    term2 = p.reciprocal()
    return (one_minus_p ** gamma) * (term1 - term2)
"""
def focal_map(q, gamma=2):
    inverse_grad = 1 / focal_derivative(q, gamma=gamma)
    p = inverse_grad / torch.sum(inverse_grad, dim=-1, keepdim=True)
    return p
"""
def focal_map(q, gamma=2):
    der = focal_derivative(q, gamma)      # shape = [batch, classes]
    inv  = der.reciprocal()              # same as 1/der, but sometimes clearer to read
    row_sum = inv.sum(dim=-1, keepdim=True)
    p = inv.div(row_sum)                  # normalize so each row sums to 1
    return p

#def check_overflow(tensor_x):
#    return (tensor_x > torch.finfo(tensor_x.dtype).min) & (tensor_x < torch.finfo(tensor_x.dtype).max)
"""
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

    overflowed_rows_indices = torch.nonzero(overflowed_rows_indices, as_tuple=False).squeeze()

    p[overflowed_rows_indices] = 1e-5/nr_classes
    p[overflowed_rows_indices, overflowed_max_indices] = 1-1e-5
    return p
"""
def multi_focal_link(x, a=2):
    # 1) compute q = softmax(x), shape = [batch, n_classes]
    q = F.softmax(x, dim=-1)

    # 2) focal_map → p, shape [batch, n_classes]
    p = focal_map(q, gamma=a)

    # 3) find any non-finite entries (NaN or ±Inf)
    #    Note: torch.isfinite(p) is a [batch, n_classes] boolean
    #    (True for finite entries). We want rows with ANY non-finite.
    mask_rows = (~p.isfinite()).any(dim=1)       # [batch] boolean

    if mask_rows.any():
        # 4) identify those row indices
        rows = mask_rows.nonzero(as_tuple=True)[0]  # 1D LongTensor of size [K]

        # 5) find which class has the max logit for each row
        max_indices = torch.argmax(x, dim=1)         # [batch]
        max_for_rows = max_indices[rows]             # [K]

        # 6a) Set every entry in those rows to a small uniform value
        nr_classes = p.size(1)
        tiny_val = 1e-5            # total “mass” we’ll give to the off‐max terms
        other_val = tiny_val / (nr_classes - 1)

        # Assign “other_val” to every column
        p[rows.unsqueeze(1), torch.arange(nr_classes, device=x.device)] = other_val

        # 6b) At the max‐logit column, give it (1 - tiny_val)
        p[rows, max_for_rows] = 1.0 - tiny_val
        # → Now each row sums to exactly 1
    return p

def apply_link(logits, link='softmax', a=1):
    if link == 'softmax':
        probs = torch.nn.Softmax(dim=1)(logits)
    elif link == 'focal':
        probs = multi_focal_link(logits, a)
        eps = 1e-12
        probs = probs.clamp(min=eps, max=1.0)
    elif link == 'generalized_focal':
        probs = multi_link(logits, link, a[0], a[1])
        eps = 1e-12
        probs = probs.clamp(min=eps, max=1.0)
    else:
        probs = multi_link(logits, link, a)
        eps = 1e-12
        probs = probs.clamp(min=eps, max=1.0)
    return probs


def get_probs_ts_first(logits, T=1.0, a=1.0, link='softmax'):
    return apply_link(logits / T, link=link, a=a)


def get_probs_ts_last(logits, T=1.0, a=1.0, link='softmax'):
    base_probs = apply_link(logits, link=link, a=a)

    eps = torch.finfo(base_probs.dtype).tiny
    log_base_probs = torch.log(base_probs.clamp_min(eps))

    return torch.softmax(log_base_probs / T, dim=1)


def get_probs_with_temperature_order(logits, T=1.0, a=1.0, link='softmax',
                                     posthoc_order='ts_first'):
    if posthoc_order == 'ts_first':
        return get_probs_ts_first(logits, T=T, a=a, link=link)
    if posthoc_order == 'ts_last':
        return get_probs_ts_last(logits, T=T, a=a, link=link)
    raise ValueError(
        "Unknown posthoc_order '{}'. Expected 'ts_first' or 'ts_last'.".format(
            posthoc_order
        )
    )


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True, a=1, link='softmax',
                 posthoc_order='ts_first'):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.temperature_ece = 1.0
        self.temperature_nll = 1.0
        self.log = log
        self.a = a
        self.link = link
        self.posthoc_order = posthoc_order


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
                        logits, labels,
                        cross_validate='ce',
                        device='cuda'):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        self.to(device)
        self.model.eval()

        nll_criterion = nn.NLLLoss().to(device)
        ece_criterion = AdaptiveECELoss().to(device)

        # Calculate NLL and ECE before temperature scaling
        print("Current link parameter is ", self.a)
        probs = get_probs_with_temperature_order(
            logits,
            T=1.0,
            link=self.link,
            a=self.a,
            posthoc_order=self.posthoc_order,
        )
        
        before_temperature_nll = nll_criterion(torch.log(probs), labels.long()).item()
        before_temperature_ece = ece_criterion(probs, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = before_temperature_nll
        ece_val = before_temperature_ece
        
        self.nll_vals = []
        self.ece_vals = []
        
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.05
        
        for i in range(100):
            self.temperature = T
            # self.cuda()
            self.to(device)
            probs = None
            
            probs = get_probs_with_temperature_order(
                logits,
                T=T,
                link=self.link,
                a=self.a,
                posthoc_order=self.posthoc_order,
            )

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
        self.temperature_ece = T_opt_ece
        self.temperature_nll = T_opt_nll

        self.to(device)

        # Calculate NLL and ECE after temperature scaling
        probs = get_probs_with_temperature_order(
            logits,
            T=self.temperature,
            link=self.link,
            a=self.a,
            posthoc_order=self.posthoc_order,
        )

        after_temperature_nll = nll_criterion(torch.log(probs), labels.long()).item()
        after_temperature_ece = ece_criterion(probs, labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % self.temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


    def get_temperature(self, metric=None):
        if metric is None:
            return self.temperature
        elif metric == 'ce':
            return self.temperature_nll
        elif metric == 'ece':
            return self.temperature_ece
