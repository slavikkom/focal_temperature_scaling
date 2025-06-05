'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
'''

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.adafocal import AdaFocal
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore
from Losses.new_losses import LinearDecayLoss, ExpPLoss, Exp1mpLoss, OneMinusPowerLoss, GeneralizedFocalLoss, LogPowerLoss


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)


def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)


def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)


def adafocal(logits, targets, **kwargs):
    model = AdaFocal(device=kwargs['device'])
    return model(logits, targets)

def linear_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for LinearDecayLoss:
      f(p) = (1 − β p)(−log p), summed over the batch.
    Expects:
      kwargs['gamma']  → β
      kwargs['device'] → torch device
    """
    beta = kwargs['gamma']
    device = kwargs['device']
    return LinearDecayLoss(beta=beta, reduction='sum').to(device)(logits, targets)


def exp_p_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for ExpPLoss:
      f(p) = exp(−α p)(−log p), summed over the batch.
    Expects:
      kwargs['gamma']  → α
      kwargs['device'] → torch device
    """
    alpha = kwargs['gamma']
    device = kwargs['device']
    return ExpPLoss(alpha=alpha, reduction='sum').to(device)(logits, targets)


def exp_1mp_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for Exp1mpLoss:
      f(p) = exp(−α (1 − p))(−log p), summed over the batch.
    Expects:
      kwargs['gamma']  → α
      kwargs['device'] → torch device
    """
    alpha = kwargs['gamma']
    device = kwargs['device']
    return Exp1mpLoss(alpha=alpha, reduction='sum').to(device)(logits, targets)


def one_minus_power_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for OneMinusPowerLoss:
      f(p) = (1 − p^β)(−log p), summed over the batch.
    Expects:
      kwargs['gamma']  → β
      kwargs['device'] → torch device
    """
    beta = kwargs['gamma']
    device = kwargs['device']
    return OneMinusPowerLoss(beta=beta, reduction='sum').to(device)(logits, targets)


def generalized_focal_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for GeneralizedFocalLoss:
      f(p) = (1 − p^β)^γ (−log p), summed over the batch.
    Expects:
      kwargs['gamma2'] → β
      kwargs['gamma3'] → γ
      kwargs['device'] → torch device
    """
    beta = kwargs['beta']
    gamma_ = kwargs['gamma']
    device = kwargs['device']
    return GeneralizedFocalLoss(beta=beta, gamma=gamma_, reduction='sum') \
               .to(device)(logits, targets)


def log_power_loss_fn(logits, targets, **kwargs):
    """
    Wrapper for LogPowerLoss:
      f(p) = (−log p)^κ (−log p) = (−log p)^(κ+1), summed over the batch.
    Expects:
      kwargs['gamma']  → κ
      kwargs['device'] → torch device
    """
    kappa = kwargs['gamma']
    device = kwargs['device']
    return LogPowerLoss(kappa=kappa, reduction='sum').to(device)(logits, targets)
