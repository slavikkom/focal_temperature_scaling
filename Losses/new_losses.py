import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-8

def _safe_prob(pt: torch.Tensor) -> torch.Tensor:
    """Clamp probabilities into (eps, 1−eps) to avoid log(0) and
       fractional powers of negatives."""
    return pt.clamp(min=_EPS, max=1.0 - _EPS) 

# --------------------------------------------
# 1. Linear‐decay loss g(p) = 1 − β p
# --------------------------------------------
class LinearDecayLoss(nn.Module):
    def __init__(self, beta: float = 0.5, reduction: str = 'sum'):
        """
        Linear‐decay loss: f(p) = (1 − β p)(−log p) evaluated at p = model’s predicted probability
        for the true class.
        
        Args:
            beta: the β parameter in g(p) = 1 − β p
            reduction: 'sum' or 'mean'
        """
        super().__init__()
        self.beta = beta
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: logits of shape [N, C] or [N, C, H, W]
        # target: integer class labels shape [N] or [N, H, W]
        if input.dim() > 2:
            # reshape spatial dimensions so we treat every spatial location as a separate sample
            n, c, *rest = input.shape  # rest = [H, W] or more
            input = input.view(n, c, -1)           # [N, C, H*W]
            input = input.transpose(1, 2)           # [N, H*W, C]
            input = input.contiguous().view(-1, c)  # [N*H*W, C]
            target = target.view(-1)                # [N*H*W]

        # Compute log‐softmax over classes
        logpt_all = F.log_softmax(input, dim=1)     # [M, C], M = flattened batch
        # Gather log‐probability of the true class
        target = target.view(-1, 1)                  # [M, 1]
        logpt = logpt_all.gather(1, target).view(-1) # [M]
        pt = _safe_prob(logpt.exp())                           # [M]

        # g(pt) = 1 − β p_t
        g = (1.0 - self.beta * pt).clamp(min=0.0)                      # [M]

        # loss_i = g(pt) * (−log(pt)) = (1 − β pt) * (−logpt)
        loss = -g * logpt                             # [M]

        if self.reduction == 'mean':
            return loss.mean()
        else:  # 'sum'
            return loss.sum()


# --------------------------------------------
# 2a. Exponential weight g(p) = exp(−α p)
# --------------------------------------------
class ExpPLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, reduction: str = 'sum'):
        """
        Exponential‐on‐p loss: f(p) = exp(−α p)(−log p)
        """
        super().__init__()
        self.alpha = alpha
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            n, c, *rest = input.shape
            input = input.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
            target = target.view(-1)

        logpt_all = F.log_softmax(input, dim=1)
        target = target.view(-1, 1)
        logpt = logpt_all.gather(1, target).view(-1)
        pt = _safe_prob(logpt.exp()) 

        # g(pt) = exp(−α pt)
        g = torch.exp(-self.alpha * pt)

        loss = -g * logpt

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# --------------------------------------------
# 2b. Exponential weight g(p) = exp(−α (1−p))
# --------------------------------------------
class Exp1mpLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, reduction: str = 'sum'):
        """
        Exponential‐on‐(1−p) loss: f(p) = exp(−α (1−p))(−log p)
        """
        super().__init__()
        self.alpha = alpha
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            n, c, *rest = input.shape
            input = input.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
            target = target.view(-1)

        logpt_all = F.log_softmax(input, dim=1)
        target = target.view(-1, 1)
        logpt = logpt_all.gather(1, target).view(-1)
        pt = _safe_prob(logpt.exp())  

        # g(pt) = exp(−α (1 − pt))
        g = torch.exp(-self.alpha * (1.0 - pt))

        loss = -g * logpt

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# --------------------------------------------
# 3. “One minus power” g(p) = 1 − p^β
# --------------------------------------------
class OneMinusPowerLoss(nn.Module):
    def __init__(self, beta: float = 2.0, reduction: str = 'sum'):
        """
        One‐minus‐power loss: f(p) = (1 − p^β)(−log p)
        """
        super().__init__()
        self.beta = beta
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            n, c, *rest = input.shape
            input = input.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
            target = target.view(-1)

        logpt_all = F.log_softmax(input, dim=1)
        target = target.view(-1, 1)
        logpt = logpt_all.gather(1, target).view(-1)
        pt = _safe_prob(logpt.exp())                               # --- STABILITY FIX ---
        base = (1.0 - pt.pow(self.beta)).clamp(min=0.0)            # --- STABILITY FIX ---
        g = base      

        loss = -g * logpt  # = (1 − pt^β)*(−logpt)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# --------------------------------------------
# 4. Generalised focal g(p) = (1 − p^β)^γ
# --------------------------------------------
class GeneralizedFocalLoss(nn.Module):
    def __init__(self, beta: float = 2.0, gamma: float = 2.0, reduction: str = 'sum'):
        """
        Generalized‐focal loss: f(p) = (1 − p^β)^γ (−log p)
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            n, c, *rest = input.shape
            input = input.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
            target = target.view(-1)

        logpt_all = F.log_softmax(input, dim=1)
        target = target.view(-1, 1)
        logpt = logpt_all.gather(1, target).view(-1)

        pt = _safe_prob(logpt.exp())                               # --- STABILITY FIX ---
        base = (1.0 - pt.pow(self.beta)).clamp(min=0.0)            # --- STABILITY FIX ---
        g = base.pow(self.gamma)                                   # safe even for γ<1
        loss = -g * logpt

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# --------------------------------------------
# 5. Log‐power weight g(p) = (−log p)^κ
# --------------------------------------------
class LogPowerLoss(nn.Module):
    def __init__(self, kappa: float = 2.0, reduction: str = 'sum'):
        """
        Log‐power loss: f(p) = (−log p)^κ (−log p) = (−log p)^(κ+1)
        """
        super().__init__()
        self.kappa = kappa
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            n, c, *rest = input.shape
            input = input.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
            target = target.view(-1)

        logpt_all = F.log_softmax(input, dim=1)
        target = target.view(-1, 1)
        logpt = logpt_all.gather(1, target).view(-1)
        pt = _safe_prob(logpt.exp())                               # --- STABILITY FIX ---
        L  = -torch.log(pt) 
        g = L.pow(self.kappa)

        # loss = g * L = (−log p)^(κ+1)
        loss = g * L

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# --------------------------------------------
# Example of how to instantiate and use:
# --------------------------------------------
#
# # 1) For a standard focal loss (γ = 2):
# focal_loss = FocalLoss(gamma=2, size_average=False)  
#   → This is the “ordinary focal” you already have.
#
# # 2) For linear‐decay (β = 0.5), summed over batch:
# lin_loss = LinearDecayLoss(beta=0.5, reduction='sum')
#
# # 3) For exponential‐on‐p (α = 2.0), averaged over batch:
# exp_p_loss = ExpPLoss(alpha=2.0, reduction='mean')
#
# # 4) For exponential‐on‐(1−p) (α = 2.0), summed:
# exp_1mp_loss = Exp1mpLoss(alpha=2.0, reduction='sum')
#
# # 5) For one‐minus‐power (β = 2.0), averaged:
# omp_loss = OneMinusPowerLoss(beta=2.0, reduction='mean')
#
# # 6) For generalised‐focal (β = 2.0, γ = 2.0), summed:
# gen_focal_loss = GeneralizedFocalLoss(beta=2.0, gamma=2.0, reduction='sum')
#
# # 7) For log‐power (κ = 2.0), averaged:
# log_pow_loss = LogPowerLoss(kappa=2.0, reduction='mean')
#
# Then in your training loop, simply call:
#     logits = model(inputs)    # shape [batch, classes]
#     loss = chosen_loss_fn(logits, targets)
#     loss.backward()
#     optimizer.step()
#
# These loss classes mirror the “g(p)(−log p)” form for each link type, optimized in PyTorch:
#   • We apply F.log_softmax → gather → compute pt
#   • Define g(pt) according to the chosen form
#   • Compute loss_i = g(pt) * (−log pt)
#   • Sum or average across the batch per `reduction`.
