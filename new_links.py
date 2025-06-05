import torch
import torch.nn.functional as F

# --------------------------------------------
# 1. Linear‐decay weight g(p) = 1 − β p
# --------------------------------------------
def linear_invlink(q: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """
    PyTorch version of linear_invlink:
    invlink(p) = 1 / f'(p), then row‐normalize so that sum over classes = 1.
    """
    # q: [batch, n_classes]
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    # g(p) = 1 − β p
    g = 1.0 - beta * p          # same shape as p
    # g'(p) = derivative of (1 − β p) = −β
    g_prime = torch.full_like(p, -beta)
    # f'(p) = d/dp [ (1−βp)(−log p) ] = −g'(p)⋅log p − g(p)/p
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime         # elementwise inverse
    # Normalize across classes so each row sums to 1
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# 2a. Exponential weight g(p) = exp(−α p)
# --------------------------------------------
def exp_p_invlink(q: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    PyTorch version of exp_p_invlink:
    g(p) = exp(−α p)
    """
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    g = torch.exp(-alpha * p)
    # g'(p) = derivative of exp(−α p) = −α ⋅ exp(−α p) = −α g
    g_prime = -alpha * g
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# 2b. Exponential weight g(p) = exp(−α (1−p))
# --------------------------------------------
def exp_1mp_invlink(q: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    PyTorch version of exp_1mp_invlink:
    g(p) = exp(−α (1 − p))
    """
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    g = torch.exp(-alpha * (1.0 - p))
    # g'(p) = derivative of exp(−α (1−p)) = +α exp(−α (1−p)) = α g
    g_prime = alpha * g
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# 3. “One minus power” g(p) = 1 − p^β
# --------------------------------------------
def one_minus_power_invlink(q: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """
    PyTorch version of one_minus_power_invlink:
    g(p) = 1 − p^β
    """
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    g = 1.0 - p.pow(beta)
    # g'(p) = derivative of (1 − p^β) = −β p^(β−1)
    g_prime = -beta * p.pow(beta - 1.0)
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# 4. Generalised focal g(p) = (1 − p^β)^γ
# --------------------------------------------
def generalized_focal_invlink(
    q: torch.Tensor, beta: float = 2.0, gamma: float = 2.0
) -> torch.Tensor:
    """
    PyTorch version of generalized_focal_invlink:
    g(p) = (1 − p^β)^γ
    """
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    one_minus_pb = (1.0 - p.pow(beta))
    g = one_minus_pb.pow(gamma)
    # g'(p) = −γ β p^(β−1) ⋅ (1 − p^β)^(γ−1)
    g_prime = -gamma * beta * p.pow(beta - 1.0) * one_minus_pb.pow(gamma - 1.0)
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# 5. Log‐power weight g(p) = (−log p)^κ
# --------------------------------------------
def log_power_invlink(q: torch.Tensor, kappa: float = 2.0) -> torch.Tensor:
    """
    PyTorch version of log_power_invlink:
    g(p) = (−log p)^κ
    """
    eps = 1e-12
    p = q.clamp(min=eps, max=1.0 - eps)
    L = -torch.log(p)                     # positive tensor
    g = L.pow(kappa)
    # g'(p) = derivative of (L^κ) = κ L^(κ−1) ⋅ (−1/p) = −κ L^(κ−1) / p
    g_prime = -kappa * L.pow(kappa - 1.0) / p
    f_prime = -g_prime * torch.log(p) - g / p
    inv = 1.0 / f_prime
    row_sum = inv.sum(dim=-1, keepdim=True) + eps
    return inv / row_sum


# --------------------------------------------
# Registry of invlink functions
# --------------------------------------------
INVLINK_FUNCS = {
    'focal_linear':           linear_invlink,
    'exp_p':            exp_p_invlink,
    'exp_1mp':          exp_1mp_invlink,
    'one_minus_power':  one_minus_power_invlink,
    'generalized_focal':generalized_focal_invlink,
    'log_power':        log_power_invlink,
}


# --------------------------------------------
# Generalized multi‐link wrapper
# --------------------------------------------
def multi_link(
    logits: torch.Tensor,
    link: str = 'softmax',
    a: float = 2.0,
    beta: float = 2.0,
    gamma: float = 2.0,
    kappa: float = 2.0
) -> torch.Tensor:
    """
    Apply a chosen link function to logits:
      - 'softmax'           → standard Softmax
      - 'linear'            → linear_invlink  (parameter beta)
      - 'exp_p'             → exp_p_invlink   (parameter alpha=a)
      - 'exp_1mp'           → exp_1mp_invlink (parameter alpha=a)
      - 'one_minus_power'   → one_minus_power_invlink (parameter beta)
      - 'generalized_focal' → generalized_focal_invlink (parameters beta,gamma)
      - 'log_power'         → log_power_invlink (parameter kappa)

    Then patch any rows with NaN/Inf by assigning nearly‐one‐hot on the max‐logit class.
    Returns a probability tensor of shape [batch, n_classes], row‐normalized.
    """
    # 1) Compute base probabilities q via softmax
    q = F.softmax(logits, dim=-1)
    eps = 1e-12

    # 2) Choose and apply the appropriate invlink function
    if link == 'softmax':
        p = q
    else:
        # Map link name to the correct invlink function
        if link not in INVLINK_FUNCS:
            raise ValueError(f"Unknown link '{link}'. Available: {list(INVLINK_FUNCS.keys())}")
        invlink_fn = INVLINK_FUNCS[link]

        # Pass the right parameters:
        if link == 'focal_linear':
            p = invlink_fn(q, beta=a)
        elif link in ('exp_p', 'exp_1mp'):
            p = invlink_fn(q, alpha=a)
        elif link == 'one_minus_power':
            p = invlink_fn(q, beta=a)
        elif link == 'generalized_focal':
            p = invlink_fn(q, beta=a, gamma=beta)
        elif link == 'log_power':
            p = invlink_fn(q, kappa=a)
        else:
            # Fallback: single‐parameter versions (use a as gamma)
            p = invlink_fn(q, gamma=a)

    # 3) Clamp into (eps, 1−eps) to avoid numerical problems
    p = p.clamp(min=eps, max=1.0 - eps)

    # 4) Identify any rows with non‐finite entries (should be rare)
    mask_rows = (~p.isfinite()).any(dim=1)     # [batch] boolean mask
    if mask_rows.any():
        rows = mask_rows.nonzero(as_tuple=True)[0]      # shape [K]
        # Find argmax of logits per row
        max_indices = torch.argmax(logits, dim=1)       # [batch]
        max_for_rows = max_indices[rows]                # [K]

        batch_size, nr_classes = p.shape
        tiny_val = 1e-5
        other_val = tiny_val / (nr_classes - 1)

        # 4a) Set all columns in overflowed rows to other_val
        # We can write it as broadcasting: for each row in `rows`,
        # fill entire row with other_val
        p[rows.unsqueeze(1), torch.arange(nr_classes, device=logits.device)] = other_val

        # 4b) At the max‐logit column, set to (1 − tiny_val)
        p[rows, max_for_rows] = 1.0 - tiny_val
        # Now each overflowed row sums exactly to 1

    return p


# --------------------------------------------
# Example usage in a training loop:
# --------------------------------------------
# logits: output from your network, shape [batch_size, num_classes]
# Choose link='generalized_focal', beta=2.0, gamma=2.0, for instance:
# probs = multi_link(logits, link='generalized_focal', beta=2.0, gamma=2.0)
#
# For plain softmax:
# probs = multi_link(logits, link='softmax')
#
# For exponential on p: (alpha = a)
# probs = multi_link(logits, link='exp_p', a=3.0)
#
# And so on for each variant.
