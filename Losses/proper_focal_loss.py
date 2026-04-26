# ═════════════════════════════════════════════════════════════════════════════
 #  ProperFocalLoss with EXACT IFT backward
 # ─────────────────────────────────────────────────────────────────────────────
 #  Forward :  same as STE version,  p = phi^{-1}(q),  loss = L_FL(p, y)
 #  Backward:  uses  d phi^{-1} / d q  =  (d phi / d p)^{-1}  restricted to the
 #             simplex tangent T = { v in R^K : sum(v) = 0 }  (IFT).
 #
 #  Why tangent-restricted: J = d phi / d p is K x K with column-sums zero
 #  (its image is T); on T it is invertible.  The softmax backward that
 #  consumes grad_q is blind to span(1), so we solve entirely in T:
 #
 #      x   in T     s.t.   J^T x  =  grad_p    on T
 #
 #  implemented by picking an orthonormal basis B of T (K x (K-1)) and solving
 #  the reduced  (K-1)x(K-1)  linear system  (B^T J^T B) y = B^T grad_p,
 #  then lifting back as grad_q = B y.
 # ═════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F

FL_GAMMA = 2.0
FL_EPS = 1e-8
NUM_CLASSES = 3

# ---- Focal link phi and phi^{-1} (scalar-nested bisection) -----------------
def focal_weight(p, gamma=FL_GAMMA, eps=FL_EPS):
    """w(p) = -1/ell'(p)  with  ell(p) = -(1-p)^gamma log p."""
    p = p.clamp(eps, 1 - eps)
    omp = 1.0 - p
    ell_prime = gamma * omp.pow(gamma - 1) * torch.log(p) - omp.pow(gamma) / p
    return -1.0 / ell_prime.clamp(max=-1e-30)

def focal_link(p, gamma=FL_GAMMA, eps=FL_EPS):
    w = focal_weight(p, gamma=gamma, eps=eps)
    return w / w.sum(-1, keepdim=True).clamp_min(1e-30)

def _w_inverse_bisect(x, gamma=FL_GAMMA, eps=FL_EPS, n_iter=40):
    x64 = x.to(torch.float64)
    lo = torch.full_like(x64, eps)
    hi = torch.full_like(x64, 1.0 - eps)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        go_up = focal_weight(mid, gamma=gamma, eps=eps) < x64
        lo = torch.where(go_up, mid, lo)
        hi = torch.where(go_up, hi, mid)
    return 0.5 * (lo + hi)

@torch.no_grad()
def focal_link_inv(q, gamma=FL_GAMMA, eps=FL_EPS, n_iter_outer=30, n_iter_inner=40):
    """Return p on simplex with phi(p) = q. Nested bisection, O(K B n_iter)."""
    q64 = q.to(torch.float64).clamp_min(eps)
    shape_Z = q64.shape[:-1] + (1,)
    log_Z_lo = torch.full(shape_Z, -25.0, dtype=torch.float64, device=q.device)
    log_Z_hi = torch.full(shape_Z,  25.0, dtype=torch.float64, device=q.device)
    for _ in range(n_iter_outer):
        log_Z = 0.5 * (log_Z_lo + log_Z_hi)
        p = _w_inverse_bisect(q64 * log_Z.exp(), gamma=gamma, eps=eps, n_iter=n_iter_inner)
        go_up = p.sum(-1, keepdim=True) < 1.0
        log_Z_lo = torch.where(go_up, log_Z, log_Z_lo)
        log_Z_hi = torch.where(go_up, log_Z_hi, log_Z)
    log_Z = 0.5 * (log_Z_lo + log_Z_hi)
    p = _w_inverse_bisect(q64 * log_Z.exp(), gamma=gamma, eps=eps, n_iter=n_iter_inner)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return p.to(q.dtype)

_TANGENT_BASIS_CACHE = {}
def _tangent_basis(K, device=None, dtype=torch.float64):
    """Orthonormal basis of {v in R^K : sum(v) = 0}, shape (K, K-1), cached."""
    key = (K, str(device), dtype)
    if key in _TANGENT_BASIS_CACHE:
        return _TANGENT_BASIS_CACHE[key]
    M = torch.zeros(K, K - 1, dtype=dtype, device=device)
    for i in range(K - 1):
        M[:, i] = -1.0 / K
        M[i, i] += 1.0
    Q, _ = torch.linalg.qr(M)
    _TANGENT_BASIS_CACHE[key] = Q
    return Q

class _PhiInvFunction(torch.autograd.Function):
    """Custom op: forward = phi^{-1}(q),   backward = IFT-restricted solve."""
    @staticmethod
    def forward(ctx, q, gamma, eps):
        p = focal_link_inv(q, gamma=gamma, eps=eps)
        ctx.save_for_backward(p.detach())
        ctx.gamma = gamma
        ctx.eps = eps
        return p
    @staticmethod
    def backward(ctx, grad_p):
        (p,) = ctx.saved_tensors
        gamma, eps = ctx.gamma, ctx.eps
        B, K = p.shape
        p64 = p.to(torch.float64).clamp(eps, 1.0 - eps)
        omp  = 1.0 - p64
        logp = torch.log(p64)
        ell_p  = (gamma * omp.pow(gamma - 1) * logp - omp.pow(gamma) / p64).clamp(max=-1e-30)
        ell_pp = (-gamma * (gamma - 1) * omp.pow(gamma - 2) * logp
                + 2 * gamma * omp.pow(gamma - 1) / p64
                + omp.pow(gamma) / p64.pow(2))
        w        = -1.0 / ell_p
        w_prime  = ell_pp / ell_p.pow(2)
        T        = w.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        phi      = w / T
        # J[b, k, j] = w'(p_j) (delta_kj - phi_k) / T      (column sums = 0; image in T)
        I_K = torch.eye(K, device=p.device, dtype=torch.float64).unsqueeze(0)
        J   = w_prime.unsqueeze(1) * (I_K - phi.unsqueeze(-1)) / T.unsqueeze(-1)   # (B, K, K)
        Bmat = _tangent_basis(K, p.device, torch.float64)             # (K, K-1)
        BT   = Bmat.T                                                 # (K-1, K)
        JT_bas   = BT @ J.transpose(-1, -2) @ Bmat                    # (B, K-1, K-1)
        grad_p64 = grad_p.to(torch.float64)
        g_bas    = grad_p64 @ Bmat                                    # (B, K-1)
        y        = torch.linalg.solve(JT_bas, g_bas.unsqueeze(-1)).squeeze(-1)  # (B, K-1)
        x        = y @ BT                                             # (B, K), lies in T
        return x.to(grad_p.dtype), None, None

class ProperFocalLossIFT(nn.Module):
    """S(q, y) = L_FL(phi^{-1}(q), y)  with exact IFT backward."""
    def __init__(self, gamma=FL_GAMMA, eps=FL_EPS):
        super().__init__()
        self.gamma, self.eps = gamma, eps
    def forward(self, logits, targets):
        q = F.softmax(logits, dim=-1)
        p = _PhiInvFunction.apply(q, self.gamma, self.eps)
        p = p.clamp(self.eps, 1 - self.eps)
        p_y = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-(1 - p_y).pow(self.gamma) * torch.log(p_y)).mean()

# ── Sanity check: IFT backward vs finite differences (double precision) ──────
def _ift_fd_sanity(B=3, K=8, h=1e-4, gamma=FL_GAMMA):
    torch.manual_seed(0)
    logits = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, K, (B,))
    loss_fn = ProperFocalLossIFT(gamma=gamma)
    loss_fn(logits, targets).backward()
    g_ift = logits.grad.detach().clone()
    g_fd = torch.zeros_like(logits)
    with torch.no_grad():
        for b in range(B):
            for k in range(K):
                e = torch.zeros_like(logits); e[b, k] = h
                lp = loss_fn(logits + e, targets).item()
                lm = loss_fn(logits - e, targets).item()
                g_fd[b, k] = (lp - lm) / (2 * h)
    rel = (g_ift - g_fd).abs().max().item() / (g_fd.abs().max().item() + 1e-30)
    print(f"IFT backward vs FD (B={B}, K={K}, h={h}): max rel error = {rel:.2e}")
    assert rel < 1e-3, "IFT backward does not agree with finite differences!"


if __name__ == "__main__":
    _ift_fd_sanity(B=3, K=NUM_CLASSES)
    print("ProperFocalLossIFT defined and verified against finite differences.")