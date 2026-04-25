# ── Modulated Cross-Entropy: f(p) = −log(p) · g(p) ──────────────────────────
 #
 # Parametric family:
 #   g(p) = c₀ + Σᵢ cᵢ · (1−p)^{αᵢ},   cᵢ ≥ 0,  αᵢ ≥ 1
 #
 # Convexity of f on (0, 1]:
 #   f''(p) = g(p)/p²  −  2g'(p)/p  −  log(p)·g''(p)
 #              [≥0]        [≥0]           [≥0]
 #   because  g≥0,  g'≤0 (non-increasing),  g''≥0 (convex),  −log(p)≥0.
 #
 # Also: f(0)=+∞  (from −log(0), g(0)>0),   f(1)=0  (from −log(1)=0).
 #
 # Special cases:
 #   CE:       g ≡ 1                     ⟺  c₀=1, K=0
 #   Focal(γ): g = (1−p)^γ  (γ ≥ 1)     ⟺  c₀=0, c₁=1, α₁=γ
 #
 # Normalisation: g(0.5) = 1  ⟹  f(0.5) = log 2  (matches CE).
 # ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ModulatedCELoss(nn.Module):
    """f(p_y) = −log(p_y) · g(p_y),  applied to softmax true-class probability."""
    def __init__(self, seed=None):
        super().__init__()
        rng = np.random.RandomState(seed)
        K = rng.randint(1, 5)
        c0_raw = rng.exponential(0.5)
        cs_raw = rng.exponential(1.0, size=K)
        alphas = np.sort(rng.uniform(1.0, 5.0, size=K))
        g_half = c0_raw + (cs_raw * 0.5 ** alphas).sum()
        c0 = c0_raw / g_half
        cs = cs_raw / g_half
        self.register_buffer("c0", torch.tensor(c0, dtype=torch.float64))
        self.register_buffer("cs", torch.tensor(cs, dtype=torch.float64))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float64))
        
    def g(self, p):
        """Modulation: g(p) = c₀ + Σᵢ cᵢ(1−p)^{αᵢ}"""
        p_ = p.unsqueeze(-1).double()
        return self.c0 + (self.cs * (1.0 - p_).pow(self.alphas)).sum(-1)

    def f(self, p):
        """Per-sample loss on true-class probability."""
        return -torch.log(p.double().clamp(min=1e-12)) * self.g(p)

    def forward(self, logits, targets):
        probs = torch.softmax(logits.double(), dim=-1).clamp(1e-7, 1 - 1e-7)
        p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return self.f(p_true).mean().float()

    def describe(self):
        parts = []
        c0 = self.c0.item()
        if c0 > 0.005:
            parts.append(f"{c0:.3f}")
        for c, a in zip(self.cs.numpy(), self.alphas.numpy()):
            parts.append(f"{c:.3f}·(1−p)^{a:.2f}")
        return "g(p) = " + " + ".join(parts)


if __name__ == "__main__":
    # ── Visualisation ─────────────────────────────────────────────────────────────
    SEEDS = [0, 7, 42, 123, 999]
    p_np  = np.linspace(0.01, 0.999, 500)
    p_t   = torch.tensor(p_np, dtype=torch.float64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(SEEDS)))
    # ── Left: f(p) ───────────────────────────────────────────────────────────────
    for i, seed in enumerate(SEEDS):
        fn = ModulatedCELoss(seed=seed)
        with torch.no_grad():
            fp = fn.f(p_t).numpy()
        ax1.plot(p_np, fp, color=palette[i], lw=1.8, alpha=0.85, label=f"seed={seed}")
    ce = -np.log(p_np)
    ax1.plot(p_np, ce, "k--", lw=2.5, label="CE  (g ≡ 1)")
    focal_gamma = 2.0
    focal_raw = -((1 - p_np) ** focal_gamma) * np.log(p_np)
    focal_scale = np.log(2) / (-((1 - 0.5) ** focal_gamma) * np.log(0.5))
    ax1.plot(p_np, focal_raw * focal_scale, color="crimson", lw=2, ls="-.",
            label=f"Focal γ={focal_gamma:.0f} (scaled)")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 6)
    ax1.set_xlabel("p  (predicted prob for true class)", fontsize=11)
    ax1.set_ylabel("f(p)", fontsize=11)
    ax1.set_title("Loss:  f(p) = −log(p) · g(p)", fontsize=12)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)
    ax1.axvline(0.5, color="grey", lw=0.6, ls=":")
    # ── Right: g(p) ──────────────────────────────────────────────────────────────
    for i, seed in enumerate(SEEDS):
        fn = ModulatedCELoss(seed=seed)
        with torch.no_grad():
            gp = fn.g(p_t).numpy()
        ax2.plot(p_np, gp, color=palette[i], lw=1.8, alpha=0.85, label=f"seed={seed}")
    ax2.axhline(1.0, color="black", lw=2, ls="--", label="CE  (g ≡ 1)")
    g_focal_scaled = ((1 - p_np) ** focal_gamma) * focal_scale
    ax2.plot(p_np, g_focal_scaled, color="crimson", lw=2, ls="-.",
            label=f"Focal γ={focal_gamma:.0f} (scaled)")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("p", fontsize=11)
    ax2.set_ylabel("g(p)", fontsize=11)
    ax2.set_title("Modulation:  g(p) = c₀ + Σ cᵢ(1−p)^{αᵢ}", fontsize=12)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    ax2.axvline(0.5, color="grey", lw=0.6, ls=":")
    ax2.axhline(0.0, color="grey", lw=0.5)
    fig.suptitle(
        "Modulated CE: f(p) = −log(p)·g(p)    |    "
        "g ≥ 0, g′ ≤ 0, g″ ≥ 0  ⟹  f convex    |    "
        "normalised: f(0.5) = log 2",
        fontsize=10.5)
    plt.tight_layout()
    plt.savefig("modulated_ce_losses.png", dpi=150)
    plt.show()