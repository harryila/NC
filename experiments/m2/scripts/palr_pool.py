"""PALR load-bearing piece: a DIFFERENTIABLE learned attention-pool over the L1 phase field, replacing the
fixed mean-pool (_pool_phase_state) that every M2/M3 null shared. Plus a unit test (grad flows + matches numpy
group_directions structure + deterministic) to run BEFORE any GPU spend.

torch_group_dirs: (B,C,H,W) phase -> (B, G, n, H*W) unit group-vectors (normalized over the n-axis), G=C//n.
LearnedPhasePool: attention over sites preserving the group axis -> context c (d_ctx). Tiny (<5k params).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def torch_group_dirs(state, n):
    """state (B,C,H,W) -> (B, G, n, HW) L2-normalized over the n axis. Differentiable."""
    B, C, H, W = state.shape
    assert C % n == 0, f"C={C} not divisible by n={n}"
    G = C // n
    a = state.reshape(B, G, n, H * W)                       # (B,G,n,HW)
    a = a / a.norm(dim=2, keepdim=True).clamp(min=1e-12)
    return a

class LearnedPhasePool(nn.Module):
    """Reads (B,G,n,HW) group-directions; K attention maps over sites (shared across groups); weight-pools
    the per-group unit n-vectors -> (B, G*K*n) -> Linear -> c (B, d_ctx). Preserves group/relational structure
    the mean-pool destroys. Co-trained end-to-end."""
    def __init__(self, G, n, d_ctx=16, K=8):
        super().__init__()
        self.G, self.n, self.K = G, n, K
        # scorer: per-site feature = the G*n group-dir values -> K attention logits per site
        self.score = nn.Linear(G * n, K)
        self.proj = nn.Linear(G * K * n, d_ctx)
    def forward(self, gd):                                   # gd: (B,G,n,HW)
        B, G, n, HW = gd.shape
        feat = gd.reshape(B, G * n, HW).transpose(1, 2)      # (B, HW, G*n)
        logits = self.score(feat)                            # (B, HW, K)
        attn = F.softmax(logits, dim=1)                      # over sites, (B,HW,K)
        # weight-pool each group's n-vector by each head's site-attention
        # gd (B,G,n,HW) x attn (B,HW,K) -> (B,G,n,K)
        pooled = torch.einsum('bgnh,bhk->bgnk', gd, attn)    # (B,G,n,K)
        c = self.proj(pooled.reshape(B, G * self.K * n))     # (B,d_ctx)
        return c

# ----------------- UNIT TEST (run before any GPU spend) -----------------
def _unit_test():
    import sys; sys.path.insert(0, "experiments/m1_wk0")
    torch.manual_seed(0)
    B, C, H, W, n = 2, 64, 8, 8, 4
    state = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    gd = torch_group_dirs(state, n)
    # (1) shape
    assert gd.shape == (B, C // n, n, H * W), f"bad gd shape {gd.shape}"
    # (2) unit norm over n-axis
    norms = gd.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "group dirs not unit norm"
    # (3) match numpy h3.group_directions structure (no subsample) up to row ordering
    import h3
    np_gd = h3.group_directions(state.detach().numpy(), n=n, max_sites=0)   # (B*G*HW, n)
    t_rows = gd.permute(0, 1, 3, 2).reshape(-1, n).detach().numpy()         # (B,G,HW,n)->rows
    assert np_gd.shape == t_rows.shape, f"np {np_gd.shape} vs torch {t_rows.shape}"
    diff = np.abs(np.sort(np_gd, axis=0) - np.sort(t_rows, axis=0)).max()
    assert diff < 1e-4, f"torch vs numpy group_directions mismatch {diff}"
    # (4) grad flows through pool into state (=> will reach coupling J when state=feature(x))
    pool = LearnedPhasePool(C // n, n, d_ctx=16, K=8).double()
    c = pool(torch_group_dirs(state, n))
    assert c.shape == (B, 16)
    loss = c.sum(); loss.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0, "grad dead"
    npar = sum(p.numel() for p in pool.parameters())
    print(f"UNIT_TEST_PASS: shapes ok | unit-norm ok | numpy-match {diff:.2e} | grad finite&nonzero | pool_params={npar}")

if __name__ == "__main__":
    _unit_test()
