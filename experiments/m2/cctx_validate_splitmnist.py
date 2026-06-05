"""M2 instrument validation on Split-MNIST: confirm ctx_channel_capacity reads a REAL trained
phase/task-conditioned theta-generator as C_ctx >> 0, and a task-AGNOSTIC generator as C_ctx ~ 0 —
reproducing CCC's (arXiv 2603.07415) dichotomy (hypernetwork C_ctx>>0 vs state-modifier ~0) on GRADIENT-
TRAINED weights (not synthetic theta). This validates the estimator BEFORE trusting it on AKOrN phase-context.

NOT a scientific claim (CCC warns Split-MNIST is non-discriminating for the headline). Purely: does our
measurement instrument behave correctly on a known case with real trained generators?

Design (small, CPU/GPU-agnostic, OOM-safe — tiny MLP, 5 binary tasks):
  - context c per sample = the TASK's one-hot-ish signal embedded + jittered (a clean "context that carries
    task identity", standing in for what AKOrN phase-state provides — here we control its task-info exactly).
  - CONDITIONED generator g_cond: c -> theta(head)  (a 2-layer hypernet). Trained so the generated head
    solves each task's 2-way split. Should yield C_ctx ~ high (theta varies with task-context).
  - AGNOSTIC generator g_agno: ignores c (context replaced by a constant) -> one shared theta. Trained the
    same way. Should yield C_ctx ~ 0 (theta constant across contexts).
Then run ctx_channel_capacity.estimate_c_ctx on {task -> theta(c)} for both and assert cond >> agno.

Usage:  python cctx_validate_splitmnist.py            # full (needs torch; uses CPU if no cuda)
        python cctx_validate_splitmnist.py --demo     # numpy-only logic check, no torch/training
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_M1 = os.path.normpath(os.path.join(_HERE, "..", "m1_wk0"))
for p in (_HERE, _M1):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import ctx_channel_capacity as ccc


def _make_splitmnist(n_tasks=5, per_class=200, dim=64, seed=0):
    """Tiny synthetic stand-in for Split-MNIST: n_tasks 2-way classification problems in R^dim.
    Each task t has two Gaussian blobs (its 2 classes) at task-specific, separable means. Returns
    {task: (X[n,dim], y[n] in {0,1})}. Deterministic. (We don't need real MNIST pixels to validate the
    ESTIMATOR — we need real gradient-trained generator weights on a separable multi-task problem.)"""
    rng = np.random.default_rng(seed)
    tasks = {}
    centers = rng.normal(0, 3.0, size=(n_tasks, 2, dim))   # 2 class-centers per task, well separated
    for t in range(n_tasks):
        X, y = [], []
        for c in (0, 1):
            X.append(rng.normal(centers[t, c], 1.0, size=(per_class, dim)))
            y.append(np.full(per_class, c))
        tasks[t] = (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64))
    return tasks, dim


def run_validation(n_tasks=5, dim=64, epochs=60, ctx_dim=16, hidden=32, device="cpu", seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    tasks, D = _make_splitmnist(n_tasks=n_tasks, dim=dim, seed=seed)
    n_classes = 2

    # context: a per-task vector carrying task identity (clean stand-in for AKOrN phase-context).
    ctx_table = torch.randn(n_tasks, ctx_dim, generator=torch.Generator().manual_seed(seed + 1))

    class HyperGen(nn.Module):
        """c -> theta (head weight (D,n_classes) + bias). If agnostic=True, ignores c (uses a learned
        constant context) so theta is shared across tasks -> the CCC state-modifier limit (C_ctx~0)."""
        def __init__(self, agnostic=False):
            super().__init__()
            self.agnostic = agnostic
            self.const_ctx = nn.Parameter(torch.zeros(ctx_dim))   # used only when agnostic
            self.net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, D * n_classes + n_classes))

        def theta(self, c):
            if self.agnostic:
                c = self.const_ctx.expand(c.shape[0], -1)
            out = self.net(c)
            W = out[:, :D * n_classes].reshape(-1, D, n_classes)
            b = out[:, D * n_classes:]
            return W, b

        def forward(self, c, x):
            W, b = self.theta(c)
            return torch.einsum("bd,bdk->bk", x, W) + b     # per-sample generated head

    def train(agnostic):
        g = HyperGen(agnostic=agnostic).to(device)
        opt = torch.optim.Adam(g.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xall = {t: torch.tensor(tasks[t][0], device=device) for t in tasks}
        Yall = {t: torch.tensor(tasks[t][1], device=device) for t in tasks}
        for _ in range(epochs):
            for t in range(n_tasks):
                c = ctx_table[t].to(device).expand(Xall[t].shape[0], -1)
                opt.zero_grad()
                loss = lossf(g(c, Xall[t]), Yall[t])
                loss.backward(); opt.step()
        # accuracy + per-task generated theta (flattened) for the C_ctx estimator
        accs, theta_by_task = [], {}
        g.eval()
        with torch.no_grad():
            for t in range(n_tasks):
                c1 = ctx_table[t].to(device).unsqueeze(0)
                W, b = g.theta(c1)
                theta_by_task[t] = torch.cat([W.flatten(), b.flatten()]).cpu().numpy()
                c = ctx_table[t].to(device).expand(Xall[t].shape[0], -1)
                pred = g(c, Xall[t]).argmax(1)
                accs.append((pred == Yall[t]).float().mean().item())
        return float(np.mean(accs)), theta_by_task

    acc_cond, theta_cond = train(agnostic=False)
    acc_agno, theta_agno = train(agnostic=True)

    # C_ctx on {task -> theta(c)} for each generator. Replicate each task's theta a few times with tiny
    # jitter so the decodability probe has samples per class (theta is deterministic per task otherwise).
    def cls_dict(theta_by_task, reps=12, jit=1e-4):
        rng = np.random.default_rng(seed + 7)
        return {t: [theta_by_task[t] + rng.normal(0, jit, size=theta_by_task[t].shape) for _ in range(reps)]
                for t in theta_by_task}

    C_cond = ccc.compute_C_ctx(cls_dict(theta_cond))
    C_agno = ccc.compute_C_ctx(cls_dict(theta_agno))
    return {
        "acc_conditioned": acc_cond, "acc_agnostic": acc_agno,
        "C_ctx_conditioned_bits": C_cond["mi_lower_bits"], "C_ctx_agnostic_bits": C_agno["mi_lower_bits"],
        "Hmax_bits": C_cond["Hmax_bits"],
        "eff_dim_cond": C_cond["eff_dim_bits"], "eff_dim_agno": C_agno["eff_dim_bits"],
        "VALIDATION_PASS": bool(C_cond["mi_lower_bits"] > C_agno["mi_lower_bits"] + 0.5
                                and C_agno["mi_lower_bits"] < 0.3
                                and acc_cond > 0.9),
    }


def _demo():
    """numpy-only: confirm the estimator dichotomy on synthetic conditioned vs agnostic theta (no torch)."""
    rng = np.random.default_rng(0)
    n_tasks, dim = 5, 130
    cond = {t: [rng.normal(t * 2.0, 0.05, size=dim) for _ in range(12)] for t in range(n_tasks)}  # task-sep
    agno = {t: [rng.normal(0.0, 0.05, size=dim) for _ in range(12)] for t in range(n_tasks)}      # shared
    Cc = ccc.compute_C_ctx(cond); Ca = ccc.compute_C_ctx(agno)
    print(f"[demo] conditioned C_ctx={Cc['mi_lower_bits']:.3f} bits  agnostic C_ctx={Ca['mi_lower_bits']:.3f} "
          f"(Hmax={Cc['Hmax_bits']:.3f})")
    assert Cc["mi_lower_bits"] > Ca["mi_lower_bits"] + 0.5 and Ca["mi_lower_bits"] < 0.3
    print("=== VALIDATE DEMO OK (estimator separates conditioned vs agnostic) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    if args.demo:
        _demo(); return
    import json
    out = run_validation(epochs=args.epochs, device=args.device)
    print(json.dumps(out, indent=2, default=str))
    os.makedirs(os.path.join(_M1, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(_M1, "results", "cctx_validate_splitmnist.json"), "w"), indent=2, default=str)
    print("VALIDATION_PASS:", out["VALIDATION_PASS"])


if __name__ == "__main__":
    main()
