"""STEP 10 — M3 FULLY-ONLINE on REAL IMAGES (SplitCIFAR-10): the make-or-break generalization of the toy-construct
forgetting-bypass. Combines step9's fully-online co-training loop (sequential co-train {context-gen, g} + raw
replay, label-free at test, frozen-random trunk) with step5's deterministic CIFAR-10 loader. Arms: R6 (learned
synchrony), R6s (random coupling), plainCNN (no oscillator). 10-way (5 tasks x 2 classes), chance 0.10.
GATE: does R6 >> R6s/plainCNN online on CIFAR as it did on shapes (R6 0.96, plainCNN collapse)?
Usage: python step10_cifar_online.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import _bootstrap  # noqa
import m2_hypernet as H
from h3 import _seeded as h3_seeded
from palr_pool import torch_group_dirs
from avalanche_backbone import LadderClassifier

ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}
NC = 10

def mean_pool_ctx(state, n):
    gd = torch_group_dirs(state, n)
    return torch.cat([gd.mean(dim=(1, 3)), (gd * gd).mean(dim=(1, 3))], dim=1)

class PlainCNNContext(nn.Module):
    def __init__(self, ctx_dim, seed):
        super().__init__(); torch.manual_seed(20_000 + seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                                 nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,ctx_dim))
    def forward(self, x): return self.net(x)

def _materialize(ds, cap_per_class, seed):
    from torch.utils.data import DataLoader
    xs, ys = [], []
    for b in DataLoader(ds, batch_size=256): xs.append(b[0]); ys.append(b[1])
    X = torch.cat(xs); Y = torch.cat(ys); rng = np.random.RandomState(seed); keep = []
    for c in range(NC):
        idx = np.where(Y.numpy() == c)[0]
        if len(idx): keep.append(rng.permutation(idx)[:cap_per_class])
    keep = np.concatenate(keep) if keep else np.array([], int)
    return X[keep], Y[keep]

def run(arm, seed, n_tasks=5, cl_epochs=40, n_anchors=300, n_replay=128, device="cuda", train_pc=600, eval_pc=400):
    from avalanche.benchmarks.classic import SplitCIFAR10
    bench = SplitCIFAR10(n_experiences=n_tasks, return_task_id=False, seed=seed)
    task_data = []
    for t in range(n_tasks):
        Xt, Yt = _materialize(bench.test_stream[t].dataset, train_pc + eval_pc, seed)
        rng = np.random.RandomState(1000 + seed + t); perm = rng.permutation(len(Yt))
        cut = int(len(perm) * train_pc / (train_pc + eval_pc))
        task_data.append((Xt[perm[:cut]], Yt[perm[:cut]].to(device), Xt[perm[cut:]], Yt[perm[cut:]].to(device)))
    trunk = H._build_pieces("R6", device, seed)[1]
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    ctx_dim = 8; feat_dim = 64; crit = nn.CrossEntropyLoss()
    if arm == "plainCNN":
        cg = PlainCNNContext(ctx_dim, seed).to(device); n = 4
        def context(xb): return cg(xb)
        cg_params = list(cg.parameters())
    else:
        rung, kw = ARM_RUNG[arm]; torch.manual_seed(seed)
        cg = LadderClassifier(rung, num_classes=NC, eval_inits=1, base_seed=seed, **kw).to(device)
        n = int(getattr(cg.net, "n", 4))
        def context(xb):
            with h3_seeded(0):
                _c, _x, xs, _es = cg.net.feature(xb)
            return mean_pool_ctx(xs[1][-1], n)
        cg_params = [p for p in cg.parameters() if p.requires_grad]
    hh = H.HyperHead(ctx_dim, feat_dim, NC, device, seed)
    opt = torch.optim.Adam([{"params": cg_params, "lr": 1e-4}, {"params": list(hh.gen.parameters()), "lr": 1e-3}])

    bx = by = None; T = n_tasks; A = np.full((T, T), np.nan)
    for t in range(T):
        Xc, yc = task_data[t][0], task_data[t][1]
        cg.train(); hh.gen.train()
        for ep in range(cl_epochs):
            pr = torch.randperm(len(yc))
            for i in range(0, len(yc), 128):
                idx = pr[i:i + 128]; xb = Xc[idx].to(device); opt.zero_grad()
                with torch.no_grad(): f = trunk(xb)
                loss = crit(hh.logits(context(xb), f), yc[idx])
                if bx is not None:
                    rb = torch.randint(0, len(by), (min(n_replay, len(by)),)); xr = bx[rb].to(device)
                    with torch.no_grad(): fr = trunk(xr)
                    loss = loss + crit(hh.logits(context(xr), fr), by[rb].to(device))
                loss.backward(); opt.step()
        cg.eval(); hh.gen.eval()
        with torch.no_grad():
            for tj in range(T):
                Xk, yk = task_data[tj][2], task_data[tj][3]; preds = []
                for i in range(0, len(Xk), 128):
                    xb = Xk[i:i + 128].to(device); preds.append((hh.logits(context(xb), trunk(xb)).argmax(1) == yk[i:i+128]).float())
                A[t, tj] = float(torch.cat(preds).mean())
        k = max(1, n_anchors // T); sel = torch.randperm(len(yc))[:k]
        sx, sy = Xc[sel].detach().cpu(), yc[sel].detach().cpu()
        bx = sx if bx is None else torch.cat([bx, sx]); by = sy if by is None else torch.cat([by, sy])
    learn = [float(A[k, k]) for k in range(T)]; fwd = [A[k, k] - A[T - 1, k] for k in range(T - 1)]
    out = {"arm": arm, "seed": seed, "learn_acc": float(np.mean(learn)), "final_acc": float(np.mean(A[T - 1])),
           "forgetting": float(np.mean(fwd)) if fwd else 0.0}
    del cg, trunk, hh.gen
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step10_cifar_online.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = run(a.arm, s, device=a.device); recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[{a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forgetting={r['forgetting']:.3f}  (shapes: R6=0.96, plainCNN=0.11; chance 0.10)", flush=True)
    print("STEP10_DONE")
