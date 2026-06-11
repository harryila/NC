"""STEP 9 — FULLY-ONLINE M3 (drops the staged joint-pretrain caveat): co-train the oscillator SEQUENTIALLY.
At each task t, train {akorn coupling J, hypernet g} on task t's data + a small RAW-exemplar replay buffer
(re-fed through the LIVE akorn so old task-contexts stay anchored = the anti-forgetting force, addressing the
C3 hazard that the oscillator forgets its own task-phase regions). Label-free at test (phase from x). NO joint
pre-training -- everything is online. Measure final_acc + forgetting. Arms: R6 (learned synchrony) vs R6s
(random coupling) vs plainCNN (no oscillator). This is the hard, real M3 = the original ambition.
Usage: python step9_fully_online.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import _bootstrap  # noqa
import m2_hypernet as H
import m2_shapes_construct as shp
from h3 import _seeded as h3_seeded
from palr_pool import torch_group_dirs
from avalanche_backbone import LadderClassifier

ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {}),
            # RECURRENCE-MATCHED controls (synchrony projection OFF) — isolates synchrony from mere recurrence.
            # R6 - R5 = the causal effect of the Kuramoto phase-coupling (single apply_proj flip, identical machinery).
            "R5": ("R5", {"variant": "no_proj"}),     # apply_proj=False: recurrence+norm ON, synchrony projection OFF (surgical)
            "R5d": ("R5", {"variant": "depthwise"})}  # pure per-oscillator normalized recurrent dynamics (no neuron coupling)

def mean_pool_ctx(state, n):
    gd = torch_group_dirs(state, n)
    return torch.cat([gd.mean(dim=(1, 3)), (gd * gd).mean(dim=(1, 3))], dim=1)

class PlainCNNContext(nn.Module):
    def __init__(self, ctx_dim, seed):
        super().__init__(); torch.manual_seed(20_000 + seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                                 nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,ctx_dim))
    def forward(self, x): return self.net(x)

def run(arm, seed, n_tasks=5, cl_epochs=40, n_anchors=300, n_replay=128, device="cuda",
        n_per_class=600, max_probe=160):
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_tr = shp._experiences(Xtr, ytr, n_exp=n_tasks); exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)
    trunk = H._build_pieces("R6", device, seed)[1]
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    ctx_dim = 8; feat_dim = 64; crit = nn.CrossEntropyLoss()
    if arm == "plainCNN":
        cg = PlainCNNContext(ctx_dim, seed).to(device); n = 4
        def context(xb): return cg(xb)
        cg_params = list(cg.parameters())
    else:
        rung, kw = ARM_RUNG[arm]; torch.manual_seed(seed)
        cg = LadderClassifier(rung, num_classes=shp.N_CLASSES, eval_inits=1, base_seed=seed, **kw).to(device)
        n = int(getattr(cg.net, "n", 4))
        def context(xb):
            with h3_seeded(0):
                _c, _x, xs, _es = cg.net.feature(xb)
            return mean_pool_ctx(xs[1][-1], n)
        cg_params = [p for p in cg.parameters() if p.requires_grad]
    hh = H.HyperHead(ctx_dim, feat_dim, shp.N_CLASSES, device, seed)
    opt = torch.optim.Adam([{"params": cg_params, "lr": 1e-4}, {"params": list(hh.gen.parameters()), "lr": 1e-3}])

    # raw replay buffer (images + labels); re-fed through LIVE cg each step
    bx = by = None; T = n_tasks; A = np.full((T, T), np.nan)
    te_imgs = [(torch.tensor(exps_te[t][0]), torch.tensor(exps_te[t][1]).to(device)) for t in range(T)]
    for t in range(T):
        Xc = torch.tensor(exps_tr[t][0]); yc = torch.tensor(exps_tr[t][1]).to(device)
        cg.train(); hh.gen.train()
        for ep in range(cl_epochs):
            pr = torch.randperm(len(yc))
            for i in range(0, len(yc), 128):
                idx = pr[i:i + 128]; xb = Xc[idx].to(device); opt.zero_grad()
                with torch.no_grad(): f = trunk(xb)
                loss = crit(hh.logits(context(xb), f), yc[idx])
                if bx is not None:
                    rb = torch.randint(0, len(by), (min(n_replay, len(by)),))
                    xr = bx[rb].to(device)
                    with torch.no_grad(): fr = trunk(xr)
                    loss = loss + crit(hh.logits(context(xr), fr), by[rb].to(device))
                loss.backward(); opt.step()
        cg.eval(); hh.gen.eval()
        with torch.no_grad():
            for tj in range(T):
                Xk, yk = te_imgs[tj]; preds = []
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
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step9_online.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = run(a.arm, s, device=a.device); recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[{a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forgetting={r['forgetting']:.3f}  (staged R6=0.97; capture-freeze=0.13)", flush=True)
    print("STEP9_DONE")
