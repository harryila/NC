"""PALR M0 GATE (the cheapest GO/NO-GO): does a CO-TRAINED LEARNED attention-pool lift the within-shuffle
TASK channel vs the fixed MEAN-pool, and does it beat R6s? Co-trains {coupling J, pool P, hypernet g} end-to-end
through the frozen-random trunk (C1 preserved). NO CL yet — joint training; the gate is purely "is the readout
the bottleneck". GATE (pre-registered): learned-pool R6 task-lift >= mean-pool R6 task-lift + 0.08 AND
learned-pool R6 > learned-pool R6s. If not -> the pool is not the bottleneck -> honest NO-GO.
Usage: python step6_palr_m0.py --arm R6 --pool learned --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import _bootstrap  # noqa
import m2_hypernet as H
import m2_shapes_construct as shp
from h3 import _seeded as h3_seeded
from palr_pool import torch_group_dirs, LearnedPhasePool
from avalanche_backbone import LadderClassifier

ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}

def mean_pool_ctx(state, n):
    """Differentiable mean+meansq pool over sites (the OLD fixed descriptor), -> (B, 2n)."""
    gd = torch_group_dirs(state, n)             # (B,G,n,HW)
    m = gd.mean(dim=(1, 3))                      # (B,n)
    ms = (gd * gd).mean(dim=(1, 3))             # (B,n)
    return torch.cat([m, ms], dim=1)            # (B,2n)

def run(arm, pool_kind, seed, n_tasks=5, epochs=30, device="cuda", n_per_class=600, max_probe=160,
        lr_j=1e-4, lr_pg=1e-3, d_ctx=16):
    rung, kw = ARM_RUNG[arm]
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)
    torch.manual_seed(seed)
    akorn = LadderClassifier(rung, num_classes=shp.N_CLASSES, eval_inits=1, base_seed=seed, **kw).to(device)
    n = int(getattr(akorn.net, "n", 4))
    # frozen-random trunk (C1)
    trunk, feat_dim = H._build_pieces(arm, device, seed)[1], 64
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    # probe C at layer 1
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    with torch.no_grad(), h3_seeded(0):
        _c, _x, xs, _es = akorn.net.feature(Xt[:2].to(device))
    C = xs[1][-1].shape[1]; G = C // n
    pool = LearnedPhasePool(G, n, d_ctx=d_ctx, K=8).to(device) if pool_kind == "learned" else None
    ctx_dim = d_ctx if pool_kind == "learned" else 2 * n
    hh = H.HyperHead(ctx_dim, feat_dim, shp.N_CLASSES, device, seed)

    def context(xb):
        with h3_seeded(0):
            _c, _x, xs, _es = akorn.net.feature(xb)
        st = xs[1][-1]
        return pool(torch_group_dirs(st, n)) if pool_kind == "learned" else mean_pool_ctx(st, n)

    params = [{"params": [p for p in akorn.parameters() if p.requires_grad], "lr": lr_j},
              {"params": list(hh.gen.parameters()), "lr": lr_pg}]
    if pool is not None: params.append({"params": list(pool.parameters()), "lr": lr_pg})
    opt = torch.optim.Adam(params); crit = nn.CrossEntropyLoss()
    akorn.train()
    if pool: pool.train()
    hh.gen.train()
    yt_dev = yt.to(device)
    for ep in range(epochs):
        pr = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = pr[i:i + 128]; xb = Xt[idx].to(device)
            opt.zero_grad()
            c = context(xb)
            with torch.no_grad():
                f = trunk(xb)
            loss = crit(hh.logits(c, f), yt_dev[idx])
            loss.backward(); opt.step()
    akorn.eval()
    if pool: pool.eval()
    hh.gen.eval()
    # eval per task: real / within-task-shuffle / const
    test_cf = []
    with torch.no_grad():
        for t in range(n_tasks):
            Xk = torch.tensor(exps_te[t][0]); yk = torch.tensor(exps_te[t][1]).to(device)
            cs, fs = [], []
            for i in range(0, len(Xk), 128):
                xb = Xk[i:i + 128].to(device); cs.append(context(xb)); fs.append(trunk(xb))
            test_cf.append((torch.cat(cs), torch.cat(fs), yk))
    g = torch.Generator(device=device).manual_seed(7000 + seed)
    def ev(mode):
        accs = []
        with torch.no_grad():
            for (c, f, y) in test_cf:
                if mode == "real": cc = c
                elif mode == "within": cc = c[torch.randperm(c.shape[0], generator=g, device=device)]
                elif mode == "const": cc = torch.ones_like(c)
                accs.append(float((hh.logits(cc, f).argmax(1) == y).float().mean()))
        return float(np.mean(accs))
    real, within, const = ev("real"), ev("within"), ev("const")
    out = {"arm": arm, "pool": pool_kind, "seed": seed, "acc_real": real, "acc_within": within,
           "acc_const": const, "task_lift": within - const, "instance_lift": real - within, "ctx_dim": ctx_dim}
    del akorn, trunk, hh.gen
    if pool: del pool
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--pool", default="learned", choices=["learned", "mean"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step6_palr_m0.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = run(a.arm, a.pool, s, epochs=a.epochs, device=a.device); recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[{a.arm} pool={a.pool} s{s}] real={r['acc_real']:.3f} within={r['acc_within']:.3f} "
              f"const={r['acc_const']:.3f} | TASK_lift={r['task_lift']:.3f} inst={r['instance_lift']:.3f} ctxdim={r['ctx_dim']}", flush=True)
    print("PALR_M0_DONE")
