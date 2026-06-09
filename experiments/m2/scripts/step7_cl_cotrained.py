"""STEP 7 — THE M3 PRIZE TEST: does a CO-TRAINED (near-oracle) phase context survive SEQUENTIAL CL?
Phase 1: jointly co-train {akorn coupling J, a temporary g} on ALL tasks via the generated head on a frozen-
random trunk -> a near-oracle LABEL-FREE phase-context generator (M0 showed within-task acc ~0.49 ~ oracle 0.52).
FREEZE the akorn. Phase 2: a FRESH hypernet g_cl learns the 5 tasks SEQUENTIALLY with small raw replay; the
frozen co-trained context is the only task signal. Measure final_acc + forgetting. Mirrors the oracle test
(which retained 0.52) but with the REAL co-trained phase context instead of one-hot.
Compare arms R6 vs R6s; reference points (from step4c): capture-then-freeze=0.13 (forgets), oracle=0.52 (retains).
STAGED CAVEAT (honest): the context generator is pre-trained JOINTLY (sees all tasks); g_cl is the sequential
learner. This is the upper-bound feasibility test; fully-online co-training (grow+freeze) is the harder follow-up.
Usage: python step7_cl_cotrained.py --arm R6 --seeds 0 1 2 3 --device cuda
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

ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}

def mean_pool_ctx(state, n):
    gd = torch_group_dirs(state, n)
    return torch.cat([gd.mean(dim=(1, 3)), (gd * gd).mean(dim=(1, 3))], dim=1)   # (B,2n)

def run(arm, seed, n_tasks=5, joint_epochs=30, cl_epochs=40, n_anchors=300, n_replay=128,
        device="cuda", n_per_class=600, max_probe=160):
    rung, kw = ARM_RUNG[arm]
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_tr = shp._experiences(Xtr, ytr, n_exp=n_tasks); exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)
    torch.manual_seed(seed)
    akorn = LadderClassifier(rung, num_classes=shp.N_CLASSES, eval_inits=1, base_seed=seed, **kw).to(device)
    n = int(getattr(akorn.net, "n", 4))
    trunk = H._build_pieces(arm, device, seed)[1]
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); yt_dev = yt.to(device)
    feat_dim = 64; ctx_dim = 2 * n; crit = nn.CrossEntropyLoss()

    def context(xb):
        with h3_seeded(0):
            _c, _x, xs, _es = akorn.net.feature(xb)
        return mean_pool_ctx(xs[1][-1], n)

    # ---- PHASE 1: joint co-train {akorn J, g_tmp} on ALL tasks (produces near-oracle context generator) ----
    g_tmp = H.HyperHead(ctx_dim, feat_dim, shp.N_CLASSES, device, seed)
    opt1 = torch.optim.Adam([{"params": [p for p in akorn.parameters() if p.requires_grad], "lr": 1e-4},
                             {"params": list(g_tmp.gen.parameters()), "lr": 1e-3}])
    akorn.train(); g_tmp.gen.train()
    for ep in range(joint_epochs):
        pr = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = pr[i:i + 128]; xb = Xt[idx].to(device); opt1.zero_grad()
            with torch.no_grad(): f = trunk(xb)
            loss = crit(g_tmp.logits(context(xb), f), yt_dev[idx]); loss.backward(); opt1.step()
    akorn.eval()
    for p_ in akorn.parameters(): p_.requires_grad_(False)
    del g_tmp.gen

    # cache frozen contexts + features for CL (akorn now frozen)
    def cache(X):
        cs, fs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = X[i:i + 128].to(device); cs.append(context(xb)); fs.append(trunk(xb))
        return torch.cat(cs), torch.cat(fs)
    train_cf = [(cache(torch.tensor(exps_tr[t][0])) + (torch.tensor(exps_tr[t][1]).to(device),)) for t in range(n_tasks)]
    test_cf = [(cache(torch.tensor(exps_te[t][0])) + (torch.tensor(exps_te[t][1]).to(device),)) for t in range(n_tasks)]

    # ---- PHASE 2: FRESH g_cl learns tasks SEQUENTIALLY with replay (akorn frozen) ----
    hh = H.HyperHead(ctx_dim, feat_dim, shp.N_CLASSES, device, seed); opt = torch.optim.Adam(hh.gen.parameters(), lr=1e-3)
    bctx = bf = by = None; T = n_tasks; A = np.full((T, T), np.nan)
    for t in range(T):
        ctx_t, f_t, y_t = train_cf[t]; hh.gen.train()
        for ep in range(cl_epochs):
            pr = torch.randperm(len(y_t), device=device)
            for i in range(0, len(y_t), 128):
                idx = pr[i:i + 128]; opt.zero_grad(); loss = crit(hh.logits(ctx_t[idx], f_t[idx]), y_t[idx])
                if by is not None:
                    rb = torch.randint(0, len(by), (min(n_replay, len(by)),), device=device)
                    loss = loss + crit(hh.logits(bctx[rb], bf[rb]), by[rb])
                loss.backward(); opt.step()
        hh.gen.eval()
        with torch.no_grad():
            for tj in range(T):
                c, f, y = test_cf[tj]; A[t, tj] = float((hh.logits(c, f).argmax(1) == y).float().mean())
        k = max(1, n_anchors // T); sel = torch.randperm(len(y_t), device=device)[:k]
        sc, sf, sy = ctx_t[sel].detach(), f_t[sel].detach(), y_t[sel].detach()
        bctx = sc if bctx is None else torch.cat([bctx, sc]); bf = sf if bf is None else torch.cat([bf, sf]); by = sy if by is None else torch.cat([by, sy])
    learn = [float(A[k, k]) for k in range(T)]; fwd = [A[k, k] - A[T - 1, k] for k in range(T - 1)]
    out = {"arm": arm, "seed": seed, "learn_acc": float(np.mean(learn)), "final_acc": float(np.mean(A[T - 1])),
           "forgetting": float(np.mean(fwd)) if fwd else 0.0}
    del akorn, trunk, hh.gen
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step7_cl_cotrained.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = run(a.arm, s, device=a.device); recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[{a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forgetting={r['forgetting']:.3f}  (capture-freeze=0.13, oracle=0.52)", flush=True)
    print("STEP7_DONE")
