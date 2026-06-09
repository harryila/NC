"""STEP 12 / B2 + A-gate — ROUTER TEST on CIFAR: after co-training+freezing the context generator, can a
label-free router send a held-out input to its correct TASK from the phase context alone, and is it
SYNCHRONY-SPECIFIC (R6 > R6s) and BETTER than a non-oscillator centroid baseline? This is the cheap go/no-go
for PHLOX (freeze-and-grow + prototype routing): PHLOX converts drift-forgetting into routing-error, so if
routing fails here, PHLOX is capped. Oracle ceiling is 0.70 (headroom exists), chance routing = 0.20 (5 tasks).
Routers: (proto) nearest per-task prototype in the co-trained PHASE-context space; (trunkcentroid) nearest
per-task centroid in the frozen-random TRUNK feature space (non-oscillator baseline). GO: proto slot-acc high
AND R6 proto > R6s proto AND R6 proto > trunkcentroid.
Usage: python step12_router_test.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import _bootstrap  # noqa
import m2_hypernet as H
from h3 import _seeded as h3_seeded
from palr_pool import torch_group_dirs
from avalanche_backbone import LadderClassifier
ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}; NC = 10

def mean_pool_ctx(state, n):
    gd = torch_group_dirs(state, n)
    return torch.cat([gd.mean(dim=(1, 3)), (gd * gd).mean(dim=(1, 3))], dim=1)

def _materialize(ds, cap, seed):
    from torch.utils.data import DataLoader
    xs, ys = [], []
    for b in DataLoader(ds, batch_size=256): xs.append(b[0]); ys.append(b[1])
    X = torch.cat(xs); Y = torch.cat(ys); rng = np.random.RandomState(seed); keep = []
    for c in range(NC):
        idx = np.where(Y.numpy() == c)[0]
        if len(idx): keep.append(rng.permutation(idx)[:cap])
    return X[np.concatenate(keep)], Y[np.concatenate(keep)]

def run(arm, seed, n_tasks=5, joint_epochs=30, device="cuda", train_pc=600, eval_pc=400):
    from avalanche.benchmarks.classic import SplitCIFAR10
    bench = SplitCIFAR10(n_experiences=n_tasks, return_task_id=False, seed=seed)
    td = []
    for t in range(n_tasks):
        Xt, Yt = _materialize(bench.test_stream[t].dataset, train_pc + eval_pc, seed)
        rng = np.random.RandomState(1000 + seed + t); perm = rng.permutation(len(Yt)); cut = int(len(perm) * train_pc / (train_pc + eval_pc))
        td.append((Xt[perm[:cut]], Yt[perm[:cut]], Xt[perm[cut:]], Yt[perm[cut:]]))
    trunk = H._build_pieces("R6", device, seed)[1]
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    rung, kw = ARM_RUNG[arm]; torch.manual_seed(seed)
    akorn = LadderClassifier(rung, num_classes=NC, eval_inits=1, base_seed=seed, **kw).to(device)
    n = int(getattr(akorn.net, "n", 4)); crit = nn.CrossEntropyLoss()
    def context(xb):
        with h3_seeded(0):
            _c, _x, xs, _es = akorn.net.feature(xb)
        return mean_pool_ctx(xs[1][-1], n)
    # joint co-train {akorn, g_tmp}
    Xtr = torch.cat([td[t][0] for t in range(n_tasks)]); Ytr = torch.cat([td[t][1] for t in range(n_tasks)]).to(device)
    g = H.HyperHead(2*n, 64, NC, device, seed)
    opt = torch.optim.Adam([{"params":[p for p in akorn.parameters() if p.requires_grad],"lr":1e-4},{"params":list(g.gen.parameters()),"lr":1e-3}])
    akorn.train(); g.gen.train()
    for ep in range(joint_epochs):
        pr = torch.randperm(len(Ytr))
        for i in range(0,len(Ytr),128):
            idx=pr[i:i+128]; xb=Xtr[idx].to(device); opt.zero_grad()
            with torch.no_grad(): f=trunk(xb)
            loss=crit(g.logits(context(xb),f),Ytr[idx]); loss.backward(); opt.step()
    akorn.eval()
    for p_ in akorn.parameters(): p_.requires_grad_(False)
    # prototypes from TRAIN, route TEST
    def cache_ctx(X):
        cs=[]
        with torch.no_grad():
            for i in range(0,len(X),128): cs.append(context(X[i:i+128].to(device)))
        return torch.cat(cs)
    def cache_feat(X):
        fs=[]
        with torch.no_grad():
            for i in range(0,len(X),128): fs.append(trunk(X[i:i+128].to(device)))
        return torch.cat(fs)
    proto_ctx = torch.stack([cache_ctx(td[t][0]).mean(0) for t in range(n_tasks)])     # (T, 2n)
    cent_feat = torch.stack([cache_feat(td[t][0]).mean(0) for t in range(n_tasks)])     # (T, 64)
    # eval: route each held-out test sample to nearest prototype/centroid
    proto_correct=cent_correct=tot=0
    with torch.no_grad():
        for t in range(n_tasks):
            cx=cache_ctx(td[t][2]); ff=cache_feat(td[t][2])
            dp=torch.cdist(cx, proto_ctx); proto_correct += int((dp.argmin(1)==t).sum())
            dc=torch.cdist(ff, cent_feat); cent_correct += int((dc.argmin(1)==t).sum())
            tot += len(cx)
    out={"arm":arm,"seed":seed,"proto_slotacc":proto_correct/tot,"trunkcentroid_slotacc":cent_correct/tot,"chance":1.0/n_tasks}
    del akorn,trunk,g.gen
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step12_router.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.arm,s,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[{a.arm} s{s}] proto_slotacc={r['proto_slotacc']:.3f} trunkcentroid={r['trunkcentroid_slotacc']:.3f} (chance {r['chance']:.2f})",flush=True)
    print("STEP12_DONE")
