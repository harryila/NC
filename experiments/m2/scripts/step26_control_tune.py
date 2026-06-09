"""STEP 26 — VERIFICATION of audit risk #2 (controls under-learn). The plainCNN control had learn_acc ~0.47 (vs
R6 ~0.99) -> it fails to FIT each task, so '+0.85 collapse' may be a learning failure, not a clean forgetting
result. Here we TUNE plainCNN's context-generator (lr 1e-4 -> 1e-3, ctx_dim 8 -> 32) and test:
 (1) SINGLE-task fit (n_tasks=1): can tuned plainCNN reach learn_acc near R6's ~0.99 on one task in isolation?
 (2) Full CL (n_tasks=5, TIGHT): with tuned plainCNN that CAN fit, does it still FORGET (final low)?
If (1) plainCNN still cannot fit -> collapse is architectural -> NECESSITY HOLDS. If (1) fits but (2) forgets ->
necessity holds (oscillator advantage = retention). If (1) fits AND (2) retains -> necessity was a tuning artifact.
Parametrized copy of step9's loop (cg_lr, ctx_dim tunable).
Usage: python step26_control_tune.py --arm plainCNN --cg_lr 1e-3 --ctx_dim 32 --n_tasks 5 --seeds 0 1 2 3
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import m2_shapes_construct as shp
shp.TIGHT = True
import m2_hypernet as H
from h3 import _seeded as h3_seeded
from palr_pool import torch_group_dirs
from avalanche_backbone import LadderClassifier
ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}

def mean_pool_ctx(state, n):
    gd = torch_group_dirs(state, n); return torch.cat([gd.mean(dim=(1,3)), (gd*gd).mean(dim=(1,3))], dim=1)

class PlainCNNContext(nn.Module):
    def __init__(self, ctx_dim, seed):
        super().__init__(); torch.manual_seed(20_000+seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                                 nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,ctx_dim))
    def forward(self,x): return self.net(x)

def run(arm, cg_lr, ctx_dim, seed, n_tasks=5, cl_epochs=40, n_anchors=300, n_replay=128, device="cuda",
        n_per_class=600, max_probe=160):
    Xtr,ytr=shp._gen_split(n_per_class,seed=1000+seed); Xte,yte=shp._gen_split(max(60,max_probe//5),seed=5000+seed)
    exps_tr=shp._experiences(Xtr,ytr,n_exp=n_tasks); exps_te=shp._experiences(Xte,yte,n_exp=n_tasks)
    trunk=H._build_pieces("R6",device,seed)[1]
    for p_ in trunk.parameters(): p_.requires_grad_(False)
    feat_dim=64; crit=nn.CrossEntropyLoss()
    if arm=="plainCNN":
        cg=PlainCNNContext(ctx_dim,seed).to(device); n=4
        def context(xb): return cg(xb)
        cgp=list(cg.parameters())
    else:
        rung,kw=ARM_RUNG[arm]; torch.manual_seed(seed)
        cg=LadderClassifier(rung,num_classes=shp.N_CLASSES,eval_inits=1,base_seed=seed,**kw).to(device); n=int(getattr(cg.net,"n",4))
        def context(xb):
            with h3_seeded(0): _c,_x,xs,_es=cg.net.feature(xb)
            return mean_pool_ctx(xs[1][-1],n)
        cgp=[p for p in cg.parameters() if p.requires_grad]
    hh=H.HyperHead(ctx_dim if arm=="plainCNN" else 2*n, feat_dim, shp.N_CLASSES, device, seed)
    opt=torch.optim.Adam([{"params":cgp,"lr":cg_lr},{"params":list(hh.gen.parameters()),"lr":1e-3}])
    bx=by=None; T=n_tasks; A=np.full((T,T),np.nan)
    for t in range(T):
        Xc=torch.tensor(exps_tr[t][0]); yc=torch.tensor(exps_tr[t][1]).to(device); cg.train(); hh.gen.train()
        for ep in range(cl_epochs):
            pr=torch.randperm(len(yc))
            for i in range(0,len(yc),128):
                idx=pr[i:i+128]; xb=Xc[idx].to(device); opt.zero_grad()
                with torch.no_grad(): f=trunk(xb)
                loss=crit(hh.logits(context(xb),f),yc[idx])
                if bx is not None:
                    rb=torch.randint(0,len(by),(min(n_replay,len(by)),)); xr=bx[rb].to(device)
                    with torch.no_grad(): fr=trunk(xr)
                    loss=loss+crit(hh.logits(context(xr),fr),by[rb].to(device))
                loss.backward(); opt.step()
        cg.eval(); hh.gen.eval()
        with torch.no_grad():
            for tj in range(T):
                Xk=torch.tensor(exps_te[tj][0]); yk=torch.tensor(exps_te[tj][1]).to(device); pr=[]
                for i in range(0,len(Xk),128):
                    xb=Xk[i:i+128].to(device); pr.append((hh.logits(context(xb),trunk(xb)).argmax(1)==yk[i:i+128]).float())
                A[t,tj]=float(torch.cat(pr).mean())
        k=max(1,n_anchors//T); sel=torch.randperm(len(yc))[:k]; sx,sy=Xc[sel].detach().cpu(),yc[sel].detach().cpu()
        bx=sx if bx is None else torch.cat([bx,sx]); by=sy if by is None else torch.cat([by,sy])
    learn=[float(A[k,k]) for k in range(T)]; fwd=[A[k,k]-A[T-1,k] for k in range(T-1)]
    out={"arm":arm,"cg_lr":cg_lr,"ctx_dim":ctx_dim,"seed":seed,"learn_acc":float(np.mean(learn)),
         "final_acc":float(np.mean(A[T-1])),"forgetting":float(np.mean(fwd)) if fwd else 0.0}
    del cg,trunk,hh.gen
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",default="plainCNN"); ap.add_argument("--cg_lr",type=float,default=1e-3); ap.add_argument("--ctx_dim",type=int,default=32)
    ap.add_argument("--n_tasks",type=int,default=5); ap.add_argument("--seeds",type=int,nargs="+",default=[0]); ap.add_argument("--device",default="cuda")
    ap.add_argument("--out",default="step26_tune.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.arm,a.cg_lr,a.ctx_dim,s,n_tasks=a.n_tasks,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[{a.arm} lr={a.cg_lr} ctx={a.ctx_dim} T={a.n_tasks} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}",flush=True)
    print("STEP26_DONE")
