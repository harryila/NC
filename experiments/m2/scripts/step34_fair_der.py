"""STEP 34 / GATE 2 (FAIRNESS crux) — FAIR head-to-head: give DER/Replay the SAME self-sup AE backbone that our
method uses (train-only SSL, no leak), on REAL Tetrominoes. Two variants per strategy:
  freeze  : AE backbone FROZEN + trainable head -> isolates the CONTEXT mechanism (same frozen features as ours).
  ft      : AE backbone FINETUNED (DER's normal full-model mode) -> DER's best case.
Compares to ours R6-learned (~0.93). If R6 >= DER even with DER getting the same backbone -> the oscillator context
genuinely helps beyond features. If DER-ft catches up -> we're competitive (backbone-driven), report honestly.
Usage: python step34_fair_der.py --strategy der --mode ft --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step31_tetrominoes_task as s31
import m2_shapes_construct as shp

def _pretrain_ae(seed, device, n_per_class=300, epochs=15):
    X, _ = shp._gen_split(n_per_class, seed=2000 + seed)   # TRAIN partition only (no test leak)
    Xt = torch.tensor(X, dtype=torch.float32); torch.manual_seed(30_000 + seed)
    enc = nn.Sequential(nn.Conv2d(3,32,5,2,2), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU()).to(device)
    dec = nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), nn.ReLU(),
                        nn.ConvTranspose2d(32,3,5,2,2,output_padding=1), nn.Sigmoid()).to(device)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3); mse = nn.MSELoss(); enc.train(); dec.train()
    for ep in range(epochs):
        pr = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            xb = Xt[pr[i:i+128]].to(device); opt.zero_grad(); loss = mse(dec(enc(xb)), xb); loss.backward(); opt.step()
    enc.eval(); return enc

class AEBackboneNet(nn.Module):
    def __init__(self, enc, nc=6, freeze=True):
        super().__init__(); self.enc = enc
        if freeze:
            for p in self.enc.parameters(): p.requires_grad_(False)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, nc))
    def forward(self, x): return self.head(self.enc(x))

def make_streams(seed, n_tasks=3, train_pc=600, eval_pc=200):
    Xtr, ytr = s31._gen_split(train_pc, seed=1000+seed); Xte, yte = s31._gen_split(eval_pc, seed=5000+seed)
    return s31._experiences(Xtr, ytr, n_exp=n_tasks), s31._experiences(Xte, yte, n_exp=n_tasks)

def run(strategy_name, mode, seed, n_tasks=3, epochs=40, mem=300, device="cuda"):
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.training.supervised import Naive, Replay, DER
    tr, te = make_streams(seed, n_tasks)
    def tens(p): return (torch.tensor(p[0], dtype=torch.float32), torch.tensor(np.asarray(p[1]), dtype=torch.long))
    bench = tensors_benchmark(train_tensors=[tens(tr[t]) for t in range(n_tasks)],
                              test_tensors=[tens(te[t]) for t in range(n_tasks)], task_labels=[0]*n_tasks)
    enc = _pretrain_ae(seed, device)
    model = AEBackboneNet(enc, 6, freeze=(mode=="freeze")).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3); crit = nn.CrossEntropyLoss()
    common = dict(train_mb_size=128, eval_mb_size=256, train_epochs=epochs, device=device)
    strat = {"naive":Naive,"replay":Replay,"der":DER}[strategy_name]
    strat = strat(model, opt, crit, **({} if strategy_name=="naive" else {"mem_size":mem}), **common)
    for exp in bench.train_stream: strat.train(exp)
    res = strat.eval(bench.test_stream)
    accs = [v for k,v in res.items() if "Top1_Acc_Stream" in k]
    return {"strategy":strategy_name,"mode":mode,"seed":seed,"final_acc":float(np.mean(accs))}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--strategy",default="der"); ap.add_argument("--mode",default="ft",choices=["freeze","ft"])
    ap.add_argument("--seeds",type=int,nargs="+",default=[0]); ap.add_argument("--device",default="cuda"); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--out",default="step34_fair.json"); a=ap.parse_args()
    path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.strategy,a.mode,s,epochs=a.epochs,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[fairDER {a.strategy}/{a.mode} s{s}] final_acc={r['final_acc']:.3f}  (ours R6-learned ~0.93; chance 0.167)",flush=True)
    print("STEP34_DONE")
