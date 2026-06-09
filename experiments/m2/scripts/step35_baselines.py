"""STEP 35 baselines — standard CL (DER/Replay/Naive, avalanche, own trainable CNN = each method's best) on the
SCALE Tetrominoes benchmark. mem_size = 60*NEXP (matched to ours' replay). nc=NC, n_tasks=NEXP.
Usage: NC_SCALE=12 NEXP_SCALE=6 python step35_baselines.py --strategy der --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step35_scale_data as sd

class SimpleCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                               nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.c = nn.Linear(64, nc)
    def forward(self,x): return self.c(self.f(x))

def run(strategy_name, seed, epochs=40, device="cuda"):
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.training.supervised import Naive, Replay, DER
    NC, NEXP = sd.NC, sd.NEXP; mem = 60*NEXP
    Xtr, ytr = sd._gen_split(600, seed=1000+seed); Xte, yte = sd._gen_split(200, seed=5000+seed)
    tr = sd._experiences(Xtr, ytr); te = sd._experiences(Xte, yte)
    def tens(p): return (torch.tensor(p[0], dtype=torch.float32), torch.tensor(np.asarray(p[1]), dtype=torch.long))
    bench = tensors_benchmark(train_tensors=[tens(tr[t]) for t in range(NEXP)],
                              test_tensors=[tens(te[t]) for t in range(NEXP)], task_labels=[0]*NEXP)
    torch.manual_seed(seed); model = SimpleCNN(NC).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    common = dict(train_mb_size=128, eval_mb_size=256, train_epochs=epochs, device=device)
    strat = {"naive":Naive,"replay":Replay,"der":DER}[strategy_name]
    strat = strat(model, opt, crit, **({} if strategy_name=="naive" else {"mem_size":mem}), **common)
    for exp in bench.train_stream: strat.train(exp)
    res = strat.eval(bench.test_stream)
    accs = [v for k,v in res.items() if "Top1_Acc_Stream" in k]
    return {"strategy":strategy_name,"seed":seed,"final_acc":float(np.mean(accs)),"NC":NC,"NEXP":NEXP}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--strategy",default="der"); ap.add_argument("--seeds",type=int,nargs="+",default=[0])
    ap.add_argument("--device",default="cuda"); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--out",default="step35_bl.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.strategy,s,epochs=a.epochs,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[scaleBL {a.strategy} s{s}] NC={r['NC']} T={r['NEXP']} final={r['final_acc']:.3f}  (chance {1.0/r['NC']:.3f})",flush=True)
    print("STEP35_BL_DONE")
