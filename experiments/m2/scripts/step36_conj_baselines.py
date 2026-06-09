"""STEP 36 / the PIVOTAL main-track test — do standard CL methods (DER/Replay/Naive) FAIL on the presence-proof
BINDING task where our oscillator succeeds? Conjunction-binding (step27: 3-object color x shape permutation,
presence=chance 0.167, must bind color->shape). Ours: R6 0.432, R6s 0.484, plainCNN-ctx 0.187. If DER/Replay also
collapse (~chance) -> standard CL CANNOT bind -> our oscillator's UNIQUE contribution (main-track differentiator).
If DER solves it -> binding is not our unique edge. mem matched to ours (~300/3task=100/task -> mem 300).
Usage: python step36_conj_baselines.py --strategy der --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step27_conjunction_binding as s27   # patch shp -> conjunction-binding data (6cls/3task, presence=chance)
import m2_shapes_construct as shp

class SimpleCNN(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                               nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.c = nn.Linear(64, nc)
    def forward(self,x): return self.c(self.f(x))

def run(strategy_name, seed, n_tasks=3, epochs=40, mem=300, device="cuda"):
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.training.supervised import Naive, Replay, DER
    Xtr, ytr = shp._gen_split(600, seed=1000+seed); Xte, yte = shp._gen_split(200, seed=5000+seed)
    tr = shp._experiences(Xtr, ytr, n_exp=n_tasks); te = shp._experiences(Xte, yte, n_exp=n_tasks)
    def tens(p): return (torch.tensor(p[0], dtype=torch.float32), torch.tensor(np.asarray(p[1]), dtype=torch.long))
    bench = tensors_benchmark(train_tensors=[tens(tr[t]) for t in range(n_tasks)],
                              test_tensors=[tens(te[t]) for t in range(n_tasks)], task_labels=[0]*n_tasks)
    torch.manual_seed(seed); model = SimpleCNN(6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    common = dict(train_mb_size=128, eval_mb_size=256, train_epochs=epochs, device=device)
    strat = {"naive":Naive,"replay":Replay,"der":DER}[strategy_name]
    strat = strat(model, opt, crit, **({} if strategy_name=="naive" else {"mem_size":mem}), **common)
    for exp in bench.train_stream: strat.train(exp)
    res = strat.eval(bench.test_stream)
    accs = [v for k,v in res.items() if "Top1_Acc_Stream" in k]
    return {"strategy":strategy_name,"seed":seed,"final_acc":float(np.mean(accs))}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--strategy",default="der"); ap.add_argument("--seeds",type=int,nargs="+",default=[0])
    ap.add_argument("--device",default="cuda"); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--mem",type=int,default=300); ap.add_argument("--out",default="step36_conj_bl.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.strategy,s,epochs=a.epochs,mem=a.mem,device=a.device); r["mem"]=a.mem; recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[conjBL {a.strategy} mem{a.mem} s{s}] final={r['final_acc']:.3f}  (ours R6 0.43/R6s 0.48; chance 0.167)",flush=True)
    print("STEP36_CONJ_BL_DONE")
