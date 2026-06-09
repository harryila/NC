"""STEP 32 / (c) — standard CL baselines (avalanche) on the REAL Tetrominoes task (step31 data), for placement on
real object-centric images (not just MNIST). Reuses step28's avalanche run but points it at step31's real-data
generator (6 classes / 3 tasks, class-incremental, disjoint train/test). nc=6.
Usage: python step32_baseline_tetro.py --strategy der --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step31_tetrominoes_task as s31   # sets up the REAL tetrominoes pool + _gen_split/_experiences

def make_streams(seed, n_tasks=3, train_pc=600, eval_pc=200):
    Xtr, ytr = s31._gen_split(train_pc, seed=1000+seed); Xte, yte = s31._gen_split(eval_pc, seed=5000+seed)
    tr = s31._experiences(Xtr, ytr, n_exp=n_tasks); te = s31._experiences(Xte, yte, n_exp=n_tasks)
    return tr, te

class SimpleCNN(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                               nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.c = nn.Linear(64, nc)
    def forward(self,x): return self.c(self.f(x))

def run(strategy_name, seed, n_tasks=3, epochs=40, mem=300, device="cuda"):
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.training.supervised import Naive, EWC, Replay
    try: from avalanche.training.supervised import DER
    except Exception: DER=None
    tr, te = make_streams(seed, n_tasks)
    def tens(pair): return (torch.tensor(pair[0], dtype=torch.float32), torch.tensor(np.asarray(pair[1]), dtype=torch.long))
    bench = tensors_benchmark(train_tensors=[tens(tr[t]) for t in range(n_tasks)],
                              test_tensors=[tens(te[t]) for t in range(n_tasks)], task_labels=[0]*n_tasks)
    torch.manual_seed(seed); model = SimpleCNN(6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    common = dict(train_mb_size=128, eval_mb_size=256, train_epochs=epochs, device=device)
    if strategy_name=="naive": strat = Naive(model, opt, crit, **common)
    elif strategy_name=="replay": strat = Replay(model, opt, crit, mem_size=mem, **common)
    elif strategy_name=="der": strat = DER(model, opt, crit, mem_size=mem, **common)
    else: raise ValueError(strategy_name)
    for exp in bench.train_stream: strat.train(exp)
    res = strat.eval(bench.test_stream)
    accs = [v for k,v in res.items() if "Top1_Acc_Stream" in k]
    return {"strategy":strategy_name,"seed":seed,"final_acc":float(np.mean(accs))}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--strategy",default="der"); ap.add_argument("--seeds",type=int,nargs="+",default=[0])
    ap.add_argument("--device",default="cuda"); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--out",default="step32_tetro_bl.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.strategy,s,epochs=a.epochs,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[REALtetro {a.strategy} s{s}] final_acc={r['final_acc']:.3f}  (ours R6 0.60/R6s 0.65, plainCNN 0.167; chance 0.167)",flush=True)
    print("STEP32_DONE")
