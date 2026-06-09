"""STEP 28 / (c) — STANDARD CL BASELINES to PLACE the result. Runs avalanche's standard strategies (Naive, EWC,
Replay, DER) on the SAME multi-object online stream (2-digit MNIST, 5 tasks, 10 classes, class-incremental =
label-free at test) with a standard CNN, and reports final avg accuracy. Compares to our oscillator method
(MNIST-2obj: R6 ~0.65, plainCNN ~0.19). Uses avalanche's OWN implementations (not hand-rolled) for fairness.
Replay/DER memory matched to our buffer (~300).
Usage: python step28_cl_baselines.py --strategy ewc --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step16_multiobj_mnist as s16   # reuse the validated 2-digit MNIST generator (_gen_split/_experiences)

def make_streams(seed, n_tasks=5, train_pc=600, eval_pc=200):
    Xtr, ytr = s16._gen_split(train_pc, seed=1000+seed); Xte, yte = s16._gen_split(eval_pc, seed=5000+seed)
    tr = s16._experiences(Xtr, ytr, n_exp=n_tasks); te = s16._experiences(Xte, yte, n_exp=n_tasks)
    return tr, te

class SimpleCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                               nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.c = nn.Linear(64, nc)
    def forward(self,x): return self.c(self.f(x))

def run(strategy_name, seed, n_tasks=5, epochs=40, mem=300, device="cuda"):
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.training.supervised import Naive, EWC, Replay
    try: from avalanche.training.supervised import DER, SynapticIntelligence as SI, LwF
    except Exception: DER=SI=LwF=None
    tr, te = make_streams(seed, n_tasks)
    def tens(pair):
        return (torch.tensor(pair[0], dtype=torch.float32), torch.tensor(np.asarray(pair[1]), dtype=torch.long))
    bench = tensors_benchmark(
        train_tensors=[tens(tr[t]) for t in range(n_tasks)],
        test_tensors=[tens(te[t]) for t in range(n_tasks)],
        task_labels=[0]*n_tasks)   # class-incremental (label-free at test): single task label
    torch.manual_seed(seed); model = SimpleCNN(10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    common = dict(train_mb_size=128, eval_mb_size=256, train_epochs=epochs, device=device)
    if strategy_name=="naive": strat = Naive(model, opt, crit, **common)
    elif strategy_name=="ewc": strat = EWC(model, opt, crit, ewc_lambda=10.0, **common)
    elif strategy_name=="si": strat = SI(model, opt, crit, si_lambda=1.0, **common)
    elif strategy_name=="lwf": strat = LwF(model, opt, crit, alpha=1.0, temperature=2.0, **common)
    elif strategy_name=="replay": strat = Replay(model, opt, crit, mem_size=mem, **common)
    elif strategy_name=="der":
        if DER is None: return {"strategy":"der","seed":seed,"final_acc":None,"note":"DER unavailable"}
        strat = DER(model, opt, crit, mem_size=mem, **common)
    else: raise ValueError(strategy_name)
    for exp in bench.train_stream: strat.train(exp)
    res = strat.eval(bench.test_stream)
    # extract avg class-incremental accuracy across the stream
    accs = [v for k,v in res.items() if "Top1_Acc_Stream" in k]
    final = float(np.mean(accs)) if accs else float(np.mean([v for k,v in res.items() if "Acc_Exp" in k]))
    return {"strategy":strategy_name,"seed":seed,"final_acc":final}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--strategy",default="naive"); ap.add_argument("--seeds",type=int,nargs="+",default=[0]); ap.add_argument("--device",default="cuda")
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--out",default="step28_baselines.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    if os.path.exists(path):
        try: recs=json.load(open(path)).get("runs",[])
        except: recs=[]
    for s in a.seeds:
        r=run(a.strategy,s,epochs=a.epochs,device=a.device); recs.append(r)
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[{a.strategy} s{s}] final_acc={r.get('final_acc')}  (ours: R6 0.65, plainCNN 0.19; chance 0.10)",flush=True)
    print("STEP28_DONE")
