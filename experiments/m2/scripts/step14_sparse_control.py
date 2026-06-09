"""STEP 14 / May-30 open item — SPARSE-ACTIVATION CONTROL ('synchrony vs just sparsity'). The May-30 de-risking
named this non-negotiable: is AKOrN's benefit beyond what sparse/structured representations (Elephant/k-WTA)
already give? Tested in the STRONGEST setting (TIGHT genuine-binding M3-online), where if sparsity-alone were the
mechanism it would show. A non-oscillator CNN context generator with k-WTA SPARSE activations replaces the dense
plainCNN; everything else identical (reuses the validated step9 loop via monkey-patch). If sparse-CNN collapses
like dense plainCNN (and << R6), synchrony != sparsity -> thesis safe. If sparse-CNN retains like R6 -> it's
sparsity, not synchrony.
Usage: python step14_sparse_control.py --seeds 0 1 2 3 4 5 6 7 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import torch, torch.nn as nn
import m2_shapes_construct as shp
shp.TIGHT = True   # strongest binding setting
import step9_fully_online as s9
assert shp.TIGHT is True

class SparseCNNContext(nn.Module):
    """Non-oscillator CNN with k-WTA SPARSE activations (keep top-frac per feature map) -> 2n-dim context.
    The sparse-representation control (Elephant/k-WTA spirit)."""
    def __init__(self, ctx_dim, seed, frac=0.2):
        super().__init__(); torch.manual_seed(20_000 + seed); self.frac = frac
        self.c1 = nn.Conv2d(3,32,5,2,2); self.c2 = nn.Conv2d(32,64,3,2,1); self.c3 = nn.Conv2d(64,64,3,1,1)
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(64, ctx_dim)
    def _kwta(self, x):
        # keep top-frac of channels per spatial location (sparse code), zero the rest
        B, C = x.shape[0], x.shape[1]
        k = max(1, int(C * self.frac))
        thr = x.kthvalue(C - k + 1, dim=1, keepdim=True).values
        return x * (x >= thr).float()
    def forward(self, x):
        x = self._kwta(torch.relu(self.c1(x)))
        x = self._kwta(torch.relu(self.c2(x)))
        x = self._kwta(torch.relu(self.c3(x)))
        return self.fc(self.pool(x).flatten(1))

# monkey-patch: run the EXACT step9 plainCNN path but with the sparse generator
s9.PlainCNNContext = SparseCNNContext

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="step14_sparse.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run("plainCNN", s, device=a.device); r["arm"] = "sparseCNN"; r["tight"] = True; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[TIGHT sparseCNN s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forgetting={r['forgetting']:.3f}  (TIGHT: R6=0.96, plainCNN-dense=0.11; chance 0.10)", flush=True)
    print("STEP14_DONE")
