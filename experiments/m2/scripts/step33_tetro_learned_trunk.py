"""STEP 33 / MAIN-TRACK GATE 0 (CRUX) — does a LEARNED realistic backbone close the DER gap on REAL Tetrominoes?
Our frozen-RANDOM trunk capped us at 0.59 (DER 0.87). Here we give our method a self-supervised AE trunk
(task-agnostic, frozen -> preserves unbypassability) pretrained on tetrominoes images, then run R6/R6s/plainCNN.
CLEAN: SSL pretraining uses TRAIN-PARTITION images ONLY (seed<4000) -> NO test-pixel leak (fixed from step20's
9999+seed which routed to the test partition under step31's data). GATE: learned-trunk R6/R6s approach DER ~0.87.
Usage: python step33_tetro_learned_trunk.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step31_tetrominoes_task as s31     # 1) patch shp -> REAL tetrominoes data (red-shape, 6cls/3task, disjoint)
import step20_learned_trunk as s20        # 2) imports + applies step20's (leaky) trunk patch -- we OVERRIDE it below
import m2_hypernet as H
import m2_shapes_construct as shp

_TRUNK = {}
def _pretrain_trunk_clean(seed, device, n_per_class=300, epochs=15):
    if seed in _TRUNK: return _TRUNK[seed]
    X, _ = shp._gen_split(n_per_class, seed=2000 + seed)   # TRAIN partition (2000<4000) -> NO test leak
    Xt = torch.tensor(X, dtype=torch.float32)
    torch.manual_seed(30_000 + seed)
    enc = nn.Sequential(nn.Conv2d(3,32,5,2,2), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU()).to(device)
    dec = nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), nn.ReLU(),
                        nn.ConvTranspose2d(32,3,5,2,2,output_padding=1), nn.Sigmoid()).to(device)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3); mse = nn.MSELoss()
    enc.train(); dec.train()
    for ep in range(epochs):
        pr = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            xb = Xt[pr[i:i+128]].to(device); opt.zero_grad(); loss = mse(dec(enc(xb)), xb); loss.backward(); opt.step()
    enc.eval(); trunk = nn.Sequential(enc, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
    for p in trunk.parameters(): p.requires_grad_(False)
    _TRUNK[seed] = trunk; return trunk

# OVERRIDE step20's patch with the clean (train-only SSL) trunk
def _clean_build(arm, device, seed, ctx_dim_target=8):
    return (None, _pretrain_trunk_clean(seed, device), 4, 64)
H._build_pieces = _clean_build
s20._pretrain_trunk = _pretrain_trunk_clean   # in case anything calls step20's directly
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step33_tetro_learned.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=3, device=a.device); r["task"]="tetro-real-learnedtrunk-clean"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[tetroLEARN {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (frozen-random R6 0.59; DER 0.87; chance 0.167)", flush=True)
    print("STEP33_DONE")
