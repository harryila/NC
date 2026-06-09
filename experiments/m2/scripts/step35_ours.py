"""STEP 35 ours — our method (clean learned AE trunk, train-only SSL) on the SCALE Tetrominoes benchmark.
Adequate replay: n_anchors = 60*NEXP (the per-task budget that recovered scale degradation earlier).
Usage: NC_SCALE=12 NEXP_SCALE=6 python step35_ours.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step35_scale_data as sd          # patch shp -> SCALE tetrominoes data
import step20_learned_trunk as s20       # (applies leaky patch; overridden below)
import m2_hypernet as H
import m2_shapes_construct as shp

_TRUNK = {}
def _pretrain_trunk_clean(seed, device, n_per_class=300, epochs=15):
    if seed in _TRUNK: return _TRUNK[seed]
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
    enc.eval(); trunk = nn.Sequential(enc, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
    for p in trunk.parameters(): p.requires_grad_(False)
    _TRUNK[seed] = trunk; return trunk
H._build_pieces = lambda arm, device, seed, ctx_dim_target=8: (None, _pretrain_trunk_clean(seed, device), 4, 64)
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step35_ours.json")
    a = ap.parse_args(); NEXP = sd.NEXP
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=NEXP, n_anchors=60*NEXP, device=a.device); r["task"]=f"tetro-scale-NC{sd.NC}-T{NEXP}"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[scaleOURS {a.arm} s{s}] NC={sd.NC} T={NEXP} learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (chance {1.0/sd.NC:.3f})", flush=True)
    print("STEP35_OURS_DONE")
