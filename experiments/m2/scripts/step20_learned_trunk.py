"""STEP 20 / main-track push — LEARNED-then-FROZEN realistic backbone (vs frozen-RANDOM). Addresses the reviewer
criticism that a random trunk is unrealistic. The trunk must stay TASK-AGNOSTIC to preserve unbypassability, so
we pretrain it SELF-SUPERVISED (conv autoencoder reconstruction, NO labels) on the dataset images, freeze the
encoder, and use it as the trunk. Then run M3-online R6/R6s/plainCNN. UNBYPASSABILITY CHECK: plainCNN must still
collapse (if a learned trunk leaks task info, plainCNN would suddenly retain -> trunk is solving it, comparison
confounded). Tested on TIGHT shapes (cleanest synchrony setting). If R6>>R6s/plainCNN still holds with a learned
trunk, the result is not an artifact of random features.
Usage: python step20_learned_trunk.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import m2_shapes_construct as shp
shp.TIGHT = True   # cleanest synchrony setting
import m2_hypernet as H

_TRUNK_CACHE = {}
def _pretrain_trunk(seed, device, n_per_class=300, epochs=15):
    if seed in _TRUNK_CACHE: return _TRUNK_CACHE[seed]
    X, _ = shp._gen_split(n_per_class, seed=9999 + seed)   # unlabeled images for SSL
    Xt = torch.tensor(X, dtype=torch.float32)
    torch.manual_seed(30_000 + seed)
    enc = nn.Sequential(nn.Conv2d(3,32,5,2,2), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU()).to(device)  # ->(B,64,8,8)
    dec = nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), nn.ReLU(),
                        nn.ConvTranspose2d(32,3,5,2,2,output_padding=1), nn.Sigmoid()).to(device)        # ->(B,3,32,32)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3); mse = nn.MSELoss()
    enc.train(); dec.train()
    for ep in range(epochs):
        pr = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            xb = Xt[pr[i:i+128]].to(device); opt.zero_grad()
            loss = mse(dec(enc(xb)), xb); loss.backward(); opt.step()
    enc.eval()
    trunk = nn.Sequential(enc, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
    for p in trunk.parameters(): p.requires_grad_(False)
    _TRUNK_CACHE[seed] = trunk
    return trunk

# monkeypatch H._build_pieces so step9's trunk (slot [1]) = the learned-frozen encoder
_orig_build = H._build_pieces
def _patched_build(arm, device, seed, ctx_dim_target=8):
    return (None, _pretrain_trunk(seed, device), 4, 64)
H._build_pieces = _patched_build
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step20_learned.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["trunk"] = "learned-frozen-AE"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[learned-trunk {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (random-trunk TIGHT: R6=0.95,R6s=0.66,plain=0.11; chance 0.10)", flush=True)
    print("STEP20_DONE")
