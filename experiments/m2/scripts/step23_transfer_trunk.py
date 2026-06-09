"""STEP 23 — CLEAN learned-trunk (transfer-pretrained, removes the leak caveat). step20's AE was pretrained on
the SAME shapes -> features correlated with class -> plainCNN rose to 0.41 (partial unbypass leak). Here the AE
is pretrained SELF-SUPERVISED on a DIFFERENT dataset (CIFAR-10, natural images), frozen, and used as the trunk
for TIGHT shapes. The trunk is realistic (learned natural features) but TRULY task-agnostic for shapes (never saw
them) -> should NOT leak (plainCNN should stay LOW ~ random-trunk's 0.11). If R6 still bypasses (R6>>R6s/plainCNN)
with a clean realistic trunk, the result is airtight: realistic features AND unbypassable.
Usage: python step23_transfer_trunk.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import m2_shapes_construct as shp
shp.TIGHT = True
import m2_hypernet as H

_CIFAR = None
def _load_cifar():
    global _CIFAR
    if _CIFAR is not None: return _CIFAR
    from torchvision import datasets
    ds = datasets.CIFAR10(root="/root/.cifar10tv", train=True, download=True)
    X = ds.data.astype(np.float32) / 255.0          # (50000,32,32,3)
    _CIFAR = np.transpose(X, (0, 3, 1, 2))          # (N,3,32,32)
    return _CIFAR

_TRUNK_CACHE = {}
def _pretrain_trunk(seed, device, n_imgs=6000, epochs=15):
    if seed in _TRUNK_CACHE: return _TRUNK_CACHE[seed]
    C = _load_cifar(); rng = np.random.RandomState(seed); idx = rng.permutation(len(C))[:n_imgs]
    Xt = torch.tensor(C[idx], dtype=torch.float32)
    torch.manual_seed(40_000 + seed)
    enc = nn.Sequential(nn.Conv2d(3,32,5,2,2), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU()).to(device)
    dec = nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), nn.ReLU(),
                        nn.ConvTranspose2d(32,3,5,2,2,output_padding=1), nn.Sigmoid()).to(device)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3); mse = nn.MSELoss()
    enc.train(); dec.train()
    for ep in range(epochs):
        pr = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            xb = Xt[pr[i:i+128]].to(device); opt.zero_grad(); loss = mse(dec(enc(xb)), xb); loss.backward(); opt.step()
    enc.eval()
    trunk = nn.Sequential(enc, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
    for p in trunk.parameters(): p.requires_grad_(False)
    _TRUNK_CACHE[seed] = trunk; return trunk

def _patched_build(arm, device, seed, ctx_dim_target=8):
    return (None, _pretrain_trunk(seed, device), 4, 64)
H._build_pieces = _patched_build
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step23_transfer.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["trunk"] = "cifar-pretrained-AE"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[transfer-trunk {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (same-data-trunk: R6 0.95,R6s 0.85,plain 0.41; random-trunk plain 0.11)", flush=True)
    print("STEP23_DONE")
