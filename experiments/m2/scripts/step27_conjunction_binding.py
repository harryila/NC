"""STEP 27 — GENUINE BINDING task (earn the word 'binding'). 3 objects per image, one of each COLOR (R,G,B) and
one of each SHAPE (triangle,square,circle), overlapping near center. Class = the PERMUTATION binding colors to
shapes (which color is which shape); 6 classes (3! perms), 3 tasks. BY CONSTRUCTION every image contains all 3
colors AND all 3 shapes -> PRESENCE is identical across classes -> a presence/bag-of-features model gets CHANCE
(1/6). ONLY binding color->shape solves it (the canonical illusory-conjunction test). If R6 solves this and beats
plainCNN, we EARN 'binding'; if R6 ~ chance too, the method does multi-object but not binding (stay 'multi-object').
Also runs a presence-detector probe to PROVE presence is uninformative here.
Usage: python step27_conjunction_binding.py --arm R6 --seeds 0 1 2 3 --device cuda  ;  --probe  for presence test
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np
from itertools import permutations
import m2_shapes_construct as shp

COL = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)   # R,G,B
PERMS = list(permutations(range(3)))                     # 6 permutations = 6 classes
def _masks(H=14):
    yy, xx = np.mgrid[0:H,0:H]; cx=cy=(H-1)/2.0
    square = np.zeros((H,H), bool); square[2:H-2,2:H-2] = True
    circle = ((yy-cy)**2+(xx-cx)**2) <= (H*0.42)**2
    tri = np.zeros((H,H), bool)
    for i in range(H):
        half=(i/(H-1))*(H/2.0); tri[i, max(0,int(round(cx-half))):int(round(cx+half))+1] = True
    return [tri, square, circle]
_M = _masks()

def _make(cls, rng, canvas=40, out=32):
    perm = PERMS[cls]; H = 14; img = np.full((3,canvas,canvas), 0.15, np.float32); c = (canvas-H)//2
    order = rng.permutation(3)            # random draw order so overlap occlusion isn't class-correlated
    for j in order:                        # object j: COLOR j bound to SHAPE perm[j]
        m = _M[perm[j]]
        px = min(max(0, c+int(rng.integers(-5,6))), canvas-H); py = min(max(0, c+int(rng.integers(-5,6))), canvas-H)
        for ch in range(3):
            sub = img[ch, px:px+H, py:py+H]; img[ch, px:px+H, py:py+H] = np.where(m, COL[j,ch], sub)
    idx = (np.arange(out)*canvas/out).astype(int)
    return img[:, idx][:, :, idx]

shp.N_CLASSES = 6; shp.N_EXP = 3
def _gen_split(n_per_class, seed, canvas=40, out=32):
    rng = np.random.default_rng(seed); X, Y = [], []
    for cls in range(6):
        for _ in range(n_per_class): X.append(_make(cls, rng, canvas, out)); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64); p = rng.permutation(len(Y)); return X[p], Y[p]
def _experiences(X, y, n_exp=3, n_classes=6):
    k = n_classes//n_exp; out=[]
    for e in range(n_exp):
        m = np.isin(y, list(range(e*k,(e+1)*k))); out.append((X[m], y[m]))
    return out
shp._gen_split = _gen_split; shp._experiences = _experiences

def presence_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = _gen_split(300, seed=0)
    # presence features: per-color channel energy (3) + per-shape template max-correlation (3) = 6-dim
    import torch, torch.nn.functional as Fnn
    Xt = torch.tensor(X)
    col_pres = Xt.amax(dim=(2,3)).numpy()          # (N,3) max per color channel -> all ~1 (all colors present)
    gray = Xt.mean(1, keepdim=True)
    shp_pres = np.zeros((len(X),3), np.float32)
    for k,m in enumerate(_M):
        t = torch.tensor(m.astype(np.float32))[None,None]; t = Fnn.interpolate(t, size=(8,8), mode='area'); t=t-t.mean()
        r = Fnn.conv2d(gray, t, padding=2); shp_pres[:,k] = r.amax(dim=(1,2,3)).numpy()
    feats = np.concatenate([col_pres, shp_pres], 1)
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), feats, y, cv=5).mean())
    print(f"PRESENCE-detector (color+shape presence, 6-dim) acc = {acc:.3f}  (chance 1/6=0.167; should be ~chance if binding-required)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step27_conj.json"); ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe: presence_probe(); sys.exit(0)
    import step9_fully_online as s9
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=3, device=a.device); r["task"]="conjunction-binding"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[conj {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (6cls/3task, chance 0.167)", flush=True)
    print("STEP27_DONE")
