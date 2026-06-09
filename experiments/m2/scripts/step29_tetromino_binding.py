"""STEP 29 / (c) bridge — TETROMINO-shape binding (richer object shapes than triangle/square/circle). Same
presence-proof 3-object color x tetromino-shape permutation design as step27 (every image has all 3 colors AND
all 3 tetromino shapes -> presence=chance 1/6; only binding color->shape solves it). Tests whether the oscillator
binding result holds with more complex/realistic object shapes. HONEST: a tetromino-SHAPE synthetic stream, NOT
the DeepMind Tetrominoes benchmark.
Usage: python step29_tetromino_binding.py --arm R6 --seeds 0 1 2 3 --device cuda ; --probe for presence test
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np
from itertools import permutations
import m2_shapes_construct as shp

COL = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)
PERMS = list(permutations(range(3)))

def _tetromino_masks(H=14):
    # 3 distinctive tetrominoes on a 4x4 cell grid, scaled to HxH filled blocks
    grids = {
        'T': [(0,0),(0,1),(0,2),(1,1)],          # T-piece
        'L': [(0,0),(1,0),(2,0),(2,1)],          # L-piece
        'S': [(1,0),(1,1),(0,1),(0,2)],          # S-piece
    }
    masks = []
    for name in ('T','L','S'):
        g = np.zeros((4,4), bool)
        for (r,c) in grids[name]: g[r,c] = True
        # scale 4x4 -> HxH (nearest)
        idx = (np.arange(H)*4//H).clip(0,3)
        m = g[np.ix_(idx, idx)]
        masks.append(m)
    return masks
_M = _tetromino_masks()

def _make(cls, rng, canvas=40, out=32):
    perm = PERMS[cls]; H = 14; img = np.full((3,canvas,canvas), 0.15, np.float32); c = (canvas-H)//2
    for j in rng.permutation(3):
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
    import torch, torch.nn.functional as Fnn
    X, y = _gen_split(300, seed=0); Xt = torch.tensor(X)
    col = Xt.amax(dim=(2,3)).numpy(); gray = Xt.mean(1, keepdim=True)
    sh = np.zeros((len(X),3), np.float32)
    for k,m in enumerate(_M):
        t = torch.tensor(m.astype(np.float32))[None,None]; t = Fnn.interpolate(t, size=(8,8), mode='area'); t=t-t.mean()
        sh[:,k] = Fnn.conv2d(gray, t, padding=2).amax(dim=(1,2,3)).numpy()
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), np.concatenate([col,sh],1), y, cv=5).mean())
    print(f"PRESENCE-detector acc = {acc:.3f} (chance 0.167; ~chance => binding-required)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step29_tetromino.json"); ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe: presence_probe(); sys.exit(0)
    import step9_fully_online as s9
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=3, device=a.device); r["task"]="tetromino-binding"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[tetro {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (6cls/3task, chance 0.167)", flush=True)
    print("STEP29_DONE")
