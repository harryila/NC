"""STEP 16 / experiment #1 — REAL-CONTENT MULTI-OBJECT generalization: 2-digit MNIST binding. The key test of
whether the binding-scoped M3 forgetting-bypass holds with REAL object appearances (handwritten digits), not
synthetic shapes. Two MNIST digits composited OVERLAPPING near center (binding required); class = the unordered
digit-pair (10 classes), 5 tasks. Monkeypatches shp._gen_split/_experiences to this dataset, then runs the
validated step9 M3-online loop. Arms R6/R6s/plainCNN. If R6 retains >> R6s/plainCNN here, the binding result
generalizes to real content.
Usage: python step16_multiobj_mnist.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch
import m2_shapes_construct as shp

# ---- load MNIST once, group by digit ----
def _load_mnist():
    from torchvision import datasets
    ds = datasets.MNIST(root="/root/.mnist", train=True, download=True)
    X = ds.data.numpy().astype(np.float32) / 255.0   # (60000,28,28)
    Y = ds.targets.numpy()
    return {d: X[Y == d] for d in range(10)}
_DIG = None
_PAIRS10 = [(0,0),(1,1),(2,2),(3,3),(4,4),(0,1),(2,3),(4,5),(6,7),(8,9)]   # 10 classes
shp.N_CLASSES = 10; shp.N_EXP = 5

def _compose(rng, a_img, b_img, canvas=40, out=32):
    H = 28; img = np.zeros((canvas, canvas), np.float32); c = (canvas - H) // 2
    for m in (a_img, b_img):
        px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)   # overlap near center -> binding required
        py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        sub = img[px:px+H, py:py+H]; img[px:px+H, py:py+H] = np.maximum(sub, m)
    idx = (np.arange(out) * canvas / out).astype(int); img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)

def _gen_split(n_per_class, seed, canvas=40, out=32):
    global _DIG
    if _DIG is None: _DIG = _load_mnist()
    rng = np.random.default_rng(seed); X, Y = [], []
    for cls, (da, db) in enumerate(_PAIRS10):
        Aimgs, Bimgs = _DIG[da], _DIG[db]
        for _ in range(n_per_class):
            a = Aimgs[rng.integers(len(Aimgs))]; b = Bimgs[rng.integers(len(Bimgs))]
            X.append(_compose(rng, a, b, canvas, out)); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64); p = rng.permutation(len(Y))
    return X[p], Y[p]

def _experiences(X, y, n_exp=5, n_classes=10):
    k = n_classes // n_exp; out = []
    for e in range(n_exp):
        cls = list(range(e*k, (e+1)*k)); m = np.isin(y, cls); out.append((X[m], y[m]))
    return out

shp._gen_split = _gen_split; shp._experiences = _experiences
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step16_mnist.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["dataset"] = "2digit-mnist"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[mnist2 {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (TIGHT-shapes: R6=0.95,R6s=0.66,plain=0.11; chance 0.10)", flush=True)
    print("STEP16_DONE")
