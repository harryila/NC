"""STEP 18 / main-track push — 2nd REAL multi-object dataset (Fashion-MNIST 2-object binding). Tests whether the
oscillator-necessity result generalizes BEYOND MNIST digits to different real content (clothing items). Same
overlapping-2-object binding setup + validated step9 M3-online loop; only the source dataset changes
(MNIST -> FashionMNIST). If R6/R6s >> plainCNN here too, the result is not MNIST-specific (cross-dataset).
Usage: python step18_fashion_multiobj.py --arm R6 --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch
import m2_shapes_construct as shp

def _load_fashion():
    from torchvision import datasets
    ds = datasets.FashionMNIST(root="/root/.fashion", train=True, download=True)
    X = ds.data.numpy().astype(np.float32) / 255.0; Y = ds.targets.numpy()
    return {d: X[Y == d] for d in range(10)}
_OBJ = None
_PAIRS10 = [(0,0),(1,1),(2,2),(3,3),(4,4),(0,1),(2,3),(4,5),(6,7),(8,9)]
shp.N_CLASSES = 10; shp.N_EXP = 5

def _compose(rng, a_img, b_img, canvas=40, out=32):
    H = 28; img = np.zeros((canvas, canvas), np.float32); c = (canvas - H) // 2
    for m in (a_img, b_img):
        px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        sub = img[px:px+H, py:py+H]; img[px:px+H, py:py+H] = np.maximum(sub, m)
    idx = (np.arange(out) * canvas / out).astype(int); img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)

def _gen_split(n_per_class, seed, canvas=40, out=32):
    global _OBJ
    if _OBJ is None: _OBJ = _load_fashion()
    rng = np.random.default_rng(seed); X, Y = [], []
    for cls, (da, db) in enumerate(_PAIRS10):
        A, B = _OBJ[da], _OBJ[db]
        for _ in range(n_per_class):
            X.append(_compose(rng, A[rng.integers(len(A))], B[rng.integers(len(B))], canvas, out)); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64); p = rng.permutation(len(Y)); return X[p], Y[p]

def _experiences(X, y, n_exp=5, n_classes=10):
    k = n_classes // n_exp; out = []
    for e in range(n_exp):
        m = np.isin(y, list(range(e*k, (e+1)*k))); out.append((X[m], y[m]))
    return out

shp._gen_split = _gen_split; shp._experiences = _experiences
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step18_fashion.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["dataset"] = "2obj-fashion"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[fashion2 {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (mnist2: R6=0.65,R6s=0.61,plain=0.19; chance 0.10)", flush=True)
    print("STEP18_DONE")
