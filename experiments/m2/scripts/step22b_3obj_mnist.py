"""STEP 22 (spare-time) — 3-OBJECT MNIST: does the synchrony effect GROW with #objects (binding hypothesis)?
2 class-digits + 1 random DISTRACTOR digit overlapping; class is still the digit-pair. More objects to segregate
-> more binding demand. Prediction: R6-R6s grows vs 2-object (+0.038). Validated step9 M3-online loop.
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch
import m2_shapes_construct as shp

def _load_mnist():
    from torchvision import datasets
    ds = datasets.MNIST(root="/root/.mnist", train=True, download=True)
    X = ds.data.numpy().astype(np.float32) / 255.0; Y = ds.targets.numpy()
    return {d: X[Y == d] for d in range(10)}
_DIG = None
_PAIRS10 = [(0,0),(1,1),(2,2),(3,3),(4,4),(0,1),(2,3),(4,5),(6,7),(8,9)]
shp.N_CLASSES = 10; shp.N_EXP = 5

def _compose3(rng, imgs, canvas=40, out=32):
    H = 28; img = np.zeros((canvas, canvas), np.float32); c = (canvas - H) // 2
    for m in imgs:
        px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        sub = img[px:px+H, py:py+H]; img[px:px+H, py:py+H] = np.maximum(sub, m)
    idx = (np.arange(out) * canvas / out).astype(int); img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)

def _gen_split(n_per_class, seed, canvas=40, out=32):
    global _DIG
    if _DIG is None: _DIG = _load_mnist()
    rng = np.random.default_rng(seed); X, Y = [], []
    for cls, (da, db) in enumerate(_PAIRS10):
        A, B = _DIG[da], _DIG[db]
        for _ in range(n_per_class):
            dd = _DIG[rng.integers(10)]
            imgs = [A[rng.integers(len(A))], B[rng.integers(len(B))], dd[rng.integers(len(dd))]]
            X.append(_compose3(rng, imgs, canvas, out)); Y.append(cls)
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
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step22_3obj.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["dataset"] = "3obj-mnist"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[mnist3 {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (2obj-mnist: R6-R6s +0.038; chance 0.10)", flush=True)
    print("STEP22_DONE")
