"""STEP 19 / main-track push (SCALE) — longer CL sequence + more classes on real multi-object MNIST: 20 classes,
10 tasks (vs the 10-class/5-task default). Directly addresses the 'small scale / only 5 tasks' main-track
concern. 2-digit overlapping MNIST, 20 digit-pair classes, 10 sequential tasks. Validated step9 M3-online loop
(n_tasks=10). Arms R6/R6s/plainCNN. If R6/R6s >> plainCNN over a 2x-longer sequence, the bypass holds at scale.
Usage: python step19_scale_mnist.py --arm R6 --seeds 0 1 2 3 --device cuda
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
_PAIRS20 = [(d, d) for d in range(10)] + [(0,1),(2,3),(4,5),(6,7),(8,9),(1,2),(3,4),(5,6),(7,8),(0,9)]   # 20
shp.N_CLASSES = 20; shp.N_EXP = 10

def _compose(rng, a_img, b_img, canvas=40, out=32):
    H = 28; img = np.zeros((canvas, canvas), np.float32); c = (canvas - H) // 2
    for m in (a_img, b_img):
        px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        sub = img[px:px+H, py:py+H]; img[px:px+H, py:py+H] = np.maximum(sub, m)
    idx = (np.arange(out) * canvas / out).astype(int); img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)

def _gen_split(n_per_class, seed, canvas=40, out=32):
    global _DIG
    if _DIG is None: _DIG = _load_mnist()
    rng = np.random.default_rng(seed); X, Y = [], []
    for cls, (da, db) in enumerate(_PAIRS20):
        A, B = _DIG[da], _DIG[db]
        for _ in range(n_per_class):
            X.append(_compose(rng, A[rng.integers(len(A))], B[rng.integers(len(B))], canvas, out)); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64); p = rng.permutation(len(Y)); return X[p], Y[p]

def _experiences(X, y, n_exp=10, n_classes=20):
    k = n_classes // n_exp; out = []
    for e in range(n_exp):
        m = np.isin(y, list(range(e*k, (e+1)*k))); out.append((X[m], y[m]))
    return out

shp._gen_split = _gen_split; shp._experiences = _experiences
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step21_bigreplay.json")
    a = ap.parse_args()
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=10, n_anchors=600, device=a.device); r["scale"] = "20cls-10task"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[scale20 {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (10task chance 0.05; 5task-mnist: R6=0.65,plain=0.19)", flush=True)
    print("STEP19_DONE")
