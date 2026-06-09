"""STEP 35 / GATE 1 — SCALE benchmark on REAL Tetrominoes. Same red-object pipeline as step31 but FINER shape
classes (class = red object's shape value binned to NC_SCALE groups) and a LONGER task sequence (NEXP tasks),
class-incremental. Configurable via env: NC_SCALE (default 12), NEXP_SCALE (default 6). Disjoint train/test pool.
Imported by step35_ours.py (learned-trunk R6/R6s/plainCNN) and step35_baselines.py (DER/Replay/Naive).
"""
import os, numpy as np
import m2_shapes_construct as shp
import step30_tetrominoes_load as t30

NC = int(os.environ.get("NC_SCALE", "12"))
NEXP = int(os.environ.get("NEXP_SCALE", "6"))
_POOL = {"Xtr": None}

def _build_pool(n_load=40000):
    if _POOL["Xtr"] is not None: return
    imgs, shapes, colors, vis = t30.load("/root/tetrominoes_train.tfrecords", n_load)
    X, Y = [], []
    for i in range(len(imgs)):
        c = colors[i]; sh = shapes[i].reshape(-1); v = vis[i].reshape(-1)
        is_red = (c[:,0] > 0.9) & (c[:,1] < 0.1) & (c[:,2] < 0.1) & (v > 0.5)
        if is_red.sum() != 1: continue
        red_shape = int(sh[np.where(is_red)[0][0]])
        cls = min(NC-1, red_shape * NC // 19)          # 19 shape-orientations -> NC groups
        im = imgs[i].astype(np.float32) / 255.0
        idx = (np.arange(32) * 35 // 32).clip(0, 34)
        im = im[np.ix_(idx, idx)].transpose(2, 0, 1)
        X.append(im); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    rng = np.random.default_rng(12345); perm = rng.permutation(len(Y)); cut = int(0.7*len(Y))
    tr, te = perm[:cut], perm[cut:]
    _POOL["Xtr"], _POOL["ytr"] = X[tr], Y[tr]; _POOL["Xte"], _POOL["yte"] = X[te], Y[te]
    print(f"SCALE pool NC={NC} NEXP={NEXP}: train {len(tr)} / test {len(te)} | train counts {np.bincount(Y[tr], minlength=NC).tolist()}", flush=True)

shp.N_CLASSES = NC; shp.N_EXP = NEXP
def _gen_split(n_per_class, seed):
    _build_pool()
    test = seed >= 4000
    X = _POOL["Xte"] if test else _POOL["Xtr"]; Y = _POOL["yte"] if test else _POOL["ytr"]
    rng = np.random.default_rng(seed); xs, ys = [], []
    for cls in range(NC):
        idx = np.where(Y == cls)[0]; rng.shuffle(idx); idx = idx[:n_per_class]
        xs.append(X[idx]); ys.append(Y[idx])
    X2 = np.concatenate(xs); Y2 = np.concatenate(ys); p = rng.permutation(len(Y2)); return X2[p], Y2[p]
def _experiences(X, y, n_exp=None, n_classes=None):
    n_exp = n_exp or NEXP; n_classes = n_classes or NC; k = n_classes // n_exp; out = []
    for e in range(n_exp):
        m = np.isin(y, list(range(e*k, (e+1)*k))); out.append((X[m], y[m]))
    return out
shp._gen_split = _gen_split; shp._experiences = _experiences
