"""STEP 31 / (c) — REAL DeepMind Tetrominoes binding-CL task. Class = shape-group (6) of the RED tetromino;
must LOCATE the red object and read its shape (distractors of other colors make a color-blind shape-presence
model insufficient => binding-flavored). 6 classes / 3 tasks, class-incremental. Honest: presence is not provably
chance (real random data), so we MEASURE the color-blind shape-histogram probe (leak quantification). Runs
R6/R6s/plainCNN through the SAME M3-online loop on REAL object-centric images (35->32).
Usage: python step31_tetrominoes_task.py --arm R6 --seeds 0 1 2 3 --device cuda ; --probe ; --smoke
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np
import m2_shapes_construct as shp
import step30_tetrominoes_load as t30

_POOL = {"Xtr": None}   # DISJOINT train/test partitions (avoid finite-pool train/test contamination)
def _build_pool(n_load=40000):
    if _POOL["Xtr"] is not None: return
    imgs, shapes, colors, vis = t30.load("/root/tetrominoes_train.tfrecords", n_load)  # imgs (N,35,35,3) uint8
    X, Y = [], []
    for i in range(len(imgs)):
        c = colors[i]; sh = shapes[i].reshape(-1); v = vis[i].reshape(-1)
        # red object(s): R>0.9, G<0.1, B<0.1, and visible, and not background (background is black [0,0,0])
        is_red = (c[:,0] > 0.9) & (c[:,1] < 0.1) & (c[:,2] < 0.1) & (v > 0.5)
        if is_red.sum() != 1: continue          # EXACTLY one red object -> unambiguous label
        red_shape = sh[np.where(is_red)[0][0]]
        cls = int(min(5, int(red_shape) * 6 // 20))   # bin 19 shape-orientations -> 6 groups
        # resize 35->32 (nearest) + CHW + [0,1]
        im = imgs[i].astype(np.float32) / 255.0
        idx = (np.arange(32) * 35 // 32).clip(0, 34)
        im = im[np.ix_(idx, idx)].transpose(2, 0, 1)
        X.append(im); Y.append(cls)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    # deterministic DISJOINT 70/30 train/test partition (no example in both)
    rng = np.random.default_rng(12345); perm = rng.permutation(len(Y)); cut = int(0.7*len(Y))
    tr, te = perm[:cut], perm[cut:]
    _POOL["Xtr"], _POOL["ytr"] = X[tr], Y[tr]; _POOL["Xte"], _POOL["yte"] = X[te], Y[te]
    print(f"pool: train {len(tr)} / test {len(te)} | train class counts = {np.bincount(Y[tr], minlength=6).tolist()}", flush=True)

shp.N_CLASSES = 6; shp.N_EXP = 3
def _gen_split(n_per_class, seed):
    _build_pool()
    # step9 convention: train uses seed=1000+s, test uses seed=5000+s -> route to the right DISJOINT partition
    test = seed >= 4000
    X = _POOL["Xte"] if test else _POOL["Xtr"]; Y = _POOL["yte"] if test else _POOL["ytr"]
    rng = np.random.default_rng(seed); xs, ys = [], []
    for cls in range(6):
        idx = np.where(Y == cls)[0]; rng.shuffle(idx); idx = idx[:n_per_class]
        xs.append(X[idx]); ys.append(Y[idx])
    X2 = np.concatenate(xs); Y2 = np.concatenate(ys); p = rng.permutation(len(Y2)); return X2[p], Y2[p]
def _experiences(X, y, n_exp=3, n_classes=6):
    k = n_classes // n_exp; out = []
    for e in range(n_exp):
        m = np.isin(y, list(range(e*k, (e+1)*k))); out.append((X[m], y[m]))
    return out
shp._gen_split = _gen_split; shp._experiences = _experiences

def presence_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    _build_pool()
    # color-BLIND shape-histogram leak: per-image, histogram of shape-group over ALL objects (ignore color) -> label
    imgs, shapes, colors, vis = t30.load("/root/tetrominoes_train.tfrecords", 14000)
    feats, labs = [], []
    for i in range(len(imgs)):
        c = colors[i]; sh = shapes[i].reshape(-1); v = vis[i].reshape(-1)
        is_red = (c[:,0]>0.9)&(c[:,1]<0.1)&(c[:,2]<0.1)&(v>0.5)
        if is_red.sum()!=1: continue
        cls = int(min(5, int(sh[np.where(is_red)[0][0]])*6//20))
        h = np.zeros(6)
        for o in range(len(v)):
            if v[o]>0.5 and not (c[o,0]<0.1 and c[o,1]<0.1 and c[o,2]<0.1):  # skip black bg
                h[min(5, int(sh[o])*6//20)] += 1
        feats.append(h); labs.append(cls)
    feats = np.array(feats); labs = np.array(labs)
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), feats, labs, cv=5).mean())
    print(f"PRESENCE leak (color-blind shape-histogram -> red-shape) acc = {acc:.3f} (chance 0.167); lower => more binding-required", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step31_tetro_real.json")
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: _build_pool(); sys.exit(0)
    if a.probe: presence_probe(); sys.exit(0)
    import step9_fully_online as s9
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=3, device=a.device); r["task"]="tetrominoes-real"; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[tetroREAL {a.arm} s{s}] learn={r['learn_acc']:.3f} final={r['final_acc']:.3f} forget={r['forgetting']:.3f}  (6cls/3task, chance 0.167)", flush=True)
    print("STEP31_DONE")
