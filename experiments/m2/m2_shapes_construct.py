"""M2 CONSTRUCT-MISMATCH TEST — does synchrony open a task-channel on a MULTI-OBJECT (binding) task?

GROUNDING (AKOrN paper, ICLR'25): synchrony binding is demonstrated on OBJECT DISCOVERY over multi-object
scenes (Tetrominoes, dSprites, CLEVR, and the authors' own *Shapes* = 2-4 objects from {triangle, square,
circle, diamond}); classification (single-object CIFAR) was only a robustness sidebar. Our M2 screen used the
single-object CIFAR class-IL codepath -> the construct where synchrony is WEAKEST. This test moves to a
multi-object scene where binding is actually engaged, on our EXACT R6-vs-R5 ladder (single apply_proj flip),
and asks the SAME question: does R6's phase-state carry more decodable task/context info than R5:no_proj?

TASK: each 32x32 image has EXACTLY 2 filled shapes from 4 types -> label = the unordered PAIR (10 classes:
4 same-type + 6 distinct). Identifying the pair requires segregating/binding the two objects. Split the 10
classes into 5 continual experiences (2 classes each), train naive-sequentially, then decode the EXPERIENCE
id (5-way, chance 0.2 -- same regime as the CIFAR screen) from the phase-state, with marginal AND relational
descriptors, R6 vs R5:no_proj.

PRE-SPECIFIED (mirrors m2_relational_probe): RESCUE iff some descriptor shows mean Delta(R6-R5) >= +0.05 at a
layer (clearly beating the CIFAR screen's ~0); else NULL-ROBUST (binding-engaging input still no synchrony
channel -> the M2 null is construct-independent).

Usage (GPU box):
    python m2_shapes_construct.py --seeds 0 1 2 --epochs 30 --layers 1 2 --device cuda
    python m2_shapes_construct.py --demo    # CPU: dataset + pipeline shape sanity
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
for p in (M1, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = os.path.join(HERE, "results")

ARMS = {"R6": {}, "R5:no_proj": {"variant": "no_proj"}}
DESCRIPTORS = ["marginal", "2nd_moment", "coh_eig", "cluster_occ", "spatial4x4"]
RELATIONAL = ["2nd_moment", "coh_eig", "cluster_occ", "spatial4x4"]
RESCUE_DELTA = 0.05
PRIMARY_LAYER = 2
N_CLASSES = 10
N_EXP = 5
TIGHT = False   # if True, force the two shapes to OVERLAP near center (binding genuinely required)


# ----------------------------- pure-numpy multi-object Shapes -----------------------------
def _templates(H=18):
    """4 filled shape masks (HxH) in {triangle, square, circle, diamond} — pure numpy, no skimage."""
    yy, xx = np.mgrid[0:H, 0:H]
    cx = cy = (H - 1) / 2.0
    square = np.zeros((H, H), np.float32); square[3:H - 3, 3:H - 3] = 1.0
    circle = (((yy - cy) ** 2 + (xx - cx) ** 2) <= (H * 0.42) ** 2).astype(np.float32)
    diamond = ((np.abs(yy - cy) + np.abs(xx - cx)) <= H * 0.46).astype(np.float32)
    triangle = np.zeros((H, H), np.float32)                     # apex top-center, widening to base
    for i in range(H):
        half = (i / (H - 1)) * (H / 2.0)
        lo = int(round(cx - half)); hi = int(round(cx + half))
        triangle[i, max(0, lo):min(H, hi + 1)] = 1.0
    return {1: triangle, 2: square, 3: circle, 4: diamond}


_PAIRS = [(1, 1), (2, 2), (3, 3), (4, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]   # 10 classes
_PAIR2CLASS = {p: i for i, p in enumerate(_PAIRS)}


def _make_image(rng, pair, T, canvas=40, out=32):
    """Composite the two shapes of `pair` at random positions on a random-bg canvas, resize to out x out,
    return (3, out, out) float32 in [0,1] (grayscale repeated to 3 channels, matching the CIFAR shape)."""
    H = next(iter(T.values())).shape[0]
    bg = float(rng.uniform(0.1, 0.6))
    img = np.full((canvas, canvas), bg, np.float32)
    for sid in pair:
        m = T[sid]
        if TIGHT:                                   # force overlap near center -> segregation needs binding
            c = (canvas - H) // 2
            px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
            py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        else:
            px = int(rng.integers(0, canvas - H + 1)); py = int(rng.integers(0, canvas - H + 1))
        sub = img[px:px + H, py:py + H]
        img[px:px + H, py:py + H] = np.where(m > 0, 1.0, sub)
    # nearest-neighbor resize canvas->out (dependency-free)
    idx = (np.arange(out) * canvas / out).astype(int)
    img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)


def _gen_split(n_per_class, seed, canvas=40, out=32):
    """Generate a labeled multi-object set; returns (X (N,3,out,out) f32, y (N,) pair-class 0-9)."""
    rng = np.random.default_rng(seed)
    T = _templates()
    X, y = [], []
    for cls, pair in enumerate(_PAIRS):
        for _ in range(n_per_class):
            X.append(_make_image(rng, pair, T, canvas, out))
            y.append(cls)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _experiences(X, y, n_exp=N_EXP, n_classes=N_CLASSES):
    """Class-IL split: experience e holds classes [e*k, (e+1)*k). Returns list of (X_e, y_e)."""
    k = n_classes // n_exp
    out = []
    for e in range(n_exp):
        cls = set(range(e * k, (e + 1) * k))
        mask = np.isin(y, list(cls))
        out.append((X[mask], y[mask]))
    return out


# ----------------------------- relational descriptors (same as m2_relational_probe) -----------------------------
def _sample_descriptors(osc_1, n, k=8):
    from h3 import group_directions, spherical_kmeans
    out = {}
    U = group_directions(osc_1, n=n)
    out["marginal"] = np.concatenate([U.mean(0), (U ** 2).mean(0)])
    M = (U.T @ U) / U.shape[0]
    out["2nd_moment"] = M.flatten()
    out["coh_eig"] = np.linalg.eigvalsh(M)[::-1].copy()
    Uk = U if U.shape[0] <= 1024 else U[np.random.default_rng(0).choice(U.shape[0], 1024, replace=False)]
    asg = spherical_kmeans(Uk, k=k, n_iter=20, seed=0)
    occ = np.bincount(asg, minlength=k).astype(float); occ = occ / max(occ.sum(), 1.0)
    out["cluster_occ"] = np.sort(occ)[::-1].copy()
    a = np.asarray(osc_1, float)[0]
    C, H, W = a.shape; G = C // n
    md = a.reshape(G, n, H, W).mean(0)
    ph = 4
    if H >= ph and W >= ph:
        hs, ws = H // ph, W // ph
        md = md[:, :ph * hs, :ph * ws].reshape(n, ph, hs, ph, ws).mean(axis=(2, 4))
    out["spatial4x4"] = md.flatten()
    return out


# ----------------------------- train (no Avalanche) + capture + decode -----------------------------
def _train_capture_decode(rung, rung_kw, exps_tr, exps_te, layers, epochs, device, eval_inits, seed,
                          max_probe=240, k=8):
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa: F401
    from avalanche_backbone import LadderClassifier
    from h3 import _seeded as h3_seeded
    import m2_primitives as m2

    base = rung.split(":")[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model = LadderClassifier(base, num_classes=N_CLASSES, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    # naive sequential CL over experiences
    for (Xtr, ytr) in exps_tr:
        Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(yt))
            for i in range(0, len(yt), 128):
                idx = perm[i:i + 128]
                xb = Xt[idx].to(device); yb = yt[idx].to(device)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    # capture phase per experience-test-set, descriptors per sample, decode EXPERIENCE id
    n = int(getattr(model.net, "n", 4))
    desc = {l: {d: {} for d in DESCRIPTORS} for l in layers}
    model.eval()
    with torch.no_grad():
        for e, (Xte, _yte) in enumerate(exps_te):
            Xe = torch.tensor(Xte[:max_probe])
            for l in layers:
                for d in DESCRIPTORS:
                    desc[l][d][e] = []
            for i in range(0, len(Xe), 100):
                xb = Xe[i:i + 100].to(device)
                acc = {l: None for l in layers}
                for j in range(eval_inits):
                    with h3_seeded(seed + j):
                        _c, _x, xs, _es = model.net.feature(xb)
                    for l in layers:
                        st = xs[l][-1].detach().float().cpu()
                        acc[l] = st if acc[l] is None else acc[l] + st
                for l in layers:
                    a = (acc[l] / float(eval_inits)).numpy()
                    for b in range(a.shape[0]):
                        dd = _sample_descriptors(a[b:b + 1], n, k=k)
                        for name in DESCRIPTORS:
                            desc[l][name][e].append(dd[name])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = {}
    for l in layers:
        out[l] = {}
        for name in DESCRIPTORS:
            cv, chance = m2.linear_task_decodability(desc[l][name], seed=seed)
            out[l][name] = {"cv": float(cv), "chance": float(chance), "margin": float(cv - chance)}
    return out


def _verdict(per_seed, layers):
    paired = [p for p in per_seed if all(a in p for a in ARMS)]
    summary = {}
    rescued = []
    for l in layers:
        summary[l] = {}
        for d in DESCRIPTORS:
            r6 = [p["R6"][l][d]["cv"] for p in paired]
            r5 = [p["R5:no_proj"][l][d]["cv"] for p in paired]
            diffs = [a - b for a, b in zip(r6, r5)]
            md = float(np.mean(diffs)) if diffs else None
            summary[l][d] = {"mean_cv_R6": float(np.mean(r6)) if r6 else None,
                             "mean_cv_R5": float(np.mean(r5)) if r5 else None,
                             "mean_delta": md, "n_pos": int(sum(1 for x in diffs if x > 0)), "n": len(diffs),
                             "deltas": [round(x, 4) for x in diffs]}
            if d in RELATIONAL and md is not None and md >= RESCUE_DELTA:
                rescued.append({"layer": l, "descriptor": d, "mean_delta": md})
    call = ("RESCUE (multi-object binding opens a synchrony task-channel the CIFAR screen lacked)"
            if rescued else
            "NULL-ROBUST (even on a binding task, synchrony adds no decodable task-channel)")
    return {"call": call, "rescued": rescued, "summary": summary}


def _save(per_seed, layers, epochs, final=False):
    os.makedirs(RESULTS, exist_ok=True)
    v = _verdict(per_seed, layers) if any(all(a in p for a in ARMS) for p in per_seed) else {"call": "no paired seeds"}
    out = {"arms": list(ARMS), "task": "shapes-pair-classIL", "n_classes": N_CLASSES, "n_exp": N_EXP,
           "descriptors": DESCRIPTORS, "rescue_delta": RESCUE_DELTA, "epochs": epochs,
           "per_seed": per_seed, "verdict": {str(k): vv for k, vv in v.items()} if isinstance(v, dict) else v}
    json.dump(out, open(os.path.join(RESULTS, "m2_shapes_construct.json"), "w"), indent=2, default=str)
    if final and isinstance(v, dict) and "summary" in v:
        print("\n=== M2 SHAPES CONSTRUCT — R6 vs R5:no_proj experience-decodability (multi-object) ===")
        for l in layers:
            print("  -- layer %d --" % l)
            for d in DESCRIPTORS:
                s = v["summary"][l][d]
                print("    %-12s R6=%.3f R5=%.3f  D=%+.4f  n_pos=%d/%d" % (
                    d, s["mean_cv_R6"], s["mean_cv_R5"], s["mean_delta"], s["n_pos"], s["n"]))
        print("VERDICT:", v["call"])
        if v["rescued"]:
            print("RESCUED BY:", v["rescued"])
        print("SHAPES_DONE")
    return out


def run(seeds, epochs=30, layers=(1, 2), device="cuda", eval_inits=4, n_per_class=600, max_probe=240):
    per_seed = []
    for s in seeds:
        Xtr, ytr = _gen_split(n_per_class, seed=1000 + s)
        Xte, yte = _gen_split(max(80, max_probe // 2), seed=5000 + s)
        exps_tr = _experiences(Xtr, ytr)
        exps_te = _experiences(Xte, yte)
        rec = {"seed": s}
        for arm, kw in ARMS.items():
            rec[arm] = _train_capture_decode(arm, kw, exps_tr, exps_te, list(layers), epochs, device,
                                             eval_inits, s, max_probe=max_probe)
            print("[seed %d] %s: %s" % (s, arm, " | ".join(
                "L%d %s" % (l, " ".join("%s=%.3f" % (d, rec[arm][l][d]["cv"]) for d in DESCRIPTORS))
                for l in layers)), flush=True)
        per_seed.append(rec)
        _save(per_seed, layers, epochs)
        print("[seed %d] saved (%d/%d)" % (s, len(per_seed), len(seeds)), flush=True)
    return _save(per_seed, layers, epochs, final=True)


def _demo():
    X, y = _gen_split(6, seed=0)
    print("dataset:", X.shape, "labels:", sorted(set(y.tolist())), "(expect 10 classes, X ~ (60,3,32,32))")
    exps = _experiences(X, y)
    print("experiences:", [(xe.shape[0], sorted(set(ye.tolist()))) for xe, ye in exps])
    d = _sample_descriptors(np.random.randn(1, 128, 8, 8).astype(np.float32), n=4)
    print("descriptor dims:", {k: v.size for k, v in d.items()})
    print("=== SHAPES CONSTRUCT DEMO OK ===")


def main():
    ap = argparse.ArgumentParser(description="M2 construct-mismatch test on multi-object Shapes")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    ap.add_argument("--n-per-class", type=int, default=600)
    ap.add_argument("--max-probe", type=int, default=240)
    ap.add_argument("--tight", action="store_true", help="force the two shapes to overlap (binding required)")
    a = ap.parse_args()
    if a.demo:
        _demo(); return
    global TIGHT
    TIGHT = a.tight
    run(a.seeds, epochs=a.epochs, layers=tuple(a.layers), device=a.device,
        eval_inits=a.eval_inits, n_per_class=a.n_per_class, max_probe=a.max_probe)


if __name__ == "__main__":
    main()
