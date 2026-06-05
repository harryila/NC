"""M2 BINDING CHECK — is synchrony actually ENGAGED on the multi-object task? (interprets the construct result)

The construct test asks "does R6's phase carry task info". This asks the prerequisite: does R6's phase cluster
into OBJECTS at all, more than R5:no_proj? This is AKOrN's home metric (object grouping). It disambiguates a
construct-test null:
  * R6 binds objects >> R5  -> synchrony IS doing its job here; a task-channel null then means "binding != task
    context channel" (a real, deep result).
  * R6 ~ R5 (neither binds) -> our setup doesn't engage binding (underpowered/wrong codepath), construct test
    inconclusive about the thesis.

METHOD: train R6 / R5:no_proj on the 10-class multi-object Shapes task (so oscillators develop), then on a
held-out probe with INSTANCE MASKS, cluster each image's per-spatial-location oscillator vector (cosine
k-means, k=3 = bg+2 objects) and score the clustering against the downsampled instance mask via Adjusted Rand
Index. Report mean ARI for R6 vs R5:no_proj per layer.

Usage: python m2_shapes_binding.py --seeds 0 1 2 --epochs 30 --layers 1 2 --device cuda
       python m2_shapes_binding.py --demo
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
for p in (M1, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = os.path.join(HERE, "results")

from m2_shapes_construct import _templates, _PAIRS  # reuse exact shape vocabulary  # noqa: E402

ARMS = {"R6": {}, "R5:no_proj": {"variant": "no_proj"}}
N_CLASSES = 10
TIGHT = False   # force the two shapes to OVERLAP near center (segregation genuinely needs binding)


def _make_image_mask(rng, pair, T, canvas=40, out=32):
    """Composite 2 shapes; return (3,out,out) image and (out,out) instance mask (0=bg,1=obj1,2=obj2)."""
    H = next(iter(T.values())).shape[0]
    bg = float(rng.uniform(0.1, 0.6))
    img = np.full((canvas, canvas), bg, np.float32)
    mask = np.zeros((canvas, canvas), np.int64)
    for inst, sid in enumerate(pair, start=1):
        m = T[sid]
        if TIGHT:
            c = (canvas - H) // 2
            px = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
            py = min(max(0, c + int(rng.integers(-4, 5))), canvas - H)
        else:
            px = int(rng.integers(0, canvas - H + 1)); py = int(rng.integers(0, canvas - H + 1))
        sub = img[px:px + H, py:py + H]
        img[px:px + H, py:py + H] = np.where(m > 0, 1.0, sub)
        msub = mask[px:px + H, py:py + H]
        mask[px:px + H, py:py + H] = np.where(m > 0, inst, msub)
    idx = (np.arange(out) * canvas / out).astype(int)
    img = img[np.ix_(idx, idx)]
    mask = mask[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0), mask


def _gen(n_per_class, seed, with_mask=False, canvas=40, out=32):
    rng = np.random.default_rng(seed)
    T = _templates()
    X, y, M = [], [], []
    for cls, pair in enumerate(_PAIRS):
        for _ in range(n_per_class):
            im, mk = _make_image_mask(rng, pair, T, canvas, out)
            X.append(im); y.append(cls); M.append(mk)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64); M = np.asarray(M, np.int64)
    perm = rng.permutation(len(y))
    if with_mask:
        return X[perm], y[perm], M[perm]
    return X[perm], y[perm]


def _downsample_mask(mask, H, W):
    out_idx_r = (np.arange(H) * mask.shape[0] / H).astype(int)
    out_idx_c = (np.arange(W) * mask.shape[1] / W).astype(int)
    return mask[np.ix_(out_idx_r, out_idx_c)]


def _binding_ari(osc_chw, mask, k=3):
    """osc_chw:(C,H,W); mask:(out,out). Cosine k-means over per-location oscillator vectors vs object mask."""
    from h3 import spherical_kmeans, _ari
    C, H, W = osc_chw.shape
    V = osc_chw.reshape(C, H * W).T.astype(np.float64)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    asg = spherical_kmeans(V, k=k, n_iter=20, seed=0)
    md = _downsample_mask(mask, H, W).reshape(-1)
    return float(_ari(asg, md))


def _train_and_bind(rung, rung_kw, Xtr, ytr, Xpr, Mpr, layers, epochs, device, eval_inits, seed, k=3,
                    max_probe=200):
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa: F401
    from avalanche_backbone import LadderClassifier
    from h3 import _seeded as h3_seeded

    base = rung.split(":")[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model = LadderClassifier(base, num_classes=N_CLASSES, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4); crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); loss = crit(model(Xt[idx].to(device)), yt[idx].to(device)); loss.backward(); opt.step()
    n = int(getattr(model.net, "n", 4))
    model.eval()
    aris = {l: [] for l in layers}
    Xp = torch.tensor(Xpr[:max_probe]); Mp = Mpr[:max_probe]
    with torch.no_grad():
        for i in range(0, len(Xp), 100):
            xb = Xp[i:i + 100].to(device)
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
                    aris[l].append(_binding_ari(a[b], Mp[i + b], k=k))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {l: float(np.mean(aris[l])) for l in layers}


def _save(per_seed, layers, epochs, final=False):
    os.makedirs(RESULTS, exist_ok=True)
    paired = [p for p in per_seed if all(a in p for a in ARMS)]
    summary = {}
    for l in layers:
        r6 = [p["R6"][l] for p in paired]; r5 = [p["R5:no_proj"][l] for p in paired]
        d = [a - b for a, b in zip(r6, r5)]
        summary[l] = {"mean_ari_R6": float(np.mean(r6)) if r6 else None,
                      "mean_ari_R5": float(np.mean(r5)) if r5 else None,
                      "mean_delta": float(np.mean(d)) if d else None,
                      "n_pos": int(sum(1 for x in d if x > 0)), "n": len(d)}
    binds = any((summary[l]["mean_delta"] or 0) >= 0.05 for l in layers)
    call = ("SYNCHRONY BINDS (R6 object-clustering > R5) -> binding engaged here"
            if binds else "NO DIFFERENTIAL BINDING (R6 ~ R5 object-clustering)")
    out = {"arms": list(ARMS), "metric": "object_ARI", "epochs": epochs,
           "per_seed": per_seed, "per_layer": {str(k): v for k, v in summary.items()}, "verdict": call}
    json.dump(out, open(os.path.join(RESULTS, "m2_shapes_binding.json"), "w"), indent=2, default=str)
    if final:
        print("\n=== M2 BINDING CHECK — object-ARI from phase clusters (R6 vs R5:no_proj) ===")
        for l in layers:
            s = summary[l]
            print("  L%d: ARI R6=%.4f R5=%.4f  D=%+.4f  n_pos=%d/%d" % (
                l, s["mean_ari_R6"], s["mean_ari_R5"], s["mean_delta"], s["n_pos"], s["n"]))
        print("VERDICT:", call)
        print("BINDING_DONE")
    return out


def run(seeds, epochs=30, layers=(1, 2), device="cuda", eval_inits=4, n_per_class=400, max_probe=200, k=3):
    per_seed = []
    for s in seeds:
        Xtr, ytr = _gen(n_per_class, seed=2000 + s)
        Xpr, ypr, Mpr = _gen(max(40, max_probe // 5), seed=7000 + s, with_mask=True)
        rec = {"seed": s}
        for arm, kw in ARMS.items():
            rec[arm] = _train_and_bind(arm, kw, Xtr, ytr, Xpr, Mpr, list(layers), epochs, device,
                                       eval_inits, s, k=k, max_probe=max_probe)
            print("[seed %d] %s: %s" % (s, arm, " ".join("L%d ARI=%.4f" % (l, rec[arm][l]) for l in layers)), flush=True)
        per_seed.append(rec)
        _save(per_seed, layers, epochs)
        print("[seed %d] saved (%d/%d)" % (s, len(per_seed), len(seeds)), flush=True)
    return _save(per_seed, layers, epochs, final=True)


def _demo():
    X, y, M = _gen(3, seed=0, with_mask=True)
    print("X", X.shape, "mask vals", sorted(set(M[0].reshape(-1).tolist())))
    # synthetic binding sanity: object-structured osc should beat random vs a 2-object mask
    rng = np.random.default_rng(0)
    mask = np.zeros((8, 8), int); mask[:, :4] = 1; mask[:, 4:] = 2
    osc_struct = np.zeros((16, 8, 8), np.float32)
    osc_struct[:, :, :4] = rng.normal(size=(16, 1, 1)); osc_struct[:, :, 4:] = rng.normal(size=(16, 1, 1))
    osc_rand = rng.normal(size=(16, 8, 8)).astype(np.float32)
    mask32 = np.kron(mask, np.ones((4, 4), int))
    print("ARI struct=%.3f rand=%.3f (struct should be higher)" % (
        _binding_ari(osc_struct, mask32, k=3), _binding_ari(osc_rand, mask32, k=3)))
    print("=== BINDING DEMO OK ===")


def main():
    ap = argparse.ArgumentParser(description="M2 binding check: object-ARI of phase clusters, R6 vs R5")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--max-probe", type=int, default=200)
    ap.add_argument("--k-clusters", type=int, default=3)
    ap.add_argument("--tight", action="store_true", help="force the two shapes to overlap (binding required)")
    a = ap.parse_args()
    if a.demo:
        _demo(); return
    global TIGHT
    TIGHT = a.tight
    run(a.seeds, epochs=a.epochs, layers=tuple(a.layers), device=a.device, eval_inits=a.eval_inits,
        n_per_class=a.n_per_class, max_probe=a.max_probe, k=a.k_clusters)


if __name__ == "__main__":
    main()
