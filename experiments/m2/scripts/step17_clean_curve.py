"""STEP 17 / experiment #2 (confound-controlled) — CLEAN binding-difficulty curve. The step15 curve confounded
overlap with POSITION VARIANCE (jitter scaled with 1-overlap; ov=1.0 => zero jitter => degenerate). Here we
DECOUPLE them: the PAIR gets a random absolute position (CONSTANT jitter across all overlap levels), and the two
objects are placed at a relative offset = (1-OVERLAP)*max_sep from each other. So position-variance is fixed;
ONLY inter-object overlap varies -> a clean 'binding demand' axis. Runs M3-online R6 vs R6s.
Usage: python step17_clean_curve.py --overlap 0.5 --arm R6 --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np
import m2_shapes_construct as shp

shp._CLEAN_OVERLAP = 0.5
def _make_image_clean(rng, pair, T, canvas=40, out=32):
    H = next(iter(T.values())).shape[0]
    bg = float(rng.uniform(0.1, 0.6)); img = np.full((canvas, canvas), bg, np.float32)
    ov = shp._CLEAN_OVERLAP
    max_sep = (canvas - H)            # max center-to-center separation that keeps both on canvas
    sep = (1.0 - ov) * max_sep / 2.0  # half-offset each side of the pair-center
    # pair center jitter: CONSTANT range regardless of overlap (decouples position variance from overlap)
    jrange = (canvas - H) // 4
    bx = (canvas - H) // 2 + int(rng.integers(-jrange, jrange + 1))
    by = (canvas - H) // 2 + int(rng.integers(-jrange, jrange + 1))
    offs = [(-sep, 0.0), (+sep, 0.0)] if rng.random() < 0.5 else [(0.0, -sep), (0.0, +sep)]
    for sid, (ox, oy) in zip(pair, offs):
        m = T[sid]
        px = int(min(max(0, bx + ox), canvas - H)); py = int(min(max(0, by + oy), canvas - H))
        sub = img[px:px + H, py:py + H]; img[px:px + H, py:py + H] = np.where(m > 0, 1.0, sub)
    idx = (np.arange(out) * canvas / out).astype(int); img = img[np.ix_(idx, idx)]
    return np.repeat(img[None], 3, axis=0)
shp._make_image = _make_image_clean
import step9_fully_online as s9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, required=True); ap.add_argument("--arm", default="R6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="step17_clean.json")
    a = ap.parse_args()
    shp._CLEAN_OVERLAP = a.overlap
    path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, device=a.device); r["overlap"] = a.overlap; recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        print(f"[clean ov={a.overlap} {a.arm} s{s}] final={r['final_acc']:.3f} forget={r['forgetting']:.3f}", flush=True)
    print("STEP17_DONE")
