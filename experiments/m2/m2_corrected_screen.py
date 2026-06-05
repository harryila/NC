"""M2 CORRECTED SCREEN — the synchrony-channel test with the 4 audit fixes applied.

The prior screens were structurally unable to test the thesis (4-agent audit, 2026-06-05):
  1. WRONG ABLATION: apply_proj=False (R5:no_proj) is NOT synchrony-off (coupling/recurrence/Omega
     still run). FIX -> real ablation = R6 (learned coupling) vs R6s (R6_scrambled: frozen RANDOM
     coupling, apply_proj=True, params+geometry identical) and R5:depthwise (coupling removed).
  2. UNFAIR/UNSTABLE baseline (R5:no_proj bimodal) -> replaced by R6s (param/geometry matched).
  3. CAPTURE bug: averaging gauge-arbitrary oscillator states over eval_inits cancels phase signal.
     FIX -> eval_inits=1 (no averaging).
  4. LOSSY descriptor (8-dim marginal pool) -> use the relational descriptor set + still report marginal.
Construct = multi-object Shapes (binding engaged), generator-free decodability of experience-id (I(T;c)).

PRIMARY CONTRAST: R6 vs R6s (does LEARNED synchrony coupling add task info to the phase context, at
matched params/geometry/capture?). Secondary: R6 vs R5:depthwise (coupling fully removed).

PASS (proceed to the full end-to-end unbypassable phase->theta hypernetwork) iff R6 carries
significantly more task-decodable info than R6s at some layer/descriptor (mean Delta >= +0.05, p<0.05).
Otherwise the corrected null stands.

Reuses m2_shapes_construct for data-gen + per-arm train/capture/decode (arm-agnostic).
Usage: python m2_corrected_screen.py --seeds 0 1 2 3 4 5 --epochs 30 --layers 1 2 --device cuda
"""
import argparse
import json
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
for p in (M1, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = os.path.join(HERE, "results")

import m2_shapes_construct as base   # _gen_split, _experiences, _train_capture_decode, DESCRIPTORS

# arm name -> (ladder rung, rung_kw)
ARMS = {"R6": ("R6", {}), "R6s": ("R6s", {}), "R5:depthwise": ("R5", {"variant": "depthwise"})}
PRIMARY_OFF = "R6s"        # the param/geometry-matched real synchrony-off control
DESCRIPTORS = base.DESCRIPTORS
RESCUE_DELTA = 0.05
ALPHA = 0.05


def _perm1(d):
    d = np.asarray(d, float); n = len(d); obs = d.mean()
    if n == 0:
        return 1.0
    if n <= 18:
        c = t = 0
        for s in product((1, -1), repeat=n):
            t += 1
            if (d * np.asarray(s)).mean() >= obs - 1e-12:
                c += 1
        return c / t
    rng = np.random.default_rng(0)
    s = rng.choice((1, -1), size=(100000, n))
    return float(((s * d).mean(1) >= obs).mean())


def _summ(per_seed, layers):
    paired = [p for p in per_seed if all(a in p for a in ARMS)]
    out = {}
    proceed = []
    for off in [PRIMARY_OFF, "R5:depthwise"]:
        out[off] = {}
        for l in layers:
            out[off][l] = {}
            for d in DESCRIPTORS:
                r6 = [p["R6"][l][d]["cv"] for p in paired]
                ro = [p[off][l][d]["cv"] for p in paired]
                diffs = [a - b for a, b in zip(r6, ro)]
                m = float(np.mean(diffs)) if diffs else None
                p1 = _perm1(diffs) if len(diffs) >= 2 else None
                out[off][l][d] = {"mean_cv_R6": float(np.mean(r6)) if r6 else None,
                                  "mean_cv_off": float(np.mean(ro)) if ro else None,
                                  "mean_delta": m, "p1": p1,
                                  "n_pos": int(sum(1 for x in diffs if x > 0)), "n": len(diffs)}
                if off == PRIMARY_OFF and m is not None and m >= RESCUE_DELTA and p1 is not None and p1 < ALPHA:
                    proceed.append({"layer": l, "descriptor": d, "mean_delta": m, "p1": p1})
    call = ("PROCEED (learned synchrony coupling adds task-channel info vs R6s -> build the end-to-end "
            "unbypassable phase->theta hypernetwork)" if proceed else
            "CORRECTED-NULL (even with real ablation + eval_inits=1 + relational descriptors on a binding "
            "construct, R6 ~ R6s -> synchrony adds no decodable context-channel info)")
    return {"call": call, "proceed_hits": proceed, "summary": out}


def _save(per_seed, layers, epochs, eval_inits, final=False):
    os.makedirs(RESULTS, exist_ok=True)
    v = _summ(per_seed, layers) if any(all(a in p for a in ARMS) for p in per_seed) else {"call": "no paired seeds"}
    out = {"arms": list(ARMS), "primary_off": PRIMARY_OFF, "eval_inits": eval_inits, "epochs": epochs,
           "fixes": "real-ablation(R6 vs R6s/depthwise)+eval_inits1+relational-desc+binding-construct",
           "per_seed": per_seed, "verdict": v}
    json.dump(out, open(os.path.join(RESULTS, "m2_corrected_screen.json"), "w"), indent=2, default=str)
    if final and "summary" in v:
        print("\n=== M2 CORRECTED SCREEN (real ablation, eval_inits=%d) ===" % eval_inits)
        for off in [PRIMARY_OFF, "R5:depthwise"]:
            print("  --- R6 vs %s ---" % off)
            for l in layers:
                for d in DESCRIPTORS:
                    s = v["summary"][off][l][d]
                    print("    L%d %-12s R6=%.3f %s=%.3f  D=%+.4f p1=%s n_pos=%d/%d" % (
                        l, d, s["mean_cv_R6"], off, s["mean_cv_off"], s["mean_delta"],
                        ("%.3f" % s["p1"]) if s["p1"] is not None else "na", s["n_pos"], s["n"]))
        print("VERDICT:", v["call"])
        if v["proceed_hits"]:
            print("PROCEED HITS:", v["proceed_hits"])
        print("CORRECTED_DONE")
    return out


def run(seeds, epochs=30, layers=(1, 2), device="cuda", eval_inits=1, n_per_class=600, max_probe=240):
    per_seed = []
    for s in seeds:
        Xtr, ytr = base._gen_split(n_per_class, seed=1000 + s)
        Xte, yte = base._gen_split(max(80, max_probe // 2), seed=5000 + s)
        exps_tr = base._experiences(Xtr, ytr)
        exps_te = base._experiences(Xte, yte)
        rec = {"seed": s}
        for arm, (rung, kw) in ARMS.items():
            rec[arm] = base._train_capture_decode(rung, kw, exps_tr, exps_te, list(layers), epochs,
                                                  device, eval_inits, s, max_probe=max_probe)
            print("[seed %d] %s: %s" % (s, arm, " ".join(
                "L%d %s" % (l, " ".join("%s=%.3f" % (d, rec[arm][l][d]["cv"]) for d in DESCRIPTORS))
                for l in layers)), flush=True)
        per_seed.append(rec)
        _save(per_seed, layers, epochs, eval_inits)
        print("[seed %d] saved (%d/%d)" % (s, len(per_seed), len(seeds)), flush=True)
    return _save(per_seed, layers, epochs, eval_inits, final=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=1)
    ap.add_argument("--n-per-class", type=int, default=600)
    ap.add_argument("--max-probe", type=int, default=240)
    a = ap.parse_args()
    run(a.seeds, epochs=a.epochs, layers=tuple(a.layers), device=a.device,
        eval_inits=a.eval_inits, n_per_class=a.n_per_class, max_probe=a.max_probe)


if __name__ == "__main__":
    main()
