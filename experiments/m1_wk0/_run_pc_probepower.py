"""Power-vs-construct disentangle (autonomous probe, fully reversible, no new task / no arc change).
Re-run the EXISTING diff0 + diff1 positive-control tasks but with a MUCH larger CKA probe
(probe_per_class 2 -> 24: ~216 probe rows vs 18) to fix the underpower the interpretation flagged
(control had C(3,2)=3 CKA pairs on 18 rows vs CIFAR's 45 pairs on ~200).

If the marginal diff0 SHARPENS toward significance with the bigger probe -> the weakness was probe-scale
underpower (a measurement artifact), not absence of effect. If it stays flat/null -> the weakness is the
CONSTRUCT (single-shape label-conjunction is not the spatial binding AKOrN helps). Either way it removes
ambiguity. Writes results/positive_control_{diff}_bigprobe.json. 10 seeds (enough to see the direction;
determinism-fixed so reproducible)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import positive_control as pc

DIFFS = {
    "diff0": {"jitter": 0.10, "bg_noise": 0.05, "pos_jitter": 4,  "radius_range": (7, 10)},
    "diff1": {"jitter": 0.18, "bg_noise": 0.10, "pos_jitter": 6,  "radius_range": (6, 11)},
}
ap = argparse.ArgumentParser()
ap.add_argument("--diff", choices=list(DIFFS), required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--probe-per-class", type=int, default=24)
a = ap.parse_args()
pc.DIFFICULTY.update(DIFFS[a.diff])
out = pc.run_positive_control(seeds=a.seeds, n_tasks=3, epochs=a.epochs, device="cuda",
                              eval_inits=8, save=False, probe_per_class=a.probe_per_class)
path = os.path.join(pc.RESULTS, f"positive_control_{a.diff}_bigprobe.json")
json.dump(out, open(path, "w"), default=str)
did = out["did"]
print(f"=== {a.diff} bigprobe (per_class={a.probe_per_class}, {a.seeds} seeds) ===")
print("raw Obar: mean=%.4f p=%.4f dz=%.3f" % (did["mean_delta_obar_R5_minus_R6"], did["p_one_sided_obar_gt_0"], did["cohens_dz_obar"]))
print("DiD inner: mean=%.4f p=%.4f dz=%.3f" % (did["mean_delta_R5_minus_R6"], did["p_one_sided_delta_gt_0"], did["cohens_dz"]))
print("wrote", path)
