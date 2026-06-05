"""Option C: full positive control at 20 seeds, parameterized difficulty. Writes a difficulty-tagged
JSON so diff0 and diff1 do not collide. Determinism fix is in run_positive_control_arm."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import positive_control as pc

DIFFS = {
    "diff0": {"jitter": 0.10, "bg_noise": 0.05, "pos_jitter": 4,  "radius_range": (7, 10)},
    "diff1": {"jitter": 0.18, "bg_noise": 0.10, "pos_jitter": 6,  "radius_range": (6, 11)},
}
ap = argparse.ArgumentParser()
ap.add_argument("--diff", choices=list(DIFFS), required=True)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--epochs", type=int, default=50)
a = ap.parse_args()
pc.DIFFICULTY.update(DIFFS[a.diff])
out = pc.run_positive_control(seeds=a.seeds, n_tasks=3, epochs=a.epochs, device="cuda",
                              eval_inits=8, save=False)
path = os.path.join(pc.RESULTS, f"positive_control_{a.diff}.json")
json.dump(out, open(path, "w"), default=str)
v = out["verdict"]; did = out["did"]
print(f"=== {a.diff} ({a.seeds} seeds) ===")
print("PASS:", out["pass"], "| primary:", v.get("primary_metric"))
print("raw Obar: mean=%.4f p=%.4f dz=%.3f" % (did["mean_delta_obar_R5_minus_R6"], did["p_one_sided_obar_gt_0"], did["cohens_dz_obar"]))
print("DiD inner: mean=%.4f p=%.4f" % (did["mean_delta_R5_minus_R6"], did["p_one_sided_delta_gt_0"]))
print("wrote", path)
