"""
Stage-3 analysis: turn results/*.json into the decisive R6-R5 synchrony increment + the gate call.

The stats here (exact paired sign-flip permutation test, TOST equivalence, paired Cohen's d) are
real and runnable on CPU (numpy/scipy) — `python analyze.py --demo` exercises them on synthetic data.
The results-loading path has ONE TODO: the exact Avalanche metric key for class-IL forgetting,
which you confirm during the Wk-0 Split-MNIST run.
"""
import argparse
import glob
import json
import os
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# ----------------------------- statistics (runnable now) -----------------------------
def paired_permutation_p(diffs, n_perm=200_000, seed=0):
    """Two-sided exact sign-flip test if n<=22, else Monte-Carlo. diffs = per-seed (B - A)."""
    d = np.asarray(diffs, float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 22:
        cnt = tot = 0
        for signs in product((1, -1), repeat=n):
            tot += 1
            if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
                cnt += 1
        return cnt / tot
    rng = np.random.default_rng(seed)
    signs = rng.choice((1, -1), size=(n_perm, n))
    return float((np.abs((signs * d).mean(1)) >= obs).mean())


def tost(diffs, margin):
    """Two one-sided t-tests. Returns p; equivalent within ±margin if p < alpha."""
    from scipy import stats
    d = np.asarray(diffs, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((m - (-margin)) / se, n - 1)   # H0: mean <= -margin
    p_upper = stats.t.cdf((m - margin) / se, n - 1)     # H0: mean >=  margin
    return float(max(p_lower, p_upper))


def paired_cohens_d(diffs):
    d = np.asarray(diffs, float)
    return float(d.mean() / d.std(ddof=1))


def decide(diffs, delta_g, delta_e, alpha=0.05):
    """diffs = per-seed (R6 - R5) on the primary endpoint (lower forgetting is better,
    so a BENEFICIAL synchrony effect is diffs < 0; pass diffs already signed that way)."""
    p_perm = paired_permutation_p(diffs)
    d = paired_cohens_d(diffs)
    p_eq = tost(diffs, delta_e)
    mean = float(np.mean(diffs))
    if (mean <= -delta_g or abs(d) >= 0.8) and p_perm < alpha:
        call = "GREENLIGHT (synchrony reduces forgetting)"
    elif p_eq < alpha:
        call = "PIVOT-A (synchrony ≈ geometry — equivalence)"
    else:
        call = "INCONCLUSIVE (add seeds)"
    return {"mean_R6_minus_R5": mean, "cohens_d": d, "perm_p": p_perm,
            "tost_p": p_eq, "call": call}


# ----------------------------- results loading -----------------------------
def _forgetting(metrics):
    # Confirmed in the Wk-0 Split-CIFAR-100 e2e run: the class-IL stream-forgetting key is
    # "StreamForgetting/eval_phase/test_stream". Prefer it exactly; the dict also contains
    # per-experience ExperienceForgetting/* keys we must NOT pick up.
    # Avalanche reports forgetting as a FRACTION in [0,1]; the pre-registered SESOI/Δg, Δe and the
    # --demo are in percentage POINTS. Return points (×100) so the gate thresholds are comparable.
    if isinstance(metrics, dict):
        exact = "StreamForgetting/eval_phase/test_stream"
        if isinstance(metrics.get(exact), (int, float)):
            return 100.0 * float(metrics[exact])
        for k, v in metrics.items():  # fallback: any stream-level forgetting scalar
            if "streamforgetting" in k.lower() and isinstance(v, (int, float)):
                return 100.0 * float(v)
        for k, v in metrics.items():
            if "forget" in k.lower() and isinstance(v, (int, float)):
                return 100.0 * float(v)
    return None


def load(primary_scenario="class", primary_nexp=10):
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        if os.path.basename(f) == "sparsity_target.json":
            continue
        r = json.load(open(f))
        if r.get("scenario") == primary_scenario and r.get("nexp") == primary_nexp:
            runs.setdefault(r["rung"], {})[r["seed"]] = _forgetting(r.get("metrics", {}))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run the stats on synthetic data")
    ap.add_argument("--delta_g", type=float, default=3.0)
    ap.add_argument("--delta_e", type=float, default=1.5)
    ap.add_argument("--r5", default="R5:depthwise", help="which R5 bracket to use as the control")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        # synthetic: R6 ~3.5 pts lower forgetting than R5, sd 2, n=12  -> should GREENLIGHT
        diffs = -3.5 + rng.normal(0, 2.0, size=12)
        print("DEMO R6-R5 diffs:", np.round(diffs, 2))
        print(json.dumps(decide(diffs, args.delta_g, args.delta_e), indent=2))
        return

    runs = load()
    if "R6" not in runs or args.r5 not in runs:
        print("Need both R6 and", args.r5, "results. Have:", sorted(runs)); return
    seeds = sorted(set(runs["R6"]) & set(runs[args.r5]))
    diffs = [runs["R6"][s] - runs[args.r5][s] for s in seeds]   # forgetting: benefit => negative
    print(f"paired seeds: {seeds}")
    print(json.dumps(decide(diffs, args.delta_g, args.delta_e), indent=2))


if __name__ == "__main__":
    main()
