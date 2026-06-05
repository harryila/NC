"""M2 ON-vs-OFF DECODABILITY SCREEN (generator-free) — the cheap, decisive prerequisite to the
C_ctx hypernetwork headline.

QUESTION: does synchrony (R6) make the oscillator phase-state carry MORE linearly-decodable TASK
information than the matched no-synchrony control (R5:no_proj — the single `apply_proj` flip)?

WHY THIS FIRST (and why it is the right screen):
  * The C_ctx estimator decodes task from the GENERATED theta; by the data-processing inequality
    (T -> context -> theta), C_ctx is UPPER-BOUNDED by the task-decodability of the context itself.
    So this screen is literally the CEILING of any C_ctx run — if R6 does not beat R5 here, the
    expensive hypernetwork headline cannot show it either.
  * It is also a clean, publishable result on its own (estimand (1) in preregistration-M2.md:
    "phase-gating raises task-information vs a matched control").
  * It reuses the VALIDATED capture path (m2_precheck._capture_phase_per_sample + _pool_phase_state
    + m2_primitives.linear_task_decodability) that reproducibly reads ~0.33 (L1) / ~0.39 (L2) on R6
    — so it cannot re-introduce the headline's wrong-layer / lossy-capture / frozen-feature bugs.

PRE-REGISTERED STOPPING RULE (set BEFORE the run; primary layer = 2, secondary = 1):
  Per seed, Delta = cv_acc(R6) - cv_acc(R5:no_proj) at the matched layer (paired: same seed=same data).
  PROCEED to C_ctx headline   iff mean Delta >= +0.03 AND one-sided paired p < 0.05  (synchrony adds channel)
  PIVOT to a NULL finding      iff equivalence within +-0.03 (TOST p < 0.05)          (synchrony reorganizes
                                  overlap [M1] but does NOT add task-channel capacity — a real constraint)
  INCONCLUSIVE / add seeds     otherwise.

Usage (GPU-2):
    python m2_screen.py --seeds 0 1 2 3 4 5 --epochs 30 --layers 1 2 --device cuda
    python m2_screen.py --demo     # CPU: verdict-logic sanity on synthetic per-seed cv
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
if M1 not in sys.path:
    sys.path.insert(0, M1)
RESULTS = os.path.join(HERE, "results")

# arm -> rung_kw passed to LadderClassifier(base, **rung_kw); base is the part before ":".
ARMS = {"R6": {}, "R5:no_proj": {"variant": "no_proj"}}
PRIMARY_LAYER = 2
EQUIV_MARGIN = 0.03      # |Delta| within this => synchrony adds nothing (PIVOT-NULL)
PROCEED_MARGIN = 0.03    # Delta >= this (with p<0.05) => synchrony adds channel (PROCEED)
ALPHA = 0.05


# ----------------------------- stats (pure-numpy + optional scipy) -----------------------------
def _ncdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _perm_one_sided_gt0(diffs):
    """One-sided paired sign-flip permutation p for H0: symmetric about 0 vs mean>0.
    Exact enumeration for n<=20 (our regime); pure numpy, no scipy needed."""
    import numpy as np
    from itertools import product
    d = np.asarray(diffs, float)
    n = len(d)
    obs = d.mean()
    if n == 0:
        return 1.0
    if n <= 20:
        cnt = tot = 0
        for signs in product((1, -1), repeat=n):
            tot += 1
            if (d * np.asarray(signs)).mean() >= obs - 1e-12:
                cnt += 1
        return cnt / tot
    rng = np.random.default_rng(0)
    signs = rng.choice((1, -1), size=(100000, n))
    return float(((signs * d).mean(1) >= obs).mean())


def _tost_equiv_p(diffs, margin):
    """Two one-sided tests for equivalence within +-margin. scipy.t if available, else normal approx."""
    import numpy as np
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return 1.0
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    try:
        from scipy import stats
        p_lower = float(stats.t.sf((m - (-margin)) / se, n - 1))   # H0: mean <= -margin
        p_upper = float(stats.t.cdf((m - margin) / se, n - 1))     # H0: mean >=  margin
    except Exception:
        p_lower = 1.0 - _ncdf((m - (-margin)) / se)
        p_upper = _ncdf((m - margin) / se)
    return max(p_lower, p_upper)


def _verdict(diffs):
    import numpy as np
    d = np.asarray(diffs, float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    dz = (m / sd) if sd > 0 else (float("inf") if m > 0 else 0.0)
    p_one = _perm_one_sided_gt0(d)
    tost_p = _tost_equiv_p(d, EQUIV_MARGIN)
    n_pos = int((d > 0).sum())
    if m >= PROCEED_MARGIN and p_one < ALPHA:
        call = "PROCEED (synchrony adds task-channel; C_ctx headline worth running)"
    elif tost_p < ALPHA:
        call = "PIVOT-NULL (synchrony does NOT add task-channel capacity beyond no-synchrony control)"
    else:
        call = "INCONCLUSIVE (add seeds)"
    return {"n": n, "mean_delta": m, "sd": sd, "dz": dz,
            "p_one_sided_gt0": p_one, "tost_p_equiv": tost_p,
            "n_pos_of_n": [n_pos, n], "call": call}


# ----------------------------- train + capture one arm -----------------------------
def _train_and_capture(rung, rung_kw, n_tasks, epochs, layers, device, eval_inits, seed, max_samples_per_task):
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa: F401  -- puts pinned external/akorn on the path
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import InteractiveLogger
    from avalanche_backbone import LadderClassifier
    from m2_precheck import _capture_phase_per_sample
    from torch.utils.data import DataLoader, Subset
    import m2_primitives as m2

    base = rung.split(":")[0]                                  # "R5:no_proj" -> "R5"
    bench = SplitCIFAR100(n_experiences=n_tasks, return_task_id=False, seed=seed)
    model = LadderClassifier(base, num_classes=100, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)
    evalp = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[InteractiveLogger()])
    cl = Naive(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
               criterion=nn.CrossEntropyLoss(), train_mb_size=128, eval_mb_size=100,
               train_epochs=epochs, evaluator=evalp, device=device)
    for exp in bench.train_stream:
        cl.train(exp)
    phase_by_task = {l: {} for l in layers}
    for i, exp in enumerate(bench.test_stream):
        ds = exp.dataset
        idx = list(range(min(max_samples_per_task, len(ds))))
        loader = DataLoader(Subset(ds, idx), batch_size=100, shuffle=False)
        for l in layers:
            phase_by_task[l][i] = _capture_phase_per_sample(
                model, loader, layer=l, device=device, eval_inits=eval_inits,
                base_seed=seed, max_samples=max_samples_per_task)
    out = {}
    for l in layers:
        cv, chance = m2.linear_task_decodability(phase_by_task[l], seed=seed)
        out[l] = {"cv": float(cv), "chance": float(chance), "margin": float(cv - chance)}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _summarize(per_seed, layers):
    import numpy as np
    summary = {}
    paired = [ps for ps in per_seed if all(a in ps for a in ARMS)]
    for l in layers:
        r6 = [ps["R6"][l]["cv"] for ps in paired]
        r5 = [ps["R5:no_proj"][l]["cv"] for ps in paired]
        diffs = [a - b for a, b in zip(r6, r5)]
        chance = paired[0]["R6"][l]["chance"] if paired else 0.2
        summary[l] = {"mean_cv_R6": (float(np.mean(r6)) if r6 else None),
                      "mean_cv_R5": (float(np.mean(r5)) if r5 else None),
                      "chance": chance,
                      "verdict": (_verdict(diffs) if len(diffs) >= 2 else {"call": "need>=2 seeds", "n": len(diffs)})}
    return summary


def _save(per_seed, layers, n_tasks, epochs, final=False):
    os.makedirs(RESULTS, exist_ok=True)
    summary = _summarize(per_seed, layers)
    out = {"arms": list(ARMS), "primary_layer": PRIMARY_LAYER, "n_tasks": n_tasks, "epochs": epochs,
           "equiv_margin": EQUIV_MARGIN, "proceed_margin": PROCEED_MARGIN,
           "per_seed": per_seed, "per_layer_summary": {str(k): v for k, v in summary.items()}}
    json.dump(out, open(os.path.join(RESULTS, "m2_screen.json"), "w"), indent=2, default=str)
    if final:
        print("\n=== M2 SCREEN VERDICT (primary layer %d) ===" % PRIMARY_LAYER)
        for l in layers:
            s = summary[l]
            v = s["verdict"]
            print("  L%d: cv R6=%s R5=%s | mean_delta=%s p1=%s tost=%s -> %s" % (
                l, _r(s["mean_cv_R6"]), _r(s["mean_cv_R5"]), _r(v.get("mean_delta")),
                _r(v.get("p_one_sided_gt0")), _r(v.get("tost_p_equiv")), v["call"]))
        prim = summary.get(PRIMARY_LAYER)
        if prim:
            print("PRIMARY (L%d) CALL: %s" % (PRIMARY_LAYER, prim["verdict"]["call"]))
        print("SCREEN_DONE")
    return out


def _r(x):
    return None if x is None else round(float(x), 4)


def run_screen(seeds, n_tasks=5, epochs=30, layers=(1, 2), device="cuda", eval_inits=4,
               max_samples_per_task=240):
    per_seed = []
    for s in seeds:
        rec = {"seed": s}
        for arm, kw in ARMS.items():
            rec[arm] = _train_and_capture(arm, kw, n_tasks, epochs, list(layers), device,
                                          eval_inits, s, max_samples_per_task)
            print("[seed %d] %s: %s" % (s, arm, " ".join(
                "L%d cv=%.3f(margin%+.3f)" % (l, rec[arm][l]["cv"], rec[arm][l]["margin"]) for l in layers)),
                flush=True)
        per_seed.append(rec)
        _save(per_seed, layers, n_tasks, epochs)            # incremental save (resumable view)
        print("[seed %d] saved (%d/%d seeds done)" % (s, len(per_seed), len(seeds)), flush=True)
    return _save(per_seed, layers, n_tasks, epochs, final=True)


def _demo():
    import numpy as np
    rng = np.random.default_rng(0)
    d_pos = [0.06 + rng.normal(0, 0.02) for _ in range(6)]
    d_null = [0.0 + rng.normal(0, 0.012) for _ in range(6)]
    print("[demo] PROCEED case:", _verdict(d_pos))
    print("[demo] NULL case   :", _verdict(d_null))
    assert _verdict(d_pos)["call"].startswith("PROCEED")
    assert _verdict(d_null)["call"].startswith("PIVOT-NULL")
    print("=== M2 SCREEN DEMO OK ===")


def main():
    ap = argparse.ArgumentParser(description="M2 generator-free ON-vs-OFF decodability screen")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    ap.add_argument("--max-samples-per-task", type=int, default=240)
    a = ap.parse_args()
    if a.demo:
        _demo()
        return
    run_screen(a.seeds, n_tasks=a.n_tasks, epochs=a.epochs, layers=tuple(a.layers),
               device=a.device, eval_inits=a.eval_inits, max_samples_per_task=a.max_samples_per_task)


if __name__ == "__main__":
    main()
