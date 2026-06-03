"""Positive-control DIFFICULTY SWEEP (Decision 1: empirical tuning before the real control run).

The positive control must avoid the SATURATION that floored the CIFAR forgetting metric: if the
synthetic binding task is too easy, BOTH R6 and R5:no_proj memorise it (learning-acc -> ~100%, retained
-> floor) and the inter-task overlap effect vanishes; too hard and neither arm learns above chance
(1/9 ~ 11%). This sweep finds an operating point where BOTH arms land in a NON-SATURATED band so the
H3 overlap contrast is measured on a task both arms actually solve-but-don't-trivially-memorise.

It varies the module-level positive_control.DIFFICULTY (jitter / bg_noise / pos_jitter) and epochs over
a small grid, runs R6 and R5:no_proj at a few seeds each (cheap: short epochs, few imgs), and scores each
setting by:
  * both_learn      : both arms' mean learning-acc (mean_k A[k,k]) in [acc_lo, acc_hi]  (default 30..85%)
  * not_saturated   : both arms' learning-acc < sat_hi (default 92%)  AND  > chance+margin
  * signal_present  : R6 inter-task O_inter < R5 O_inter (the synchrony-favoring direction)
Prints a ranked table and writes results/positive_control_sweep.json. Pick the setting that is
non-saturated AND shows the signal, then run positive_control.run_positive_control at that DIFFICULTY
with the full seed count.

Usage (GPU box):
    python positive_control_sweep.py --seeds 2 --epochs-grid 20,35 --device cuda
    python positive_control_sweep.py --demo          # CPU: enumerate the grid + scoring logic, no training
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (jitter, bg_noise, pos_jitter) operating points, easy -> hard.
DIFFICULTY_GRID = [
    {"jitter": 0.10, "bg_noise": 0.05, "pos_jitter": 4, "radius_range": (7, 10)},   # default (easiest)
    {"jitter": 0.18, "bg_noise": 0.10, "pos_jitter": 6, "radius_range": (6, 11)},   # medium
    {"jitter": 0.28, "bg_noise": 0.18, "pos_jitter": 8, "radius_range": (5, 12)},   # hard
    {"jitter": 0.40, "bg_noise": 0.28, "pos_jitter": 10, "radius_range": (4, 13)},  # very hard
]


def _score(r6_learn, r5_learn, r6_oint, r5_oint, acc_lo=30.0, acc_hi=85.0, sat_hi=92.0, chance=100.0 / 9):
    """Score one (difficulty, epochs) cell from both arms' learning-acc + inter-task overlap."""
    both_learn = (acc_lo <= r6_learn <= acc_hi) and (acc_lo <= r5_learn <= acc_hi)
    not_saturated = (r6_learn < sat_hi) and (r5_learn < sat_hi)
    above_chance = (r6_learn > chance + 8) and (r5_learn > chance + 8)
    signal_present = (r6_oint < r5_oint)
    # rank key: prefer non-saturated + above-chance + signal; tie-break by how centered in the band
    center = -(abs(r6_learn - 57.5) + abs(r5_learn - 57.5))
    ok = both_learn and not_saturated and above_chance and signal_present
    return {"both_learn": both_learn, "not_saturated": not_saturated, "above_chance": above_chance,
            "signal_present": signal_present, "usable": ok, "rank_key": (ok, center)}


def run_sweep(seeds=2, epochs_grid=(20, 35), n_tasks=3, device="cuda",
              eval_inits=4, per_class_train=200, per_class_test=60):
    import numpy as np
    import positive_control as pc

    rows = []
    for di, diff in enumerate(DIFFICULTY_GRID):
        for epochs in epochs_grid:
            pc.DIFFICULTY.update(diff)                      # set the operating point for this cell
            r6_learn, r5_learn, r6_oint, r5_oint = [], [], [], []
            for s in range(seeds):
                r6 = pc.run_positive_control_arm("R6", variant=None, n_tasks=n_tasks, seed=s,
                                                 epochs=epochs, per_class_train=per_class_train,
                                                 per_class_test=per_class_test, device=device,
                                                 eval_inits=eval_inits)
                r5 = pc.run_positive_control_arm("R5", variant="no_proj", n_tasks=n_tasks, seed=s,
                                                 epochs=epochs, per_class_train=per_class_train,
                                                 per_class_test=per_class_test, device=device,
                                                 eval_inits=eval_inits)
                r6_learn.append(float(np.mean(r6["learning_acc"])))
                r5_learn.append(float(np.mean(r5["learning_acc"])))
                r6_oint.append(float(r6["overlap_summary"]["O_inter"]))
                r5_oint.append(float(r5["overlap_summary"]["O_inter"]))
            m = lambda v: float(np.mean(v))
            sc = _score(m(r6_learn), m(r5_learn), m(r6_oint), m(r5_oint))
            row = {"difficulty_idx": di, "difficulty": diff, "epochs": epochs,
                   "r6_learn": m(r6_learn), "r5_learn": m(r5_learn),
                   "r6_O_inter": m(r6_oint), "r5_O_inter": m(r5_oint),
                   "delta_O_inter_R5_minus_R6": m(r5_oint) - m(r6_oint), **sc}
            rows.append(row)
            print(f"[diff{di} ep{epochs}] R6 learn={row['r6_learn']:.1f}% R5 learn={row['r5_learn']:.1f}%"
                  f" | O_inter R6={row['r6_O_inter']:.4f} R5={row['r5_O_inter']:.4f}"
                  f" | usable={sc['usable']}")
    rows.sort(key=lambda r: r["rank_key"], reverse=True)
    best = rows[0] if rows and rows[0]["usable"] else None
    out = {"rows": rows, "best": best,
           "recommendation": (f"use difficulty_idx={best['difficulty_idx']} epochs={best['epochs']}"
                              if best else "NO usable cell found — widen the grid or adjust the band")}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "positive_control_sweep.json"), "w"), indent=2, default=str)
    print("\n=== SWEEP RESULT ===")
    print(out["recommendation"])
    if best:
        print(f"  best: R6 learn={best['r6_learn']:.1f}% R5 learn={best['r5_learn']:.1f}% "
              f"deltaO={best['delta_O_inter_R5_minus_R6']:+.4f}")
    return out


def _demo():
    """CPU: enumerate the grid + exercise the scoring logic on synthetic accuracy/overlap inputs."""
    print(f"DIFFICULTY_GRID has {len(DIFFICULTY_GRID)} operating points x epochs => the sweep cells.")
    for di, d in enumerate(DIFFICULTY_GRID):
        print(f"  diff{di}: {d}")
    print("\nscoring logic checks:")
    cases = [
        ("saturated (both ~99%)", 99.0, 99.0, 0.45, 0.48),
        ("floored (both ~12%)", 12.0, 12.0, 0.50, 0.50),
        ("good non-saturated + signal", 60.0, 64.0, 0.45, 0.49),
        ("learns but NO signal (R6>=R5)", 60.0, 64.0, 0.50, 0.49),
    ]
    for name, a, b, o6, o5 in cases:
        sc = _score(a, b, o6, o5)
        print(f"  {name:34s} -> usable={sc['usable']}  "
              f"(learn={sc['both_learn']} notsat={sc['not_saturated']} abovech={sc['above_chance']} sig={sc['signal_present']})")
    assert _score(99, 99, 0.45, 0.48)["usable"] is False, "saturated must be unusable"
    assert _score(12, 12, 0.5, 0.5)["usable"] is False, "floored must be unusable"
    assert _score(60, 64, 0.45, 0.49)["usable"] is True, "non-saturated+signal must be usable"
    assert _score(60, 64, 0.50, 0.49)["usable"] is False, "no-signal must be unusable"
    print("\n=== SWEEP DEMO OK (grid + scoring logic validated) ===")


def main():
    ap = argparse.ArgumentParser(description="Positive-control difficulty sweep")
    ap.add_argument("--demo", action="store_true", help="CPU: grid + scoring logic, no training")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs-grid", type=str, default="20,35", help="comma list of epoch settings")
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    args = ap.parse_args()
    if args.demo:
        _demo(); return
    eg = tuple(int(x) for x in args.epochs_grid.split(","))
    run_sweep(seeds=args.seeds, epochs_grid=eg, n_tasks=args.n_tasks,
              device=args.device, eval_inits=args.eval_inits)


if __name__ == "__main__":
    main()
