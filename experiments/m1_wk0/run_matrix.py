"""
Sequential, resumable runner for the M1 ladder matrix — chain independent GPU runs
back-to-back on one box (launch under tmux/nohup and walk away). Skips completed jobs,
so a crash/preemption just resumes.

DEPENDENCY STRUCTURE (this is the "what can be chained" answer):
  Stage 1  budget.py  -> results/sparsity_target.json   (R6 fraction-active; the ONE real
                                                          cross-job dependency: R2-R4 need it)
  Stage 2  THIS script -> results/<job>.json             jobs are mutually INDEPENDENT
                                                          (output of one never feeds another)
  Stage 3  analyze.py  -> the R6-R5 increment + gate      needs ALL of stage 2

Because stage-2 jobs are independent, you can either (a) run them sequentially on one GPU
(this script, default), or (b) shard across N GPUs: on GPU k run  --shard k/N  with
CUDA_VISIBLE_DEVICES=k. Same results, N× faster.

!!! Depends on the (untested) avalanche_backbone.run_split_cifar100 driver — validate it in Wk-0. !!!
"""
import argparse
import json
import os
import traceback

import _bootstrap  # noqa: F401
from avalanche_backbone import run_split_cifar100
from budget import fraction_active

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The ladder + reference baselines. "R5:<variant>" brackets the synchrony definition.
RUNGS = ["R1", "R2", "R3", "R4",
         "R5:depthwise", "R5:no_proj", "R5:frozen_J",
         "R6", "R6s"]              # add "A4:ewc", "A5:derpp" once those strategies are wired
# class-IL is the arbiter: 10×10 (n_experiences=10) primary, 20×5 (n_experiences=20) replication.
SCENARIOS = [("class", 10), ("class", 20)]


def _jid(rung, scen, nexp, seed):
    return f"{rung.replace(':', '_')}__{scen}{nexp}__s{seed}"


def _sparsity_target(device):
    p = os.path.join(RESULTS, "sparsity_target.json")
    if os.path.exists(p):
        return json.load(open(p))
    fr = fraction_active("R6", device=device)         # per-layer fraction-active at R6 fixed point
    json.dump(fr, open(p, "w"))
    return fr


def run_one(rung, scen, nexp, seed, target, epochs, device):
    out = os.path.join(RESULTS, _jid(rung, scen, nexp, seed) + ".json")
    if os.path.exists(out):
        print("skip (done):", os.path.basename(out))
        return
    base, _, variant = rung.partition(":")
    kw = {}
    if variant:
        kw["variant"] = variant
    if base in ("R2", "R3", "R4"):
        kw["akorn_sparsity"] = sum(target) / len(target)   # TODO: per-layer match, not the mean
    print("RUN:", _jid(rung, scen, nexp, seed))
    try:
        metrics = run_split_cifar100(base, scenario=scen, n_experiences=nexp,
                                     seed=seed, epochs=epochs, device=device, **kw)
        json.dump({"rung": rung, "scenario": scen, "nexp": nexp, "seed": seed,
                   "metrics": metrics}, open(out, "w"), default=str)
        print("  done ->", os.path.basename(out))
    except Exception as e:  # never let one job kill the queue
        print("  FAIL:", _jid(rung, scen, nexp, seed), "::", e)
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shard", type=str, default="0/1", help="k/N to split across N GPUs")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))

    target = _sparsity_target(args.device)
    jobs = [(r, s, ne, sd)
            for (s, ne) in SCENARIOS
            for r in RUNGS
            for sd in range(args.seeds)]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    print(f"{len(jobs)} jobs on shard {k}/{n} (epochs={args.epochs}, seeds={args.seeds})")
    for (r, s, ne, sd) in jobs:
        run_one(r, s, ne, sd, target, args.epochs, args.device)
    print("queue drained.")


if __name__ == "__main__":
    main()
