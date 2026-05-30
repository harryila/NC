"""
Sequential, resumable runner for the M1 ladder matrix — chain independent GPU runs
back-to-back on one box (launch under tmux/nohup and walk away). Skips completed jobs.

FRONT-LOAD the decisive contrast first (recommended):
    python run_matrix.py --rungs R6,R5:no_proj --scenarios class10 --epochs <calibrated> --seeds 10
Then fill in the rest:
    python run_matrix.py --epochs <calibrated> --seeds 10        # full grid, skips the done ones

DEPENDENCY STRUCTURE:
  Stage 1  budget.py  -> results/sparsity_target.json   (R2-R4 need it; ONE real cross-job dep)
  Stage 2  THIS script -> results/<job>.json             jobs mutually INDEPENDENT
  Stage 3  analyze.py  -> R6-R5 increment + gate          needs the relevant Stage-2 jobs

NOTE: epochs/task is part of the job id, so changing --epochs does NOT falsely "skip" old runs.
Shard across N GPUs: CUDA_VISIBLE_DEVICES=k python run_matrix.py --shard k/N ...
!!! Depends on the avalanche_backbone.run_split_cifar100 driver — validated on the GPU in Wk-0. !!!
"""
import argparse
import json
import os
import traceback

import _bootstrap  # noqa: F401
from avalanche_backbone import run_split_cifar100
from budget import effective_sparsity

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

RUNGS = ["R1", "R2", "R3", "R4",
         "R5:depthwise", "R5:no_proj", "R5:frozen_J",
         "R6", "R6s",
         "A4:ewc", "A4:replay", "A5:derpp"]   # baselines: R6 backbone + CL strategy
SCENARIOS = [("class", 10), ("class", 20)]   # 10×10 (arbiter) + 20×5 (replication)


def _jid(rung, scen, nexp, epochs, seed):
    return f"{rung.replace(':', '_')}__{scen}{nexp}__e{epochs}__s{seed}"


def _sparsity_target(device):
    p = os.path.join(RESULTS, "sparsity_target.json")
    if os.path.exists(p):
        return json.load(open(p))
    fr = effective_sparsity("R6", device=device)   # participation-ratio target (NOT |x|>eps)
    json.dump(fr, open(p, "w"))
    return fr


def run_one(rung, scen, nexp, seed, target, epochs, device, eval_inits=8):
    out = os.path.join(RESULTS, _jid(rung, scen, nexp, epochs, seed) + ".json")
    if os.path.exists(out):
        print("skip (done):", os.path.basename(out)); return
    base, _, variant = rung.partition(":")
    if base in ("A4", "A5"):
        backbone, strat, kw = "R6", variant, {}            # baselines: R6 backbone + CL strategy
    else:
        backbone, strat, kw = base, "naive", ({"variant": variant} if variant else {})
        if base in ("R2", "R3", "R4"):
            kw["akorn_sparsity"] = sum(target) / len(target)   # TODO: per-layer match, not the mean
    print("RUN:", _jid(rung, scen, nexp, epochs, seed))
    try:
        metrics = run_split_cifar100(backbone, scenario=scen, n_experiences=nexp,
                                     seed=seed, epochs=epochs, device=device,
                                     eval_inits=eval_inits, strategy=strat, **kw)
        json.dump({"rung": rung, "scenario": scen, "nexp": nexp, "epochs": epochs,
                   "seed": seed, "metrics": metrics}, open(out, "w"), default=str)
        print("  done ->", os.path.basename(out))
    except Exception as e:
        print("  FAIL:", _jid(rung, scen, nexp, epochs, seed), "::", e); traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400, help="epochs PER TASK — calibrate first (calibrate_epochs.py)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rungs", type=str, default="", help="comma list to restrict, e.g. 'R6,R5:no_proj'")
    ap.add_argument("--scenarios", type=str, default="", help="comma list like 'class10,class20'")
    ap.add_argument("--eval_inits", type=int, default=8, help="fixed-seed eval forwards (matched across arms)")
    ap.add_argument("--shard", type=str, default="0/1", help="k/N to split across N GPUs")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))

    rungs = args.rungs.split(",") if args.rungs else RUNGS
    scens = SCENARIOS
    if args.scenarios:
        want = set(args.scenarios.split(","))
        scens = [(s, ne) for (s, ne) in SCENARIOS if f"{s}{ne}" in want]

    target = _sparsity_target(args.device)
    jobs = [(r, s, ne, sd) for (s, ne) in scens for r in rungs for sd in range(args.seeds)]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    print(f"{len(jobs)} jobs on shard {k}/{n}  (rungs={rungs}, scenarios={[f'{s}{ne}' for s,ne in scens]}, "
          f"epochs={args.epochs}, seeds={args.seeds}, eval_inits={args.eval_inits})")
    for (r, s, ne, sd) in jobs:
        run_one(r, s, ne, sd, target, args.epochs, args.device, eval_inits=args.eval_inits)
    print("queue drained.")


if __name__ == "__main__":
    main()
