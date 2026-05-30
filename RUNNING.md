# Running NeuralCombs on a GPU box

## What's vendored vs. pinned
- **Everything we author** lives in this repo: `experiments/m1_wk0/*.py`, the docs, `Makefile`, `setup.sh`.
- **AKOrN is NOT copied into the repo.** `setup.sh` clones it (pinned at commit `eabbe27`) into `external/akorn/` (gitignored). Our code imports it; `experiments/m1_wk0/_bootstrap.py` puts it on the path automatically. This keeps your git history yours, respects AKOrN's license, and pins reproducibility.

## SSH in, then run (in order — each is a go/no-go)
```bash
git clone <your NeuralCombs remote> && cd NeuralCombs
bash setup.sh                 # clone+pin AKOrN, install lean deps   (Stage 0)
source env.sh                 # optional; _bootstrap.py also handles paths

make repro                    # native CIFAR-10 classification trains?      ⛔ if not
make smoke                    # ladder R1–R6 build + deterministic eval?    ⛔ if not
make pilot                    # GPU-hours estimate + R6 fraction-active      -> rescope if infeasible
#   ^ also writes results/sparsity_target.json (the one cross-stage dependency)
```
Then validate one end-to-end CL run on the small benchmark before the big matrix (edit `run_matrix.py`
`SCENARIOS`/seeds down, or call `run_split_cifar100("R6", scenario="class", n_experiences=5, epochs=2)`).

## The experiment DAG — what can be chained vs. what's truly dependent
Your instinct is right: most of the work is **independent runs that just need to be queued back-to-back**, with only a couple of real dependencies at the stage seams.

```
Stage 1  pilot/budget ──► results/sparsity_target.json        (R2–R4 need this; ONE real dependency)
                              │
Stage 2  ladder matrix ──────┴─► results/<rung>__<scen>__<seed>.json
          R1 R2 R3 R4 R5:depthwise R5:no_proj R5:frozen_J R6 R6s   × seeds × {10×10, 20×5}
          → these are MUTUALLY INDEPENDENT (output of one never feeds another)
                              │
Stage 3  analyze ────────────┴─► R6 − R5 increment + permutation test + TOST + gate call
          (needs ALL of Stage 2)
```

### Chain them on one GPU (queue, walk away)
`run_matrix.py` is **sequential + resumable** — it runs every job in turn and skips any whose
`results/*.json` already exists, so a crash just resumes. Launch detached:
```bash
tmux new -s m1
make matrix EPOCHS=400 SEEDS=10        # runs the whole grid one after another
# (or chain stages so analyze only fires when the matrix finishes:)
make matrix && make analyze
```

### Or fan out across N GPUs (same results, N× faster — they're independent)
```bash
# GPU 0:
CUDA_VISIBLE_DEVICES=0 python experiments/m1_wk0/run_matrix.py --shard 0/4 &
# GPU 1:
CUDA_VISIBLE_DEVICES=1 python experiments/m1_wk0/run_matrix.py --shard 1/4 &
# ... 2/4, 3/4 ; then `make analyze` once all shards drain.
```

## Next steps toward the goal (and what to parallelize)
1. **Wk-0 (gate):** `setup → repro → smoke → pilot` + one tiny end-to-end CL run. Output: go/no-go + GPU-hour budget + a filled `experiments/m1_wk0/preregistration.md`.
2. **Build the controls (Wk-1/2):** finish `LadderCore` (R1–R4), tune k-WTA to `sparsity_target.json`, enforce ±2% param/FLOP match, wire the A4 (EWC/replay) and A5 (DER++/FOSTER) strategies into `run_split_cifar100`. *These are independent of the M1 result and can be built while pilots run.*
3. **Stage 2 (the matrix):** `make matrix` — the chained runs. Independent → queue or shard.
4. **Stage 3 (decide):** `make analyze` → R6−R5 gate. Then the positive-control + longer-stream replication (also independent of each other — queue them too).
5. **Branch on the gate:** GREENLIGHT → M2 (oscillatory-workspace channel capacity, reuses this ladder as the substrate); PIVOT-A → the R3/R4/R5 decomposition write-up; PIVOT-B → first clean synchrony-on-CL benchmark + try a non-AKOrN oscillatory substrate. All three are logged forward in `research-log.md`.

> Honesty: the `.py` files are syntax-validated but **unrun** here (no GPU/deps on the dev box) except `analyze.py --demo`. Treat every `make` step as a validation gate.
