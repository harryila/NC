"""Wk-0 PIPELINE VALIDATION ONLY (not confirmatory): drive the real Stage-2 runner over a
small subset of rungs/seeds at tiny epochs + reduced eval budget, so Stage 2 -> Stage 3
(analyze.py) produces a real R6-R5 gate decision on genuine (toy-scale) runs.
Writes results/*.json in the exact run_matrix format so `make analyze` consumes them."""
import _bootstrap  # noqa: F401
from run_matrix import run_one, _sparsity_target

RUNGS = ["R1", "R5:depthwise", "R5:no_proj", "R6", "R6s"]
SCEN, NEXP = "class", 10
SEEDS = [0, 1, 2]
EPOCHS = 2
EVAL_INITS = 2
DEVICE = "cuda"

if __name__ == "__main__":
    target = _sparsity_target(DEVICE)
    for sd in SEEDS:
        for r in RUNGS:
            run_one(r, SCEN, NEXP, sd, target, EPOCHS, DEVICE, eval_inits=EVAL_INITS)
    print("SCOPED_VALIDATION_DONE")
