# M1 · Wk-0 integration spike

The go/no-go gate before the 5–7 week build. **Goal:** prove the AKOrN classification model can be driven inside Avalanche as a ladder of rungs, with deterministic eval and a real compute budget — *before* committing to the full matrix. This is contact-with-reality, not more planning.

Built against the real repo (`autonomousvision/akorn`); see [anatomy.md](anatomy.md) for the line-level code map. The decisive finding from reading the source: **R6 − R5 is a single `apply_proj` flip** inside bit-identical KLayer machinery — the cleanest possible isolation of synchrony.

## Files
| File | Purpose |
|---|---|
| [00_setup.md](00_setup.md) | env, native CIFAR-10 repro, how to run the smokes |
| [anatomy.md](anatomy.md) | verified code map: where every ladder ingredient lives + injection points |
| [ladder.py](ladder.py) | `build(rung,…)` for R1–R6 (+ R5 variants, R6-scrambled) on the real model; `param_report` |
| [avalanche_backbone.py](avalanche_backbone.py) | deterministic-eval wrapper (fixes stochastic init) + Avalanche SplitCIFAR100 driver sketch |
| [budget.py](budget.py) | ms/step, peak VRAM, GPU-hour extrapolation, R6 fraction-active (k-WTA target) |
| [preregistration.md](preregistration.md) | the template to lock before any CIFAR run |

## Checklist (each line is a go/no-go)
1. **Env + native repro** — CIFAR-10 classification trains (`00_setup.md` §2). ⛔ if it won't reproduce.
2. **`python ladder.py`** — R1–R6 build; param-match table within ~2% (or flagged). ⛔ if R5/R6 don't instantiate from the real KLayer.
3. **`python avalanche_backbone.py`** — eval deterministic == True, train stochastic == True.
4. **Split-MNIST end-to-end** — one rung trains through `SplitCIFAR100`-style Avalanche loop on Split-MNIST (gradients flow, per-experience metrics computed, deterministic eval). ⛔ if the T-step forward won't compose with Avalanche.
5. **`python budget.py` (GPU)** — ms/step + peak VRAM → GPU-hour extrapolation for the matrix; R6 per-layer fraction-active captured. ⛔/rescope if the budget is infeasible.

## Output of Wk-0
- A go/no-go on the build + a concrete GPU-hour budget.
- The `akorn_sparsity` target (from `budget.fraction_active`) for the R2–R4 controls.
- A filled [preregistration.md](preregistration.md) (Δg, Δe, SESOI, seeds) from a Split-MNIST pilot.
- Any deltas vs the v2 protocol logged back in [../../research-log.md](../../research-log.md).

## ⚠ Honesty note
The `.py` files are **untested scaffolding** authored against a static read of the repo — they parse, but have not been executed (no GPU/deps here). Every step above is a *validation* step; treat a smoke that fails as expected Wk-0 work, not a surprise. Re-confirm `anatomy.md` line refs against your checkout first.
