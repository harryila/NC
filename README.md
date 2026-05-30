# NeuralCombs

A **living experiment** in neuro → AI/ML transfer. One stack, built deep, evolving forward: when something doesn't work, the result *redirects* the frontier rather than ending it. Every pivot is a contributive decision, logged.

## The bet

Build the **"Oscillatory Workspace"** — fuse the two neuro-inspired threads that just matured (AKOrN oscillatory binding + global-workspace bottleneck) and aim them at the one continual-learning question that just became theoretically precise (Context Channel Capacity). Orientation: **contribution** (not understanding-for-its-own-sake, not a product). The edge is building the *integration* that specialist labs won't, and shipping it faster.

**Unifying thesis:** *oscillator phase-state can serve as a label-free context channel that lets a conditional-regeneration (hypernetwork) architecture bypass the catastrophic-forgetting Impossibility Triangle — without task labels.*

## Folder map

| File | What |
|---|---|
| [thesis-and-milestones.md](thesis-and-milestones.md) | The thesis, the 3 milestones, and the per-milestone pivot tree (what we do if each fails) |
| [prior-art-derisk.md](prior-art-derisk.md) | Exhaustive adversarial prior-art de-risk — verdicts, anchor papers, near-misses, kill-risks |
| [research-log.md](research-log.md) | Dated decision log — the "living" record; every choice + what would trigger a pivot |
| [RUNNING.md](RUNNING.md) | How to run on a GPU box: setup, the experiment DAG, chaining/sharding runs |
| `setup.sh` · `Makefile` · `requirements.txt` | Bring in pinned AKOrN (→ `external/akorn`, gitignored) + staged pipeline targets |
| [experiments/M1-protocol.md](experiments/M1-protocol.md) | The v2 protocol (the control ladder, gates, stats) |
| [experiments/m1_wk0/](experiments/m1_wk0/) | M1 code: `ladder.py` (R1–R6), `avalanche_backbone.py` (+A4/A5 strategy dispatch), `budget.py`, `calibrate_epochs.py`, `run_matrix.py` (chained runner, baselines), `analyze.py` (R6−R5 gate stats), `h3.py` (CKA/phase mechanism), `anatomy.md`, `preregistration.md` |

## Status (2026-05-30)

- ✅ Direction chosen: contribution / Oscillatory Workspace
- ✅ Prior-art de-risk: all 3 milestone gaps real, thesis novel (HIGH confidence)
- ✅ Residual prior-art angles closed — no verdict flipped, confidence HIGH
- ✅ M1 protocol drafted, red-teamed (blockers found), revised to **v2** ([experiments/M1-protocol.md](experiments/M1-protocol.md)) — synchrony isolated as the R6−R5 ladder increment
- ✅ M1 Wk-0 scaffolding authored against the real AKOrN source ([experiments/m1_wk0/](experiments/m1_wk0/)) — ladder R1–R6, deterministic-eval wrapper, budget probe, pre-registration template (syntax-validated, untested)
- ✅ Wk-0 ran on A100 — all gates pass; caught + fixed the sparsity-probe bug + 400-epoch over-scope; merged GPU+local
- ✅ Phase-2 #1 (A4 EWC/Replay + A5 DER++ anchors) and #2 (H3 CKA/phase mechanism module) built + reviewed
- 🔄 GPU: calibrate epochs → front-load R6 vs R5:no_proj (decisive contrast) → analyze
- ⬜ #3 harden analyze (plasticity guard + multiplicity) + H3 driver loop · #4 positive control · #5 R1–R4 LadderCore

## Operating principle

Contact with reality early and often. Each milestone is a **gate**: it greenlights the next or saves months. A negative result is publishable and informative — there is no wasted-tuition outcome, only a sharper next decision.
