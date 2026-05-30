# Research log

The living record. Each entry: a decision, its rationale, and what would trigger a pivot. Pivots are forward moves, not failures.

---

## 2026-05-30 — Direction locked: contribution / Oscillatory Workspace
Out of 27 translation-doc proposals, chose the **contribution** orientation (vs understanding / product) and the **Oscillatory Workspace** stack (AKOrN + workspace + hypernetwork). Rationale: the strongest signal cluster (VanRullen: GWT + oscillatory binding + System-2 reasoning) consists of small-scale proofs-of-concept; the open lane is *integration + scale*, which specialist labs don't build. Resisting the "do all 27" instinct — one stack, deep, to a result.

## 2026-05-30 — Prior-art de-risk completed (15-agent adversarial workflow)
**Verdict: all 3 milestone gaps `gap_real`, unifying thesis `novel`, HIGH confidence.** No single paper or pair unifies the milestones. Two milestones ride on public code (AKOrN, GASPnet, von Oswald) → less scaffolding, more contact-with-reality. Details + anchor IDs + kill-risks in [prior-art-derisk.md](prior-art-derisk.md).
Key sharpenings (mandatory, not optional):
- M1 must out-argue **sparsity** (synchrony-vs-matched-sparsity ablation) and ablate replay OUT.
- M2 novelty is the **measurement** (channel capacity + phase-gating ablation), not the architecture.
- M3 novelty is **oscillatory-phase-as-CCC-channel**, not "label-free CL" (already done by MESU/metaplasticity).
Two theoretical traps to engineer against from day one: **CFlow bypass** (pathway must be unbypassable) and **S_N symmetry** (synchrony must be the explicit symmetry-breaker; measure I(phase;T) ≥ H(T)).

## 2026-05-30 — Residual prior-art sweep: closed, no verdict flipped (8-agent workflow)
Swept 7 previously-unaddressed communities + the AKOrN/CCC citation graphs. **Nothing flips a milestone; confidence stays HIGH.** Two reframings now locked in:
- **M1** — frame as "AKOrN-class binding-by-synchrony, **not** generic random reservoirs" (reservoir-CL exists but on generic reservoirs + regression).
- **M3** — frame as "**label-free, structurally-unbypassable** phase-state channel," not "ODE+hypernet+CL" (the latter exists via **sNODE** [2311.03600], but conditioned on a *discrete task embedding*). Label-free CL itself is occupied (MESU/metaplasticity) — the oscillatory-phase-as-CCC-channel is the wedge.
- **M2** unaffected — cleanest of the three (zero adjacency).
Named-but-unprobed (non-blocking, later glance): slot-attention/object-centric CL; traveling-waves/wave-RNN + CL.

## 2026-05-30 — De-risk complete → drafting M1 experimental protocol
Prior-art fully closed. Moving to the M1 spec: AKOrN-as-backbone on standard CL benchmarks with the 3 control arms, the matched-sparsity calibration, ACC/BWT/forgetting, the representation-overlap mechanistic analysis, and pre-registered decision gates. See [experiments/M1-protocol.md](experiments/M1-protocol.md).

## 2026-05-30 — M1 protocol red-teamed → BLOCKERS found → revised to v2
4-lens adversarial red-team (validity/confounds · AKOrN-feasibility-verified-against-repo · statistics · reviewer-defense) returned **`blockers_present`**. The living experiment working as intended — caught before any code. Core defect: **v1's decisive "AKOrN vs matched-sparsity" contrast was confounded** — AKOrN differs from a sparse net on ≥4 axes at once (structured sparsity, N-dim spherical vector-coding, per-step L2 renormalization, T-step recurrent dynamics) *before* phase-coupling; normalization + vector-coding alone reduce forgetting. So v1 could GREENLIGHT on geometry, not synchrony.
Other verified blockers: the non-Kuramoto control **doesn't ship** (ItrSA is object-discovery-only; `train_classification.py` is CIFAR-10-only → must be **built** from `knet.py`); AKOrN re-inits oscillators randomly each forward (stochastic BWT/CKA); E-voting is Sudoku-only (test-time compute unmatchable across arms); **n=5 Wilcoxon can't reach p<0.05** (floor 0.0625); task-IL saturates (can't carry the moat); no positive control to prove detection power.
**v2 fix (the key idea):** replace the flat arm matrix with a **one-ingredient-at-a-time ladder** R1→R6; synchrony's causal effect = the **R6−R5 increment** (phase-coupling added to an otherwise-identical normalized, vector-coded, recurrent, structured-sparse backbone). Plus: build+validate rungs from `knet.py`, GroupNorm, deterministic eval, class-IL as sole arbiter, positive control, TOST equivalence for the PIVOT nulls, ≥10 seeds + exact permutation test, pre-registered Δg/Δe, SOTA anchor (A5), longer-stream sign-replication, novelty reframe vs Verbeke&Verguts. See [experiments/M1-protocol.md](experiments/M1-protocol.md) v2.

## 2026-05-30 — Wk-0 scaffolding authored against the real AKOrN source
Cloned `autonomousvision/akorn` and read the actual code; key finding that sharpens v2: **R6 − R5 is a single `apply_proj` flip** — the repo's `KLayer` already exposes the tangent-space Kuramoto projection (`klayer.py:141`) as a boolean, so synchrony can be isolated inside bit-identical machinery (cleanest possible decisive contrast). Also verified: stochastic `torch.randn_like` init every forward (knet.py:182), BN default, fixed Linear head, CIFAR-10-only training path.
Wrote a runnable Wk-0 spike package → [experiments/m1_wk0/](experiments/m1_wk0/): `anatomy.md` (line-level code map), `ladder.py` (R1–R6 + R5 variants + R6-scrambled on the real model, param-match report), `avalanche_backbone.py` (deterministic-eval wrapper + SplitCIFAR100 driver sketch), `budget.py` (GPU-hour extrapolation + R6 fraction-active → k-WTA target), `00_setup.md`, `preregistration.md`, `README.md` (go/no-go checklist). All `.py` syntax-validated (py_compile); **untested** beyond that (no GPU/deps here) — execution is the Wk-0 work.

## 2026-05-30 — Repo made self-sufficient + chained runner; git-initialized
NeuralCombs is now a runnable repo (`git init` done). AKOrN is **pinned, not vendored**: `setup.sh` clones it at commit `eabbe27` into `external/akorn/` (gitignored); `experiments/m1_wk0/_bootstrap.py` puts it on the path; nothing of theirs enters our git history. Added: `requirements.txt`, `.gitignore`, `env.sh`, a `Makefile` (setup/repro/smoke/pilot/matrix/analyze), [RUNNING.md](RUNNING.md) (the experiment DAG + how to chain/shard GPU runs), `run_matrix.py` (sequential **resumable** queue — skips completed jobs; shard across GPUs via `--shard k/N`), and `analyze.py` (R6−R5 gate: exact paired permutation test + TOST + Cohen's d). The stats module was **actually executed** (`analyze.py --demo` in a venv) and behaves correctly; the torch/Avalanche pieces remain syntax-validated-only until the GPU box.
Experiment DAG: Stage-1 `budget.py` → `sparsity_target.json` (the one real cross-job dependency) → Stage-2 ladder matrix (mutually independent jobs, queue or shard) → Stage-3 `analyze.py` (needs all of Stage-2).

## 2026-05-30 — Wk-0 ran on an A100: ALL GATES PASS; two issues caught before the big spend
SSH'd A100-80GB: `setup`/`repro`/`smoke`/`pilot`/tiny-CL all passed. Scaffolding works on real hardware. Confirmed: **R6 and R5:no_proj are bit-identical at 7,046,890 params** (the decisive contrast is param-clean by construction), deterministic eval works, Avalanche metric key = `StreamForgetting/eval_phase/test_stream`. The full grid would be ~1,000+ GPU-h (~6–7 weeks) on one A100 — and the runner was ordered rung-major, so R6 ran ~8th (headline weeks away). Two bugs caught:
- **sparsity probe bug:** `fraction_active` thresholded the sphere-normalized oscillator state → ~0.999 "active" (meaningless). Fixed → `effective_sparsity` via participation ratio of the readout features. (Doesn't affect R6−R5; does fix R2–R4.)
- **400 epochs/task is AKOrN's from-scratch recipe, not a CL recipe** → over-fits per task + dominates cost. Added `calibrate_epochs.py` to set epochs by the task-0 accuracy plateau (plasticity guard). 

**Decision (NeurIPS-bar reasoning):** front-loading R6 vs R5:no_proj is *de-risking the submission*, not a shortcut; the epoch cut must be *calibrated* (plateau), not guessed; either sign is publishable if the rigor (full ladder + mechanism H3 + A5 anchor + replication + pre-registered stats + positive control) holds. Two-phase plan: **Phase 1 (days)** = calibrate epochs → fix sparsity → front-load R6+R5:no_proj @ ≥10 seeds class-IL 10×10 → decisive signal. **Phase 2 (weeks, if warranted)** = the full rigor stack for submission.
Added `run_matrix.py --rungs/--scenarios` filters + epochs-in-job-id (so changing epochs doesn't falsely skip).

> Next on the GPU: kill the current run + clear `results/`; `python calibrate_epochs.py` → pick E; `python run_matrix.py --rungs R6,R5:no_proj --scenarios class10 --epochs E --seeds 10 && python analyze.py`. Then queue the rest behind it.

## 2026-05-30 — Phase-2 #1 (A4/A5 anchors) + #2 (H3 module) built (verify→draft→adversarial-review workflow)
5-agent workflow verified the **real avalanche-lib 0.6.0 API** then drafted + reviewed the code. Key verified facts: strategy constructors are **keyword-only** (the old positional `Naive(...)` would TypeError → converted); **DER++ ships as `avalanche.training.supervised.DER` with beta>0** (no plugin needed); **FOSTER is absent** → A5 = DER++. The adversarial review (verdict `fix_first`) caught a **buggy reservoir counter in a hand-rolled DER++ plugin** → dropped it entirely for the built-in `DER`; flagged the **dark-knowledge target** subtlety (stored logits are eval-averaged, replay forward is stochastic → documented contract, eval_inits=1 escape hatch); and fixed H3's **seeding** (now reuses CPU+CUDA `_seeded`), added a **GroupNorm guard** on the CKA tap, and corrected the **DiD honesty** (O_intra=1 ⇒ it's a paired R6-vs-R5 overlap contrast, true DiD via augmentation = TODO).
Delivered: `avalanche_backbone.py` strategy dispatch {naive,ewc,replay,derpp} on the R6 backbone; `run_matrix.py` routes `A4:ewc`/`A4:replay`/`A5:derpp` → R6+strategy; `h3.py` (linear CKA, hooked feature extraction, inter-task overlap, paired contrast + BCa, phase-cluster-stability observable, sparsity/geometry partialling) — `h3.py --demo` runs clean. CKA math independently verified correct (Kornblith 2019). Remaining for the run: the H3 **driver snapshot-loop** (carve a fixed probe set; call `extract_features` per task) — folds into #3.

> Next here: #3 harden `analyze.py` (plasticity guard + Holm/intersection-union) + the H3 driver snapshot loop; then #4 positive control; #5 finalize R1–R4 LadderCore.
