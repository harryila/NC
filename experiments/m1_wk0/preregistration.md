# M1 pre-registration (fill BEFORE any Split-CIFAR-100 run)

Locking these before seeing results is what makes the GREENLIGHT/PIVOT calls honest (red-team blocker #5/#6).

> **Wk-0 status (2026-05-30, re-validated on A100-80GB):** `make setup repro smoke pilot` all PASS;
> ladder R1–R6 from real KLayer (R6 vs R5:no_proj param-identical at 7,046,890); deterministic eval OK;
> tiny CL smoke R6 class-IL 5×2ep emits `StreamForgetting/eval_phase/test_stream`; `sparsity_target.json`
> written (~0.999/layer). Confirmatory `make matrix` (180 jobs, 400 ep, 10 seeds) launched next.
> Lock σ from multi-seed pilot before interpreting the gate; **[locked]** protocol numbers unchanged below.

## Primary
- **Endpoint:** class-IL average **Forgetting** (lower = better), key `StreamForgetting/eval_phase/test_stream`.  [alt: final-average-ACC]
- **Scenario:** Split-CIFAR-100, class-IL, **10 × 10** (primary).
- **Replication stream:** **20 × 5** (Split-CIFAR-100); Split-TinyImageNet optional — effect must replicate in **sign**.
- **Decisive contrast:** **R6 − R5**, with R5 ∈ {**depthwise (primary)**, no_proj, frozen_J}.

## Numbers
- **SESOI / Δg** (GREENLIGHT effect): ΔForgetting ≥ **3.0** abs pts  OR  Cohen's d ≥ **0.8** on seed-paired diffs.  **[locked]**
- **Δe** (equivalence margin for PIVOT nulls, TOST): ± **1.5** pts.  **[locked]**
- **Inconclusive band:** 1.5 < |effect| < 3.0 → add seeds, no forced call.  **[locked]**
- **Seeds:** **10** baseline; **15** for the decisive R6−R5. Each seed keys data-order + init + augmentation RNG.  **[locked]**
- **A5 competitiveness margin:** R6 within **3.0** pts of A5 (DER++/FOSTER).  **[locked]**
- **Test:** exact paired **permutation / sign-flip** (n≤22 exact, else MC; NOT n=5 Wilcoxon). Multiplicity: intersection-union for conjunctions; Holm within the confirmatory family.  **[locked]**
- **Power:** with n=15 and pilot SD σ=____, MDES at 80% power = ____.  **[pilot]** *(Wk-0 budget probe on A100-80GB: R6 ≈152 ms/step, ~1.5 GB peak; extrapolated ~530 GPU-h for 8 arms×10 seeds×10×10×400 ep single-scenario core — full 180-job matrix with 2 scenarios is ~3× that before eval overhead; lock σ from confirmatory pilot SD.)*

## Decision rule
- **GREENLIGHT M2** iff: R6−R5 ≥ Δg (perm p<0.05) AND sign-replicates on the longer stream AND plasticity guard holds (R6 learning-acc TOST-equivalent to R5) AND H3 difference-in-differences positive AND R6 competitive with A5 (within 3.0 pts).
- **PIVOT-A** (synchrony≈geometry): |R6−R5| within ±Δe (TOST, 90% CI) AND positive control passes.
- **PIVOT-B** (no CL benefit): R6 ≈ R1 (TOST).
- **INCONCLUSIVE:** otherwise → more seeds.

## Guards
- **Task-IL** is diagnostic only; excluded from gating if any arm > 95%.
- **Positive control** (synchrony-favoring task) must PASS before any null (PIVOT-A/B) is declared.
- **Confound audit** logged per arm: params, FLOPs, the 4 sparsity metrics, norm type, lr/opt/init, eval-init/ensemble budget, effective DOF/subspace-rank, inference FLOPs.
  - *Wk-0 note:* R5:no_proj is param-identical to R6 (clean single-`apply_proj` flip); R5:depthwise / R5:frozen_J / R6-scrambled are −39% params (connectivity replaced/frozen) → report as **capacity-bounded** brackets or compensate width in Wk-1, per protocol §4.

## H3 (mechanism) — pre-specify
- **Primary metric:** seed-paired inter-task linear-CKA (subsample-matched) on a fixed held-out probe.
- **Prediction:** overlap reduction present in R6 but absent/smaller in R5 (**difference-in-differences**); the synchrony-specific observable (phase-cluster assignment stability) predicts forgetting after partialling out sparsity + normalization.

_Signed / dated before runs: ____________________________
