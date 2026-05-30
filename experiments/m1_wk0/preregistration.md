# M1 pre-registration (fill BEFORE any Split-CIFAR-100 run)

Locking these before seeing results is what makes the GREENLIGHT/PIVOT calls honest (red-team blocker #5/#6).

## Primary
- **Endpoint:** class-IL average **Forgetting** (lower = better).  [alt: final-average-ACC]
- **Scenario:** Split-CIFAR-100, class-IL, ___ × ___ (e.g. 10×10).
- **Replication stream:** ___ (e.g. 20×5 and/or Split-TinyImageNet) — effect must replicate in **sign**.
- **Decisive contrast:** **R6 − R5**, with R5 ∈ {depthwise (primary), no_proj, frozen_J}.

## Numbers (fill from the Split-MNIST pilot)
- **SESOI / Δg** (GREENLIGHT effect): ΔForgetting ≥ ____ abs pts  OR  Cohen's d ≥ ____ on seed-paired diffs.
- **Δe** (equivalence margin for PIVOT nulls, TOST): ± ____ pts.
- **Inconclusive band:** Δe < |effect| < Δg → add seeds, no forced call.
- **Seeds:** ____ (≥10; 15–20 for R6−R5). Each seed keys data-order + init + augmentation RNG.
- **Test:** exact paired **permutation / sign-flip** (NOT n=5 Wilcoxon). Multiplicity: intersection-union for conjunctions; Holm within the confirmatory family.
- **Power:** with n=____ and pilot SD σ=____, MDES at 80% power = ____.

## Decision rule
- **GREENLIGHT M2** iff: R6−R5 ≥ Δg (perm p<0.05) AND sign-replicates on the longer stream AND plasticity guard holds (R6 learning-acc TOST-equivalent to R5) AND H3 difference-in-differences positive AND R6 competitive with A5 (within ____ pts).
- **PIVOT-A** (synchrony≈geometry): |R6−R5| within ±Δe (TOST, 90% CI) AND positive control passes.
- **PIVOT-B** (no CL benefit): R6 ≈ R1 (TOST).
- **INCONCLUSIVE:** otherwise → more seeds.

## Guards
- **Task-IL** is diagnostic only; excluded from gating if any arm > 95%.
- **Positive control** (synchrony-favoring task) must PASS before any null (PIVOT-A/B) is declared.
- **Confound audit** logged per arm: params, FLOPs, the 4 sparsity metrics, norm type, lr/opt/init, eval-init/ensemble budget, effective DOF/subspace-rank, inference FLOPs.

## H3 (mechanism) — pre-specify
- **Primary metric:** seed-paired inter-task linear-CKA (subsample-matched) on a fixed held-out probe.
- **Prediction:** overlap reduction present in R6 but absent/smaller in R5 (**difference-in-differences**); the synchrony-specific observable (phase-cluster assignment stability) predicts forgetting after partialling out sparsity + normalization.

_Signed / dated before runs: ___________________________
