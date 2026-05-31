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
- **Decisive contrast:** **R6 − R5:no_proj** (PRIMARY — param-identical `apply_proj` flip, the cleanest causal isolation of synchrony). `R5:depthwise` and `R5:frozen_J` are −39%-param **robustness brackets** (Holm family), not the primary. *(Revised 2026-05-30 from the original "depthwise primary": depthwise is capacity-confounded; the H3 DiD and the gate both run against the param-identical no_proj.)*

## Numbers
- **SESOI / Δg** (GREENLIGHT effect): ΔForgetting ≥ **3.0** abs pts  OR  Cohen's d ≥ **0.8** on seed-paired diffs.  **[locked]**
- **Δe** (equivalence margin for PIVOT nulls, TOST): ± **1.5** pts.  **[locked]**
- **Inconclusive band:** 1.5 < |effect| < 3.0 → add seeds, no forced call.  **[locked]**
- **Seeds:** **10** baseline; **15** for the decisive R6−R5. Each seed keys data-order + init + augmentation RNG.  **[locked]**
- **A5 competitiveness margin:** R6 within **3.0** pts of A5 = **DER++** (`avalanche DER`, beta>0; FOSTER is absent from avalanche-lib 0.6.0). A4 refs = EWC, Replay — all on the R6 backbone.  **[locked]**
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

---

## Amendment 1 — PROPOSED 2026-05-31 (post-front-load; pending ratification before the next gate run)

The decisive front-load (R6 vs R5:no_proj, class-IL 10×10, E=50, 10 seeds) exposed two defects in the
locked rules above. This amendment is **prospective** — it does NOT rewrite the front-load's recorded
component verdict (honoring the locked rule we wrote behind the veil), it binds **future** gate runs.
**Status: proposed — Harry to ratify (or veto) before the next confirmatory run.**

**(A) σ now locked from the confirmatory pilot.** The Power line above left σ as a blank `[pilot]`
placeholder, so the `|d|≥0.8` clause was committed with an unknown denominator. **Lock σ = 1.08 pts**
(seed-paired forgetting-diff SD, n=10 front-load). At this σ, `|d|=0.8` corresponds to only **0.86 pt** —
below the ±1.5 equivalence margin — so the d-clause as written rubber-stamps a sub-equivalence effect.

**(B) d-clause gated behind a raw-effect floor (fixes the internal contradiction).** A GREENLIGHT on the
forgetting component now requires the effect to **clear the equivalence band**: `ΔForgetting ≤ −3.0`
**OR** (`|d|≥0.8` **AND** `ΔForgetting ≤ −Δe = −1.5`), with perm p<0.05. This removes the case where the
same effect both GREENLIGHTs (via d) and sits in the "≈null" band (via TOST). *(Implemented in
`analyze_hardened.decide()`, 2026-05-31.)*

**(C) INVALIDATION requires AFFIRMATIVE inferiority, not failure-to-establish non-inferiority.** The
plasticity guard's `holds=False` only means non-inferiority was **not established** (absence of evidence).
A forgetting "win" is **INVALIDATED** only when R6 is **demonstrably** worse by >Δe (one-sided test rejects
H0: μ ≥ −Δe). When the guard neither holds nor establishes inferiority, the call is
**CONFOUNDED-INCONCLUSIVE**, with the forgetting↔learning collinearity `r` reported. *(Implemented in
`analyze_hardened.plasticity_guard()` + `gate()`, 2026-05-31.)*

**(D) Saturation floor-check (endpoint validity gate).** Class-IL forgetting is only interpretable as a
**memory** endpoint when early-task final retained accuracy is bounded away from 0 for **both** arms. The
front-load had **0.00%** early-task retention (forgetting↔learning collinear at **r=0.996**), making the
endpoint a learning proxy. **Pre-register:** report class-IL forgetting only if `mean early-task
retained-acc > floor_eps` (e.g. 5 pts) for both arms; otherwise forgetting is declared **untestable** and
the gate leans on the head-free **task-IL retained-acc** + **H3 CKA DiD** (per checkpoint §3.3).

_Amendment 1 ratified / dated: ____________________________
