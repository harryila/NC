# M1 — Experimental protocol (v2)

*Revised 2026-05-30 after an adversarial red-team flagged that v1's decisive contrast (AKOrN vs. "matched sparsity") was **confounded**: it controlled only activation cardinality, leaving normalization geometry, vector-coding, recurrence, and test-time compute as alternative explanations for any forgetting advantage. v2 isolates synchrony via a one-ingredient-at-a-time **ladder**, and corrects feasibility errors (the non-Kuramoto control does not ship; it must be built).*

**Question (precise delta):** in a deep, end-to-end-trained network, does **phase-coupling (synchrony) itself** reduce catastrophic forgetting on standard CL benchmarks — *over and above* the structured sparsity, spherical vector-coding, and recurrent dynamics that AKOrN also introduces?

**Gate:** M1 greenlights M2/M3 or pivots the arc. The decisive quantity is a single increment on the ladder: **R6 − R5**.

---

## 0. The attribution problem (why v1 was blocked)

AKOrN differs from a plain sparse net on ≥4 axes *simultaneously*, each independently known to affect forgetting/representation-overlap:
(a) structured sparsity · (b) N-dim **spherical unit-vector** coding · (c) per-step **L2/spherical renormalization** · (d) **T-step recurrent fixed-point** dynamics (learned coupling J + natural-frequency Ω) · then (e) **phase-coupling = synchrony**.
Controlling only (a) leaves (b)(c)(d) as confounds. The fix is to add the ingredients one at a time and read synchrony off the final step.

## 1. Hypotheses (pre-registered)

- **H1:** synchrony reduces forgetting vs a *fully-matched non-synchrony* backbone (ladder rung R5).
- **H2 (decisive):** the **synchrony increment R6 − R5** is positive and significant — adding phase-coupling on top of an otherwise-identical normalized, vector-coded, recurrent, structured-sparse backbone reduces forgetting.
- **H3 (mechanism, discriminative):** the inter-task representation-overlap reduction is present in R6 but **absent/smaller in R5** (difference-in-differences), and a **synchrony-specific observable** (phase-cluster assignment stability across tasks) predicts forgetting after partialling out sparsity + normalization.
- **Nulls (via TOST, not failed significance):** "synchrony ≈ geometry" (R6 ≈ R5) and "no CL benefit" (R6 ≈ R1) — each an equivalence claim with a pre-set margin.

## 2. The control ladder (replaces v1's flat arm matrix)

One identical training recipe; each rung adds **exactly one** ingredient. Built by editing `source/models/classification/knet.py` (see §4).

| Rung | = previous + | isolates |
|---|---|---|
| **R1** | dense scalar baseline (ReLU/GeLU) | floor |
| **R2** | + structured/grouped **k-WTA sparsity**, matched to AKOrN fixed-point fraction-active (grouped to mirror oscillator grouping) | sparsity |
| **R3** | + **N-dim grouped (vector) coding** (channel groups of size N) | vector coding |
| **R4** | + per-step **spherical L2 normalization** on those groups | normalization geometry |
| **R5** | + **T-step recurrent fixed-point** with learned coupling J and an Ω-equivalent learnable bias, **no phase term** (dynamics preserved, synchrony destroyed) | recurrence |
| **R6** | **+ phase-coupling = full AKOrN** | **synchrony** |

**Synchrony's causal effect = R6 − R5.** Secondary informative increments: R4−R3 (normalization), R3−R2 (vector coding), R6−R2 (the "vs plain sparsity" number, for context only), R1 (floor).
**Internal negative control:** R6-scrambled = R6 with frozen/scrambled coupling phases (iteration + projection kept) — should behave like R5 if synchrony is the active ingredient.

## 3. Reference baselines (context, **not** the gate)

- **A0** Naive SGD (floor). **A4** EWC + small-buffer replay (situate effect sizes).
- **A5** Strong modern non-oscillatory CL SOTA on the same backbone (DER++/FOSTER/prompt), with & without replay — the **absolute-performance anchor**. GREENLIGHT requires R6 *competitive* with A5 within a stated margin, not merely R6 > R5.

## 4. Engineering — "build", not "locate" (red-team-corrected)

- **No param-matched non-Kuramoto classification control ships.** `ItrSA` is object-discovery only (`train_obj.py --model=vit`); `train_classification.py` imports only `AKOrN` and is CIFAR-10-only. **R1–R5 must be built** by stripping/replacing the KLayer Kuramoto update in `knet.py`, keeping strided convs, readout blocks, norm, and Linear head **bit-identical**. Validate each rung's param **and FLOP** count matches R6 within ~2% (decompose Ω/J/C; add compensating width/MLP if short). An unmatched rung is reported as a capacity-confounded bound, not a clean control.
- **Deterministic eval:** `knet.py` re-inits oscillators randomly each forward (`x = torch.randn_like(c)`) → stochastic outputs corrupt BWT/CKA. Fix the init seed across eval forwards **or** use fixed-N-init logit-averaging, and give every feedforward rung the **same N-forward averaging budget**. Compute CKA on the post-convergence state averaged over fixed inits. Track inference-init as a confound.
- **Normalization:** shipped default is BatchNorm (a documented CL confound via running-stat drift). Switch **all rungs to GroupNorm** (`--norm gn --c_norm gn`); report a BN-vs-GN ablation for R6.
- **Class-IL head:** shipped head is a fixed `nn.Linear` → implement a growable/masked head (Avalanche `IncrementalClassifier` or multi-head).
- **Test-time compute:** confirmatory contrast at **fixed equal T_eval, NO energy voting** (E-vote is Sudoku-only; classification has only logit-ensemble), **matched ensemble budget** across rungs. Extended-T / TTO numbers = clearly-labeled **secondary** only, never a gate input.

## 5. Benchmarks, scenarios & positive control

- **Arbiter: class-IL.** Task-IL = diagnostic only (saturates; excluded from gating if any rung >95%).
- **Primary:** Split-CIFAR-100 (10×10) class-IL **+ a longer/harder co-primary** (Split-CIFAR-100 20×5 and/or Split-TinyImageNet). **H2 must replicate in sign across ≥2 stream lengths/datasets.**
- **Sanity:** Split-MNIST. **Harness:** Avalanche primary; Mammoth cross-check.
- **Positive control (proves detection power):** a task where binding-by-synchrony is known to confer separability (synthetic multi-object / Sudoku-derived sequential stream) run through the *identical* BWT/CKA pipeline. **PIVOT-A/B may only be declared if the positive control PASSES** *and* the CL effect is null under TOST — so an underpowered null is never mistaken for a scientific conclusion.

## 6. Metrics

- **Primary endpoint (single, pre-registered):** class-IL **average Forgetting** (or final-average-ACC). BWT/learning-accuracy secondary (BWT ≈ Forgetting monotone — don't double-count).
- **Plasticity guard:** an H2 win counts **only if** R6's per-task learning accuracy is TOST-**equivalent** to R5 — a forgetting win can't be bought by underfitting.
- **H3 primary:** seed-paired inter-task **linear-CKA** (subsample-matched) on a fixed held-out probe, directional prediction + threshold, as a **difference-in-differences** (R6 vs R5); plus the phase-cluster-stability observable predicting forgetting after partialling sparsity+norm. Other overlap metrics exploratory.

## 7. Statistics (objective gate)

- **≥10 seeds** (15–20 for the decisive R6−R5); each seed keys data-order + init + augmentation RNG (genuinely paired).
- **Exact paired permutation / sign-flip test** (n=5 Wilcoxon floor is 2/2³² ≈ 0.0625 — cannot reach p<0.05) or a mixed-effects model with task as random effect.
- **Pre-hoc power/MDES** from a Split-MNIST pilot: "with n seeds and SD σ we detect ΔForgetting ≥ X at 80% power"; X = the pre-registered **SESOI**.
- **Multiplicity:** confirmatory family = the decisive increments × 1 primary metric × 1 primary scenario; **intersection-union** for conjunctions; Holm within family. Everything else exploratory.
- **Equivalence:** **TOST** with margin Δe for the PIVOT nulls.

## 8. Decision gate (pre-register the numbers before any CIFAR run)

Primary endpoint = class-IL average Forgetting; primary scenario = Split-CIFAR-100 class-IL. *(Fill Δg, Δe, SESOI from the Wk-0 pilot.)*

- **GREENLIGHT M2** iff **R6 − R5 ≥ Δg** (e.g. ≥3 abs pts forgetting reduction OR Cohen's d ≥ 0.8 on seed-paired diffs), permutation p<0.05; **AND** sign-replicates on the longer stream; **AND** plasticity guard holds; **AND** H3 difference-in-differences positive; **AND** R6 competitive with A5 within margin.
- **PIVOT-A ("synchrony ≈ geometry")** — |R6 − R5| within ±Δe via TOST (90% CI), positive control passing → publish the negative; redirect to *"what does phase add beyond normalized vector-coded recurrence?"* — the ladder's R3/R4/R5 increments *are* that decomposition (a contribution in itself).
- **PIVOT-B ("no CL benefit")** — R6 ≈ R1 under TOST → first clean synchrony-on-CL benchmark + mechanistic why-not; reconsider a non-AKOrN oscillatory substrate.
- **INCONCLUSIVE** (Δe < |effect| < Δg) → add seeds; never force a call.

## 9. Novelty framing (reviewer defense)

Claim the precise delta — *"first deep, end-to-end gradient-trained, parameter- **and normalization- and sparsity-**controlled test of a binding-by-synchrony backbone on standard CL benchmarks, isolating synchrony's marginal contribution via a one-ingredient ladder."* **Not** "first synchrony-for-CL." Related-work paragraph positions vs Verbeke & Verguts 2019, Phasor Agents, reservoir-CL — stating why none answers the R6−R5 question.

## 10. Timeline (rescoped — realistic)

- **Wk 0 — integration spike (go/no-go):** AKOrN T-step forward runs inside Avalanche on a toy task, gradients flow, deterministic-eval hook + growable head work (validate on Split-MNIST); reproduce **native CIFAR-10 classification** (the actual codepath); run **one full R6 Split-CIFAR-100 seed** → measure wall-clock + peak VRAM → concrete GPU-hour budget for the whole matrix.
- **Wk 1–2:** build + validate ladder R1–R6 (param/FLOP-matched), GroupNorm, deterministic eval, CIFAR-10→100 head port.
- **Wk 2–3:** class-IL Split-CIFAR-100 (R1–R6 + A0/A4/A5), ≥10 seeds on the decisive contrasts.
- **Wk 3–4+:** longer-stream replication + positive control + H3 + statistics + **gate**.

Realistic total **~5–7 weeks** (not 4) given the build-not-locate scope; T_eval sweep, optional rungs, and FlyPrompt are out of the critical path.

## 11. Deliverable

Ladder table (R1–R6 + A0/A4/A5 × class-IL Forgetting/final-ACC × 2 stream lengths), the **R6−R5 synchrony increment** with permutation CI, the difference-in-differences H3, the positive-control result, an audited confound log (params, FLOPs, the four sparsity metrics, norm type, lr/opt/init, inference-init/ensemble budget, effective DOF/subspace rank, inference FLOPs), and a logged **GREENLIGHT / PIVOT-A / PIVOT-B / INCONCLUSIVE** decision.

---

*Status: v2, revised against red-team (verdict v1 = blockers_present, all addressed). Next gate = Wk-0 integration spike — real code, not more planning.*
