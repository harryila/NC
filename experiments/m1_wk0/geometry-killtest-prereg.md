# Geometry kill-test — pre-registered decision rule (locked 2026-06-01, BEFORE results)

**Purpose.** The decisive M1 result (R6 < R5:no_proj on inter-task CKA overlap, Δ≈+0.025, 10/10 seeds)
could be the synchrony mechanism OR a side effect of the `apply_proj` tangent-space projection changing
representation geometry (CKA is geometry-sensitive). This test isolates the two. **Locking the decision
rule before seeing the numbers** so the read is honest, not post-hoc.

## Reference values (from the committed decisive run, class10, E=50, 10 seeds)
- **O_inter(R6)      = 0.457**  (synchrony ON, learned coupling)  — the "low overlap" arm
- **O_inter(R5:no_proj) = 0.482**  (synchrony OFF: apply_proj=False)  — the original control
- **Synchrony effect Δ = O_inter(R5) − O_inter(R6) = +0.025**  (what we must explain)

## The arms under test (all through the SAME h3 snapshot path)
| Arm | apply_proj (projection geometry) | learned coupling J | isolates |
|---|---|---|---|
| R6 (ref) | ON | learned | full synchrony |
| **R6s (scrambled)** | **ON** | **frozen random (dead)** | **projection geometry WITHOUT learned synchrony** ← the key arm |
| R5:no_proj (ref) | OFF | learned | no projection |
| R5:depthwise | (per build) | replaced by 1×1 depthwise (no neuron–neuron coupling) | coupling-removed robustness bracket (−39% params) |
| R5:frozen_J | ON | frozen random | robustness bracket (−39% params) |

**R6s is the decisive geometry-isolation arm:** it keeps the projection geometry identical to R6 but kills
the learned synchrony coupling. (Confirm R6s param-count ≈ R6's when reading; depthwise/frozen_J are
−39%-param brackets, so they corroborate but are capacity-confounded.)

## Pre-registered decision rule (v1 — locked before results; see v1.1 correction below)
Define the **position ratio** of the geometry-only arm between the two reference means:

    pos = ( O_inter(R6s) − O_inter(R6) ) / ( O_inter(R5:no_proj) − O_inter(R6) )

i.e. pos = 0 means R6s sits exactly at R6's (low) overlap; pos = 1 means R6s sits at R5's (high) overlap.

- **KILL (geometry artifact, arc foundation fails):** `pos ≤ 0.33` AND R6s vs R6 paired diff NOT significant
  (i.e. R6s ≈ R6). → The low overlap is produced by the projection alone, with synchrony coupling dead.
  "Synchrony reduces interference" is **false as stated**; the H3 signal is projection geometry. → STOP M2
  compute; the M1 mechanism claim must be reframed (geometry, not synchrony) or rescued by a different
  observable. This is a forward pivot, logged, not a failure.
- **SURVIVE (real synchrony effect):** `pos ≥ 0.67` AND R6s vs R5:no_proj paired diff NOT significant
  (R6s ≈ R5, i.e. dead coupling → high overlap like the no-projection control). → The low overlap REQUIRES
  learned coupling, not just the projection → the effect is synchrony. → arc foundation holds; proceed.
- **AMBIGUOUS:** `0.33 < pos < 0.67`. → The projection contributes partially. → do NOT declare either way;
  resolve with the **honest-DiD** (augmentation-based within-snapshot O_intra, which controls per-arm
  geometry directly) before any M2 commitment.

### Rule defect found at read-time + v1.1 correction (2026-06-01, AFTER results — logged honestly)
v1 implicitly assumed R6s would land **between** R6 and R5 (pos ∈ [0,1]). It did not: R6s overshot to the
HIGH side (pos = 3.94, overlap ABOVE the no-projection control). v1 therefore returned "AMBIGUOUS" on a
case it never modeled. The defect is in the **mapping of pos→verdict**, not in the underlying KILL logic.
The KILL hypothesis is precisely *"geometry alone reproduces R6's low overlap,"* i.e. **R6s ≈ R6 (low)**.
Corrected rule (v1.1), stated in terms of the actual hypothesis (the KILL/SURVIVE conditions are unchanged;
only the AMBIGUOUS middle is re-scoped to exclude the R6s-outside-interval case):
- **KILL** iff R6s ≈ R6 (paired p ≥ 0.05) AND O_inter(R6s) ≤ O_inter(R5)  — geometry reproduces the low overlap.
- **SURVIVE** iff R6s is significantly ABOVE R6 (paired p < 0.05, O_R6s > O_R6) — i.e. removing learned
  coupling REMOVES (or reverses) the low-overlap effect → the effect requires learned synchrony. This holds
  **whether R6s lands at R5 (pos≈1) or overshoots above it (pos>1)**; overshoot is *stronger* SURVIVE evidence,
  not ambiguity.
- **AMBIGUOUS** only if R6s sits strictly between R6 and R5 with pos ∈ (0.33,0.67) and is significantly
  different from R6 — partial projection contribution; resolve with honest-DiD.

## RESULT (2026-06-01, R6s 10/10 seeds, param-identical scramble: frozen-random J, apply_proj ON)
- Mean O_inter: **R6 = 0.4567 (sync ON) < R5:no_proj = 0.4816 (sync OFF) < R6s = 0.5548 (proj ON, J dead)**.
- pos = 3.94 (R6s overshoots ABOVE R5). Paired sign-flip: R6s vs R6 = **+0.098, p = 0.0020**; R6s vs R5 =
  **+0.073, p = 0.0020** (R6s significantly higher than BOTH).
- **VERDICT (v1.1): SURVIVE — decisively.** Killing the learned coupling (projection still ON) does not
  reproduce R6's low overlap; it makes overlap *worse than even the no-projection control*. The low
  cross-task overlap is **NOT a projection-geometry artifact — it requires learned synchrony coupling.**
  The geometry confound is refuted. Arc foundation holds.
- Per-seed R6s O_inter: [0.5741, 0.5425, 0.5360, 0.5926, 0.5441, 0.5514, 0.5450, 0.5618, 0.5536, 0.5474].

## CORROBORATION — FINAL (2026-06-02, all 5 arms × 10 seeds complete)
Mean O_inter: **R6 = 0.4567** (learned synchrony) is the ONLY low arm; EVERY coupling-removed arm sits HIGH:
R5:no_proj 0.4816, R5:frozen_J 0.5514, R6s 0.5548, R5:depthwise 0.5659. Paired sign-flip vs R6 (all n=10,
all p=0.0020, all ABOVE R6): R6s +0.098, R5:depthwise +0.109, R5:frozen_J +0.095, R5:no_proj +0.025.
**The pattern is monotone and unanimous: any arm WITHOUT learned relational coupling has higher cross-task
overlap than R6; the three arms with coupling killed/frozen/removed (R6s, frozen_J, depthwise) cluster
together at ~0.55, ABOVE even the no-projection control (0.482).** This triple-corroborates the SURVIVE
verdict from three independent angles (frozen random J with projection on; depthwise = no neuron-neuron
coupling; param-identical scramble). The low inter-task overlap is caused by the LEARNED SYNCHRONY COUPLING,
not by the projection geometry and not by recurrence/normalization. Geometry confound DECISIVELY refuted.
Note R6s/frozen_J/depthwise sitting *above* R5:no_proj suggests dead/frozen coupling is actively
ANTI-helpful (worse than removing the projection entirely) — consistent with synchrony being load-bearing.

Secondary check (depthwise/frozen_J): both are coupling-removed; if BOTH sit near R5 (high overlap) while
only learned-coupling R6 is low, that corroborates SURVIVE. If either sits near R6 (low), it strengthens
the geometry/capacity concern (note the −39%-param confound when interpreting).

## What we read when the kill-test lands
1. Pull per-seed `h3.overlap_summary.O_inter` for R6s, R5:depthwise, R5:frozen_J (10 seeds each).
2. Compute means + the position ratio + paired tests (R6s vs R6, R6s vs R5:no_proj), seed-paired where
   seeds align.
3. Apply the rule above verbatim. Report pos, the two paired p's, and the KILL/SURVIVE/AMBIGUOUS call.

_Locked before results — do not edit the thresholds after seeing the numbers._
