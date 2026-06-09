# Morning report draft (building through the night 2026-06-07) — honest arc state

> Living doc. Updated as each step lands. Every number adversarially checked (exact paired test, dedup,
> chance floor, R6s control). Final report compiled at ~10am.

## TL;DR (FINAL for M1/M2, n=20 verified from saved box artifacts)
- **M1 — SOLID positive.** Head-free interference reduction, dz≈2.28. Geometry kill-test survived. Untouched.
- **M2 — POSITIVE with nuance (this is the honest, triple-checked verdict).** Learned synchrony (R6) gives a
  usable label-free context channel for parameter generation, significantly > frozen-random coupling (R6s):
  - JOINT ctx-lift n=20: R6 0.287 vs R6s 0.107, 19/20 seeds, exact p≈0.0000. Context task-necessary (wrong-task
    dP5 R6 −0.41). Unbypassability holds (const ctx → chance).
  - DECOMPOSITION (within-task-shuffle, n=20): the advantage is BOTH **task-level** (R6 0.141 vs R6s 0.071,
    +0.071, 18/20, p≈0) AND **instance-level** (R6 0.186 vs R6s 0.104, +0.082, 19/20, p≈0). ~46% task / 54%
    instance. So there IS a genuine synchrony-specific task-level channel.
  - CAVEATS (honest): absolute task-level capacity is MODEST (task-only acc ~0.24 vs chance 0.10 vs oracle
    ~0.53); task-ID is NOT cleanly DECODABLE from phase (R6≈R6s, probe-fragile) — the channel carries
    task-USEFUL info a hypernet exploits, not a decodable task label.
- **M3 — open/hard, mechanistically explained.** Phase does NOT bypass online-CL forgetting. ORACLE control:
  perfect task context → ~0.53 retain (harness fine). The phase task-channel is real but too MODEST in absolute
  capacity to sustain sequential regeneration. [STEP 4b contrastive-AKOrN swing: attempting to AMPLIFY the
  task channel — the thesis payoff test.]

## Honest process note (for Harry)
Tonight I swung the M2 read THREE times — "null" → "solid" → "mostly instance-leak" → (final) "positive with
nuance". Each swing was corrected by a PROPERLY POWERED test (n=20), not by argument. The n=5 and n=3 prelims
repeatedly misled (both directions). The final verdict rests only on n≥17-20 exact-tested, artifact-backed
numbers. An adversarial workflow caught that my results lived only on the box (now pulled local) and that I'd
over-stated decodability as "flat null" (corrected to "weak/probe-fragile").

## What this session corrected (the double-check Harry asked for)
1. My own earlier "M2 ≈ null" read was WRONG — I mistook underpowered sub-tests (n=2-3) for nulls. M2 joint
   is decisively positive at n=20.
2. The "CORRECTED-NULL" decodability label is a math artifact of n=3 (sign-flip floor 0.125).
3. The relational_probe "NULL ROBUST" used the INVALID R5:no_proj ablation at n=2 — not the corrected test.

## Literature grounding (rooted-in-research, Valency)
- Verbeke & Verguts 2020 (pubmed 31430280): synchrony CAN be trained to encode task (synchronize-relevant /
  desynchronize-irrelevant) to solve stability-plasticity → our M3 null is the EXPECTED untrained baseline;
  an explicit task-synchrony objective is the principled fix.
- Vedovati & Ching 2024 (arXiv 2408.01316): contextual modulation for task-packing, context-ambiguity
  robustness. KoPE 2026 (2604.07904): Kuramoto phase-encoding in ViTs (possible later substrate).

## Honest design tension to resolve WITH Harry (do not paper over)
Thesis = "label-free context channel WITHOUT task labels." A contrastive/learned context encoder uses
CURRENT-TASK labels DURING training (legitimate in class-incremental CL; channel stays label-free AT TEST).
This weakens "no task labels" → "no task labels at inference." Must be stated plainly. The fully-unsupervised
claim is the harder one; the supervised-at-train version is a clean, defensible relaxation.

## Recommended framing (pending tonight's results)
Lock M1 (solid) + M2 (solid usable channel, modest) as the contribution. M3 as either (a) honest
negative-with-mechanism (overlap limit, oracle-validated) if STEP 4 is negative, or (b) a positive rescue via
a contrastive context-encoder if STEP 4a shows latent task-separability. Either is publishable and honest.

## Open / next
- STEP 3 verdict (decodability n=17).
- STEP 4a verdict (latent task info) → decides M3 branch.
- Conditional: STEP 4b (contrastive encoder M3) OR object-discovery swing OR consolidate.
