# Main-track campaign (NeurIPS/ICLR) — gated plan (2026-06-08)

DECISION: pursue a confident main-track submission, not a coin-flip. Per the honest assessment, that requires
(1) closing the DER gap on REAL data (learned backbone), (2) a larger/longer benchmark, (3) swept baselines
across ≥2 datasets, (4) the CCC/Impossibility-Triangle theory hook made rigorous. The strongest cards are the
EARNED binding result + the CCC theory. This is weeks of work; we GATE it so a dead crux kills the path cheaply
before we spend on scale. Pre-register every gate's decision rule BEFORE seeing results (project lesson).

## The blocker we must beat
On real Tetrominoes, DER 0.87 >> ours 0.59. Cause hypothesis: our FROZEN-RANDOM trunk is the ceiling (DER trains
its whole CNN). If a LEARNED task-agnostic trunk closes the gap, main-track is viable. If not, the method itself
is the limit and we revert to the mechanism-paper (TMLR/CoLLAs) plan.

## DESIGN FORK to resolve at GATE 0 (state explicitly in paper)
- Unbypassability requires a TASK-AGNOSTIC trunk. A self-sup (label-free) AE trunk stays task-agnostic, so it
  PRESERVES the unbypassability story while giving realistic features. A fully TRAINABLE/co-trained trunk would
  likely be most competitive but RELAXES unbypassability (the channel is no longer the only adaptive path).
- Plan: lead with self-sup-frozen trunk (keeps the clean story); report trainable-trunk as an upper-bound variant.

---

## GATE 0 (CRUX — running now): does a LEARNED realistic backbone close the DER gap? [step33]
- Self-sup AE trunk (frozen, task-agnostic) on tetrominoes images → R6/R6s/plainCNN on real red-shape task.
- PRE-REG RULE:
  - VIABLE  := learned-trunk R6 or R6s reaches ≥ 0.80 (within ~0.07 of DER 0.87) AND plainCNN stays low (no leak).
    → main-track path is live; proceed to GATE 1.
  - PARTIAL := lands 0.65–0.80 → competitive-ish; main-track possible if scale + theory carry it. Proceed but flag.
  - DEAD    := stays < 0.65 (≈ frozen-random) → the gap is NOT the trunk; the method caps below SOTA on real data.
    → STOP the main-track push; the honest home is TMLR/CoLLAs (mechanism paper). Do NOT force it.
  - If plainCNN ALSO jumps (learned trunk leaks task info) → comparison confounded; the "necessity" reads weaker.
- Also run the trainable-trunk upper-bound variant for context.

## GATE 1 (conditional on GATE 0 VIABLE/PARTIAL): a larger/longer CL benchmark
- Option A (preferred, lower-risk): a 50-class / 20-task multi-object sequence built from existing real data
  (compose MNIST+Fashion+Tetrominoes, or Tetrominoes with finer shape×color classes) — controllable, validated.
- Option B (higher signal, higher risk): REAL CLEVR (download, derive an object-centric classification CL task,
  sanity-check like we did for Tetrominoes). Heavier; only if A lands and time allows.
- Validate the benchmark (a strong baseline must learn it) before trusting any arm.

## GATE 2: swept baselines across ≥2 datasets (fairness)
- Properly TUNE DER/Replay/ER-ACE/EWC/LwF (sweep key hyperparams, not the defaults we used). Compare ours vs
  tuned baselines on ≥2 datasets (the scale benchmark + Tetrominoes/MNIST). Same memory budget, same backbone
  where possible. Report honestly — including where baselines win.

## GATE 3: the CCC / Impossibility-Triangle theory hook (a main-track card)
- Measure the Context Channel Capacity C_ctx = I(phase-context; generated params) of the oscillator vs feedforward
  context; show oscillator's C_ctx is higher and tie to the zero-forgetting bound (Cheng 2026). This makes the
  contribution THEORETICAL+empirical, not just empirical — the differentiator for main track.

## GATE 4: write the paper (only after 0–3 land)
- Full draft from PAPER-OUTLINE.md + MAIN-TRACK additions; figures; a 2nd independent code-review pass.

## Guardrails (unchanged, non-negotiable)
- NO commits/push without explicit say-so. Memory-frugal GPU (no OOM). Pre-register gate rules. Adversarially
  verify every positive. Distinguish real/modest/null honestly. Pull box results local before analysis.

## STATUS LOG
- 2026-06-08: campaign opened. GATE 0 (step33 learned-trunk on real Tetrominoes) launched. Awaiting crux verdict.

## GATE 0 RESULT (2026-06-08) — VIABLE (crux passed). Learned trunk CLOSES + EXCEEDS the DER gap.
Self-sup AE trunk (frozen, task-agnostic) on REAL Tetrominoes, n=4: R6 final=0.926 (learn 0.988), R6s 0.741,
plainCNN 0.311. vs frozen-RANDOM (R6 0.585) and DER 0.872. => the DER gap WAS the frozen-random trunk; with a
realistic self-sup backbone R6 reaches 0.926 (> DER ref). Mechanism intact: plainCNN with SAME trunk still
collapses (0.57->0.31), R6-plainCNN +0.62. MAIN-TRACK PATH IS LIVE.
CAVEATS (must address): (1) FAIRNESS — R6 got self-sup pretraining DER did not; GATE 2 MUST give DER the same
AE-pretrained backbone for an honest head-to-head (do NOT claim "beats DER" until then). (2) learned trunk relaxes
unbypassability (plainCNN 0.17->0.31 > chance) -> unbypassable = random-trunk claim; competitive = learned-trunk.
Firming to n=12. Next: GATE 1 (scale benchmark) + GATE 2 (fair pretrained-backbone baselines).

## GATE 0 CLEAN + fairness caught (2026-06-08) — VIABLE = PARITY with DER (NOT a crush)
Clean (train-only SSL, no leak) learned trunk, REAL Tetrominoes n=12: R6 0.900, R6s 0.774, plainCNN-ctx 0.292.
R6-plainCNN +0.608 12/12 p=0.0005 (mechanism intact). vs DER-best (step32 own CNN) 0.872: paired R6-DER +0.069
8/8 p=0.008 -> COMPETITIVE/slight edge. The frozen-random gap (DER 0.87 vs ours 0.59) was THE TRUNK -> closed.
ARTIFACT CAUGHT + DISCARDED: step34 "fair DER" gave DER 0.167-0.199 (chance) -- because AEBackbone+globalpool+
linear head cripples DER (no depth/spatial); contradicts our own DER=0.872. NOT a fair test; do NOT claim "crush".
Our method works on the same frozen AE features only because the OSCILLATOR adds spatial processing a linear head
lacks (a real finding, but not a DER comparison).
OPEN for GATE 2: (1) SSL confound -- R6 got unsupervised pretrain DER didn't; a clean same-backbone DER+SSL test
still needed. (2) one task. => CLAIM TODAY = "parity with DER given a learned backbone", not "beats DER".
NEXT: GATE 1 (scale: more classes/tasks on real Tetrominoes or CLEVR) -- the real main-track signal -- then GATE 2
(proper fair baselines on the scaled benchmark) + GATE 3 (CCC theory).

## GATE 1 RESULT (2026-06-08) — MIXED: method SCALES (no degradation) BUT red-shape task SATURATES -> DER wins
REAL Tetrominoes 12cls/6task, adequate replay (mem 360), chance 0.083: ours R6 0.964, R6s 0.935, plainCNN-ctx
0.718 | baselines DER 0.989, Replay 0.970, Naive 0.154. R6-plainCNN +0.246 (mechanism holds). R6-DER -0.025 (DER
slightly ahead). => GOOD: ours scales without degradation (0.90@6/3 -> 0.96@12/6; the scale-collapse fear is GONE
with adequate replay). BAD: red-shape task SATURATES (DER 0.99, Replay 0.97) -> no room to differentiate, DER>=ours.
KEY INSIGHT (reshapes thesis): across ALL data, ours wins on HARD/low-data/binding (MNIST-2obj 0.65 vs DER 0.575),
loses on EASY/saturated (Tetro-scale 0.96 vs DER 0.99). DER's replay wins whenever a task is easily learnable.
=> Main-track case CANNOT be "beat SOTA everywhere". It MUST be "a class of BINDING-REQUIRED tasks where standard
CL FAILS and oscillatory dynamics SUCCEED". PIVOT: test DER/Replay on the presence-proof conjunction-binding task
(step36). If DER ~chance there (can't bind) while R6=0.43 -> THE main-track differentiator. [running]

## FINAL VERDICT (2026-06-08) — MAIN-TRACK NOT SUPPORTED. Lock the mechanism paper. (after 30-agent probe + 3 pre-gates)
Probe (wq1s1kvke, 30 agents) + 3 pre-registered pre-gates all confirm: no regime where we robustly beat DER.
- #1 matched-n: ours-Tetro 0.900 vs DER 0.882 at FULL n=12 = +0.018 p=0.86 (NS). The earlier "+0.069 8/8 p=0.008"
  was a seeds-0-7 subset cherry-pick (dropped our collapsed seed 8=0.397). Honest = PARITY.
- #2 SSL: self-sup AE backbone HURTS a standard CNN (single-task 0.60->0.50, negative transfer) -> can't fairly
  hand DER our backbone; comparison stays ours-SSL vs DER-vanilla = parity. (Minor: ours can use a frozen SSL
  backbone DER can't.)
- #3 CROSSOVER (decisive): binding, mem 300->6: ours-DER goes -0.05 -> +0.03(R6)/+0.05(R6s); interaction +0.08.
  DIRECTION matches CCC (ours starves slower) BUT below the pre-reg GO bar (+0.10 & sig); at mem6 BOTH ~chance
  (0.22-0.27). Suggestive trend in a degenerate regime, NOT a main-track result. NO-GO.
=> MAIN-TRACK probability was ~10-15%, hinging on #3; #3 sub-threshold -> DONE. LOCK TMLR/CoLLAs mechanism paper.
PAPER SPINE (probe-recommended): lead with (1) oscillator R6/R6s >> plainCNN-ctx matched feedforward control,
p=0.0005 EVERYWHERE (the robust necessity); (2) EARNED binding (presence-proof conjunction vs matched control);
(3) UNBYPASSABILITY as a constraint-satisfaction property DER structurally cannot satisfy (the novel angle, NOT a
leaderboard); (4) HONEST parity-with-DER incl. where DER wins (scale-saturation, binding); (5) honest limits
(single-seed collapse/instability, replay-shared-buffer = no structural memory edge). DROP all "beats SOTA".
INTEGRITY fixes to apply in the paper: report full-n only, no subset paired tests; state Tetro as parity; the
crossover as a consistent-with-CCC trend not a win; flag the collapse rate openly.
