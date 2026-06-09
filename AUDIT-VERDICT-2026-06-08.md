# Code-audit + verification verdict (2026-06-08)

Independent 5-reviewer adversarial code audit flagged 3 SERIOUS risks. We ran the audit's own verification tests
on each. Outcome: **the empirical core SURVIVES, but three interpretive claims must be tightened.** Honest.

## Risk #1 — "BINDING" overstated  → REFRAME (the most substantive fix)
- Audit: class = unordered type-SET; task may be presence-solvable, not binding.
- Verify (step24 part B): a presence-only detector reaches **0.343** (chance 0.10) — so the task is NOT trivially
  presence-solvable (full perf 0.95 needs more than crude presence), BUT presence gets a meaningful chunk, and we
  have NOT proven figure-ground binding is the mechanism.
- VERDICT: drop strong "binding mechanism" language; use **"multi-object"**. To EARN "binding," add a task that
  genuinely requires segregation (same-different / relational / count-of-same-type). [open experiment]

## Risk #2 — controls UNDER-LEARN → NECESSITY HOLDS, state with caveat
- Audit: plainCNN learn_acc ~0.47 << R6 0.99, so "+0.85 collapse" may be a learning failure not forgetting.
- Verify (step26): TUNED plainCNN (lr 1e-3, ctx 32) fits a SINGLE task at **0.742** (can learn), but in full CL
  STILL **forgets to chance (final 0.109, forget 0.46)**. R6 retains 0.95.
- VERDICT: necessity SURVIVES — a tuned non-oscillator forgets to chance even when it can fit tasks. CAVEAT:
  plainCNN fits somewhat worse in isolation (0.74 vs 0.96), so the gap is not PURELY retention. State as: "tuned
  non-oscillator generators forget to chance even when they fit individual tasks; the oscillator's edge is
  retention/stability under continual co-training." Consider a capacity-matched control for the camera-ready.

## Risk #3 — UNBYPASSABILITY under learned trunks → DROP one overclaim; headline clean
- Audit: learned trunk lifts plainCNN 0.11->0.41 → trunk leaks; step23 "stays low" claim false.
- Verify (step24 part A): linear probe trunk-features->class: RANDOM trunk **0.104** (=chance, truly
  task-agnostic/unbypassable ✓); CIFAR-AE learned trunk **0.149** (barely above chance). So the learned trunk is
  only MILDLY informative; plainCNN's 0.41 there is mostly its legit co-trained context, not a trunk bypass.
- VERDICT: the RANDOM-trunk headline is cleanly UNBYPASSABLE (keep). DROP the "unbypassable with a LEARNED trunk"
  claim — present learned/transfer trunk as ROBUSTNESS (bypass survives realistic features) NOT as unbypassable.

## Minor (hygiene)
- m2_hypernet_joint.json has a stale smoke-record (negative-lift outlier) → dedup before any M2-channel claim.
- TIGHT-vs-easy provenance: ensure the TIGHT headline cites step13_* (TIGHT=True), not step9_* (TIGHT=False).

## NET HONEST STATUS OF THE RESULT
SURVIVES: co-trained oscillator phase-context resists catastrophic forgetting in fully-online multi-object CL,
where tuned non-oscillator generators forget to chance; effect generalizes to real datasets (MNIST/Fashion);
not sparsity; cleanly unbypassable with a random trunk; binding-DEMAND curve (inverted-U) is real.
MUST TIGHTEN (before claims): "binding"->"multi-object" (or add a genuine-binding task); necessity stated with
the fit-caveat; learned-trunk = robustness not unbypassability; dedup M2 file.
=> The result is REAL and defensible with HONEST claims. It is NOT the unqualified "synchrony-binding-necessary-
unbypassable" version. This is a stronger position than over-claiming and getting caught at review.

## REVISED PLAN (before heavy tier C)
1. [reframe] revise PAPER-OUTLINE claims per above (multi-object; careful necessity; robustness-trunk).
2. [1 experiment, high-value] a GENUINELY-binding task (same-different or count-of-same-type, where presence is
   insufficient) — if R6 still wins there, we EARN "binding"; if not, stay "multi-object".
3. [optional] capacity-matched control (plainCNN that fits as well as R6 in isolation) for clean necessity.
4. THEN (c): Tetrominoes/CLEVR + CL baselines, on the corrected foundation.

## CONJUNCTION-BINDING TEST RESULT (n=4, firming to 12) — "BINDING" EARNED (at the oscillator level)
Presence-matched 3-object color×shape permutation task; presence-detector = 0.163 (= chance 1/6) PROVES the task
is binding-required. M3-online: R6 final=0.423, R6s 0.485, plainCNN 0.189 (~chance). R6-plainCNN +0.234 (4/4).
=> The OSCILLATOR (R6 AND R6s) performs genuine feature BINDING that a feedforward net cannot (0.42-0.49 vs 0.19
on a presence-proof task). EARNS "binding". But it is OSCILLATORY binding (Kuramoto phase dynamics), NOT learned-
synchrony-specific (R6s = random coupling binds equally/slightly better). Honest framing: "oscillatory phase
dynamics perform feature binding feedforward nets cannot; binding-capable context resists forgetting in online
CL; learned coupling is a task-dependent extra, not required for the binding." Resolves audit risk #1 FAVORABLY:
task genuinely requires binding AND the oscillator does it. CAVEAT: modest absolute (0.42-0.49 on 6-way); n=4->12.

## (c) CL BASELINE PLACEMENT (n=4, firming competitors to n=12) — OURS BEATS DER at matched memory
2-digit MNIST, class-incremental (label-free), chance 0.10: Naive 0.139, EWC 0.126, SI 0.139, LwF 0.121 (reg
methods FAIL class-IL as expected), Replay(mem300) 0.439, DER(mem300) 0.574. OURS oscillator-R6 = 0.646 (>DER),
ours-plainCNN-context = 0.191 (~reg level). => our oscillator-context method BEATS standard CL (incl DER) at
MATCHED replay memory; the OSCILLATOR is what lifts us above baselines (plainCNN-ctx sits at reg level). CAVEATS:
n=4->12 (competitors); different backbones (method comparison not identical-arch); baselines standard-but-unswept.

## (c) REAL TETROMINOES + BASELINE PLACEMENT — IMPORTANT HONEST CORRECTION (2026-06-08)
Real DeepMind Tetrominoes (red-shape CL, presence-leak 0.49, class-incr, mem=300):
  ours R6 0.585 / R6s 0.608 (n=12) | DER 0.872 / Replay 0.605 / Naive 0.255 (n=8) | plainCNN-ctx 0.168 (chance).
- DER BEATS ours: R6-DER -0.276 p=0.016; R6s-DER -0.224 p=0.008. The MNIST-2obj "ours beats DER" (+0.07) does
  NOT generalize -- on real images DER's TRAINABLE CNN backbone wins; our FROZEN-RANDOM trunk (chosen for
  unbypassability) is the ceiling. => DO NOT claim "beats SOTA CL". We are NOT competitive with DER on real data.
- WHAT SURVIVES (the real contribution): R6 vs MATCHED feedforward control (plainCNN-ctx) = +0.417 12/12 p=0.0005
  on real Tetrominoes -> oscillator retains where matched feedforward forgets to chance. The MECHANISM claim holds
  on real images; the SOTA-competitiveness claim does not.
- VENUE IMPLICATION: confirms MECHANISM paper (oscillator-vs-feedforward, why) NOT a SOTA method. Reinforces
  TMLR/CoLLAs/workshop over a NeurIPS-main "beats baselines" story (a main-track SOTA claim dies to this DER row).
- HONEST framing for paper: report DER>ours plainly as a limitation; lead with the controlled oscillator-vs-
  feedforward mechanism + binding-earned + audit-survival, explicitly "mechanism, not SOTA".
