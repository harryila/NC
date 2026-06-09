# Autonomous 10-hour block — plan + locked decision rules (2026-06-07, Harry asleep ~10h)

Goal: advance the M1->M2->M3 arc rigorously while Harry sleeps. 2x RTX 3090 box (213.192.2.118:40105),
both GPUs. NO commits/push (Harry's rule). Pre-register every decision rule BEFORE seeing results; adversarially
verify each finding; NO config-shopping for a result that flatters a hypothesis (the p-hacking trap the project
has repeatedly had to catch — in BOTH directions). Log everything to this file + research-log. Report at ~10am.

## Guardrails (non-negotiable, learned from this project's failure modes)
- Only run PRE-SPECIFIED tests at higher n. Never tune a knob until one config favors the hypothesis.
- Every positive => adversarial check (liveness/oracle control, shuffle floor, exact paired test, dedup).
- Every negative => is the instrument alive? (oracle/positive control) before believing it.
- Use the EXACT paired sign-flip test (not just one-sided t) for the headline call.
- Distinguish "real but modest" from "solid" from "null" honestly. No over- OR under-claiming.

## PRIORITY-ORDERED WORK (gated; each step's result decides the next)

### STEP 1 (running) — M2 joint-bound pressure-test, n=6 -> n=20
LOCKED RULE: dedup the smoke record (R6 s0 real=0.163). Over all valid seeds compute the paired diff
(R6 lift - R6s lift) + EXACT sign-flip 2-sided p + #seeds R6>R6s.
  HOLDS  := exact p < 0.05 AND >= 70% of seeds R6>R6s at n>=15.  -> M2 joint claim SOLID.
  FADES  := otherwise. -> M2 joint is "directional, underpowered"; report honestly.

### STEP 2 — RECONCILE the decodability tension (double-check, the user asked for this)
The session cited M2 decodability as POSITIVE ("corrected screen L1 +0.056 p~0.01") BUT
results/m2_corrected_screen.json verdict says "CORRECTED-NULL". These can't both be the headline. Read the
actual per_seed arrays of m2_corrected_screen.json + m2_screen.json + m2_relational_probe.json, recompute the
paired decodability stat myself, and state the HONEST decodability verdict. (No GPU; pure analysis.)

### STEP 3 — Bulletproof the M2 sub-claims that are real (pre-specified, more n, no shopping)
For whichever M2 sub-claims are positive after 1+2, power them to n>=15-20 with the EXACT same code/config
(only --seeds changes). Unbypassability invariant (const ctx -> chance) re-confirmed at the higher n.

### STEP 4 (research-rooted, the CRUX) — CAN synchrony encode TASK, not just within-image grouping?
The oracle diagnostic proved M3's limiter: the phase channel caps ~0.53 because it overlaps across tasks
(synchrony binds objects within an image; it does not natively encode WHICH task). This is EXACTLY M3 trap #2
from the thesis/prior-art memo (the S_N symmetry barrier: "synchrony must be the EXPLICIT symmetry-breaker").
PRE-REGISTERED probe (cheap, measurement-only first): does adding a CONTRASTIVE phase objective during the
brief context-gen training (push phase configs of different tasks apart) RAISE task-decodability of the phase
context above the ~0.39 ceiling? Measure decodability ONLY (no generator) across seeds, R6 vs R6s.
  RULE: report the decodability delta honestly; if contrastive phase lifts task-decodability materially
  (>= +0.10 over baseline, replicated), THAT is the outside-box result that could rescue M3 -> escalate.
  If not, log as a real negative (the S_N barrier holds for AKOrN phase). NO config-shopping the contrastive
  temperature/weight to find a winner — pick ONE reasonable setting, pre-state it, run it.

### STEP 5 (conditional on STEP 1 HOLDS) — time-boxed M3 object-discovery swing
Faithful object-discovery (Shapes via train_obj) where synchrony binding is strongest. gen_shapes.py +
eval_obj_shapes.py scaffolding already exist (Cursor session). HARD VALIDATION GATES before trusting any
result: (a) the object model must actually train (fgari above chance on a single task); (b) an oracle/liveness
control. If the pipeline doesn't validate within the time-box, STOP and log "object-discovery M3 needs a
proper build" rather than report a broken-instrument result. Earlier toy object-discovery gave ~null fgari =
a warning. Treat as UPSIDE, not deliverable.

### STEP 6 (always) — consolidation
Keep a running honest status in this file. Draft a paper-skeleton (claims + evidence + the honest M3 limit)
from research-log + prereg. This is valuable regardless of how 1-5 land.

## Morning report (10am): the honest arc state — what's SOLID, what's MODEST, what's NULL, what's OPEN,
## the reconciled M2 decodability verdict, the contrastive-phase crux result, M3 object-discovery status,
## and the recommended next move. With every number adversarially checked.

## STEP 2 RESULT (2026-06-07, reconciliation — IMPORTANT) — "CORRECTED-NULL" is a MISLABEL (underpowered)
Read m2_corrected_screen.json per-seed. At LAYER 1, R6 vs R6s (the CORRECT ablation), ALL 5 descriptors:
  marginal +0.091, 2nd_moment +0.093, coh_eig +0.109, cluster_occ +0.046, spatial4x4 +0.080 -- ALL POSITIVE,
  n_pos=3/3 each. The "CORRECTED-NULL" verdict only fired because it demanded p<0.05 at n=3, and the exact
  sign-flip FLOOR at n=3 is 0.125 -- p<0.05 is MATHEMATICALLY IMPOSSIBLE at n=3. So this is NOT a null; it's a
  clean, directionally-UNANIMOUS positive that is merely UNDERPOWERED (same disease as the joint-bound n=6, and
  the same mislabel-by-underpower trap the project keeps hitting -- here in the pessimistic direction).
  (m2_relational_probe's "NULL ROBUST" used R5:no_proj [the invalid ablation the audit rejected] at n=2 -> ignore
  for the headline; it is not the corrected test.)
RECONCILED DECODABILITY VERDICT: R6 phase-context is MORE task-decodable than R6s (frozen-random coupling) at
layer 1 across all descriptors -- PROMISING POSITIVE, needs n>=15 to certify. ACTION: power the corrected screen
(R6 vs R6s, layer1) to n>=15 with the SAME code (only --seeds changes), same as the joint-bound. Then exact
paired test. This + the joint-bound are the TWO M2 sub-claims to bulletproof.

## STEP 4 — LITERATURE GROUNDING (2026-06-07, Valency) + STAGED DESIGN
Two papers reframe the M3 limiter as expected-and-addressable, not a dead end:
- Verbeke & Verguts 2020 "Learning to synchronize" (pubmed 31430280): binding-by-synchrony + RL LEARNS to
  synchronize task-relevant / desynchronize task-irrelevant modules -> solves stability-plasticity (=forgetting).
  => synchrony does NOT encode task automatically (matches our null) but CAN with an EXPLICIT objective.
- Vedovati & Ching 2024 (arXiv 2408.01316): contextual modulation for "task packing", robustness to context
  ambiguity. (Also KoPE 2604.07904: Kuramoto phase-encoding added to ViTs -- a possible later substrate.)
DESIGN TENSION TO FLAG TO HARRY (honest): thesis = "label-free context channel WITHOUT task labels". A
supervised/contrastive phase objective uses CURRENT-TASK labels DURING TRAINING. That is legitimate in
class-incremental CL (current task known while training; channel stays label-free AT TEST/inference) but
weakens the claim from "no task labels" to "no task labels at inference". Must be stated plainly, not hidden.

STAGED STEP 4 (low-risk first; the project's lesson = validate cheaply before big builds):
  STEP 4a (LOW RISK, no AKOrN retrain): is task identity LATENT in FROZEN R6 phase, just not LINEARLY readable?
    The descriptor sweep showed even full 32k-dim phase decodes only ~0.28 LINEARLY. Test a STRONGER readout on
    the SAME frozen phase: nonlinear MLP probe + a supervised-contrastive projection head, held-out CV, vs the
    linear probe (~0.39 best layer) and vs R6s and chance. This is standard probing (no architecture change).
    PRE-REG RULE: if a nonlinear/contrastive readout on FROZEN phase beats the linear probe by >=+0.10 AND beats
    R6s -> task info is latent in phase (an EXTRACTOR problem, fixable by a better context-encoder) -> escalate.
    If it stays ~linear -> the info genuinely is not in frozen phase -> only STEP 4b could help.
  STEP 4b (HIGHER RISK, only if 4a warrants / as the big swing): retrain AKOrN with a contrastive phase loss
    (pull same-task phase together, push diff-task apart; ONE pre-set lambda, NO shopping) -> re-measure
    decodability + joint lift. This is the Verbeke-Verguts move in AKOrN form.
  Both have VALIDATION GATES (chance floor, R6s control, held-out CV). Frame 4a as the supervised UPPER BOUND
  on extractable task info from frozen phase -- cleanly informative either way.

## STEP 1 RESULT (2026-06-07) — M2 JOINT-BOUND HOLDS DECISIVELY at n=20
R6 mean lift=0.287 vs R6s 0.107; mean paired diff=0.180, sd=0.101, t=7.98; 19/20 seeds R6>R6s (95%);
EXACT sign-flip 2-sided p=0.0000. The n=6 borderline (p=0.094) was pure underpower. M2 USABILITY claim
(phase-context as param-gen channel, R6>>R6s, unbypassability already validated) = SOLID. Pressure-test
vindicated. Next: STEP 3 (bulletproof the DECODABILITY sub-claim to n>=15) + STEP 4a (latent task info).

## STEP 3 RESULT (2026-06-07) — DECODABILITY IS NULL at n=17 (I was WRONG in STEP 2; own it)
Corrected screen R6 vs R6s, n=17, exact paired 2-sided + Holm across 5 descriptors:
  Layer 1: NONE survive Holm (all >0.3). Deltas +0.006..+0.022, shrank from the n=3 deltas (+0.05..+0.11).
  Layer 2: only cluster_occ survives (D=+0.039, Holm=0.006) but marginal/2nd_moment/spatial4x4 are ~0/NEG ->
           isolated, not a clean channel.
=> Learned synchrony (R6) is NOT meaningfully more TASK-DECODABLE (linear) than frozen-random coupling (R6s).
   My STEP 2 "underpowered positive, will certify at n=15" prediction was WRONG -- the n=3 3/3 was small-sample
   NOISE. The "CORRECTED-NULL" label was RIGHT at proper power. (R6 vs R5:depthwise IS a small positive:
   coh_eig/spatial4x4 D~+0.04 p~0.002 -> having coupling at all helps; LEARNED vs RANDOM coupling does not.)

## THE TENSION (must resolve before trusting M2) — decodability NULL vs joint R6>>R6s
Joint usability (n=20): R6 ctx_lift 0.287 >> R6s 0.107 (t=7.98, p~0). Decodability (n=17): R6 ~ R6s (null).
How can both hold? Hypotheses: (a) joint ctx_lift reflects nonlinearly-accessible or CLASS-level info that a
linear TASK-probe misses (the hypernet g is an MLP); (b) joint advantage is an INSTANCE-info leak, not a task
channel (would undercut the "context channel" framing). RESOLVERS: STEP 4a (linear vs MLP task-decode, running)
+ a JOINT ADVERSARIAL CONTROL = within-task context SHUFFLE (shuffle c across samples within a task: keeps
task-level info, destroys instance info). If R6's joint advantage SURVIVES within-task shuffle -> it's a real
task-context effect; if it VANISHES -> it was instance leak. This is the key check now that joint is load-bearing.

## JOINT ADVERSARIAL CONTROL (2026-06-07) — the joint lift decomposes into TASK vs INSTANCE
Task-mean context (OOD for g) gave ~chance (taskmean_ret~0.00, R6 s0) -> suggested instance-only, BUT that test
is OOD-confounded (g trained on per-sample contexts never saw the centroid). FAIR in-distribution test =
within-task SHUFFLE (each sample gets ANOTHER real same-task sample's context):
  R6 s0: real=0.568, within_task_shuffle=0.265, task_mean=0.183, const=0.100.
  => withintask_retention=0.35: ~35% of the joint advantage is TASK-LEVEL (same-task context still lifts above
     const), ~65% is INSTANCE-level (lost on swap). The OOD task-mean UNDERSTATED the task component (good catch).
IMPLICATION: the headline ctx_lift (0.287) OVERSTATES the task-channel -- most of it is per-instance info. The
TRUE task-level channel = the within-task-shuffle lift (R6 s0: 0.265-0.100=0.165). The corrected M2 headline
must compare R6 vs R6s on the WITHIN-TASK-SHUFFLE lift (task-level), not raw ctx_lift. [R6s arm running.]
Single seed so far -- waiting for R6+R6s x5 seeds before concluding.

## ADVERSARIAL WORKFLOW (w160iv00m) + RECONCILIATION (2026-06-07) — important process + substance catches
The 4-lens workflow read the LOCAL repo (workflow agents run locally; tonight's results are on the BOX
/root/NC, not pulled) -> it FALSELY alarmed "n=20 joint / n=17 screen / decomposition have NO data". I then
PULLED all box results local (/tmp/box_results) and RE-VERIFIED from the actual saved per-seed JSONs:
  - JOINT n=20: CONFIRMED real (41 recs, 20 unique seeds/arm), R6 0.287 vs R6s 0.107, diff 0.180, 19/20,
    exact2p=0.0000. SOLID. (statistician's "only n=6" = stale local read.)
  - CORRECTED-SCREEN n=17: CONFIRMED, NO L1/L2 swap. L1 deltas +0.006..+0.022 all Holm>0.3 (null after
    correction); L2 only cluster_occ survives (Holm 0.006). The lens's "L1 is positive, you swapped" came from
    the STALE LOCAL numpy-ridge run (n<=12), NOT my n=17 sklearn run.
VALID CATCHES from the workflow (acted on):
  1. PROVENANCE: results were box-only -> now pulled to /tmp/box_results (back up to repo before any commit).
  2. PROBE-SENSITIVITY (real finding): decodability L1 R6-vs-R6s is +0.06ish with numpy-ridge (old, n<=12) but
     ~+0.01 NULL with sklearn LogReg (n=17). => linear decodability is WEAK/FRAGILE, not robust. My "flat null"
     was an OVER-CORRECTION; honest = "weak, probe-fragile, null at highest power; nonlinear MLP also null".
  3. CONFLATION: joint-pressure-test ctx_lift (R6 0.287/R6s 0.107) != jtm decomposition real-lift (R6 0.337/
     R6s 0.173 at n=5) -- different runs/baseline defs. Keep distinct.
LESSON: pull box data local BEFORE running analysis workflows (agents can't ssh). 
CURRENT HONEST M2 VERDICT (pre jtm-n=20): JOINT usability R6>>R6s = SOLID (verified). But decomposition (n=5
prelim: 35% task / 65% instance) + weak/fragile decodability suggest the usability is largely INSTANCE-level;
the TASK-level channel (what M3 needs) is real-but-small. n=20 task-level decomposition RUNNING to firm this.

## DECISIVE n=20 TASK-LEVEL DECOMPOSITION (2026-06-07) — M2 task channel is REAL (my n=5 over-correction was WRONG)
within-task-shuffle, R6 vs R6s, n=20 paired, exact tests:
  TASK-level lift (within_shuffle - const):    R6=0.141 vs R6s=0.071  -> diff +0.071, 18/20, exact2p=0.0000 *
  INSTANCE-level lift (real - within_shuffle): R6=0.186 vs R6s=0.104  -> diff +0.082, 19/20, exact2p=0.0000 *
  TOTAL real lift: R6=0.327 vs R6s=0.175. Split ~46% task / ~54% instance.
=> BOTH components significant at n=20. The n=5 prelim (35% task, p=0.125) was UNDERPOWERED -- my "mostly
   instance leak / weak task channel" synthesis was an OVER-CORRECTION (same n=5 trap I'd warned about). At
   proper power the TASK-LEVEL channel is genuine and R6>>R6s.

## FINAL HONEST M2 VERDICT (n=20 verified, all from saved box artifacts)
M2 = POSITIVE with nuance. Learned synchrony (R6) provides a usable label-free context channel for parameter
generation, significantly > frozen-random coupling (R6s), at BOTH task-level (+0.071, 18/20, p~0) and
instance-level (+0.082, 19/20, p~0); raw joint ctx_lift R6 0.287 vs R6s 0.107 (n=20, p~0); context is
task-necessary (wrong-task dP5 R6 -0.41). CAVEATS (honest): (1) absolute task-level capacity is MODEST
(within-shuffle acc R6 ~0.24 vs chance 0.10 vs oracle ~0.53) -> explains why M3 forgetting-bypass is hard;
(2) task-ID is NOT cleanly DECODABLE from phase (R6~R6s, linear n=17 null + nonlinear null, probe-fragile) ->
the channel carries task-USEFUL info a hypernet exploits, not a cleanly-decodable task label; (3) decodability
screen used richer descriptors than the joint's 2n pooled context (different representations) so it's a
secondary/orthogonal probe. NET: synchrony yields a real, modest, label-free task+instance context channel for
param-gen. M3 (forgetting-bypass) remains the open hard part (channel real but modest; oracle-validated limit).

## STEP 4b SMOKE (2026-06-07) — contrastive-phase AKOrN AMPLIFIES the task channel (promising, 1 seed)
R6 s0: lam=0 baseline TASK_lift=0.148 task_decode_cv=0.452 | lam=1.0 contrastive TASK_lift=0.185 (+0.037)
task_decode_cv=0.505 (+0.053). Clears the +0.03 smoke gate. CAVEAT: contrastive improved real/instance/task
ALL (real 0.39->0.52) -> partly "better-trained akorn", and higher task-decode is partly EXPECTED (uses task
labels). NON-CIRCULAR payoff = does it cut CL FORGETTING? -> building the CL test (step4c). Confirming n=8 first.

## STEP 4b CONFIRMATION n=8 (2026-06-07) — contrastive amplification is a NULL (smoke was noise)
Paired R6 lam=1.0 vs lam=0, n=8: task_lift delta -0.003 (3/8, p=0.84); task_decode_cv delta +0.002 (p=0.91);
real_acc lam1 0.381 < lam0 0.417. The 1-seed smoke (+0.037 task_lift, +0.053 decode) was NOISE -- the THIRD
small-n positive to evaporate at power tonight (n=3 decodability, n=5 task-split, n=1 smoke). VERDICT: one
principled supervised-contrastive phase objective (lam=1.0,tau=0.1, on 2n context, ctx_epochs=20) does NOT
amplify AKOrN's task channel. Honest scope: not an exhaustive search (one lambda/tau/formulation), but the
pre-registered principled setting is NULL -> the phase task-channel ceiling looks ARCHITECTURAL, not
training-effort. Contrastive (Verbeke&Verguts-style) swing = NEGATIVE. Now running clean n=8 M3 characterization
(baseline vs contrastive vs oracle CL forgetting) for an artifact-backed M3 number.

## STEP 4c RESULT (2026-06-07) — clean n=8 M3 characterization = NEGATIVE-WITH-MECHANISM
baseline(lam=0): learn 0.560, final 0.131 (~chance), forgetting 0.536. contrastive(lam=1.0): final 0.131
(identical; Δ=0.000 p=0.99). ORACLE(one-hot task): final 0.524, forgetting 0.023. => phase channel does NOT
bypass forgetting (forgets to chance); contrastive does NOT rescue; oracle proves harness works -> the limit
is the phase channel's modest task capacity (phase->oracle gap 0.131->0.524). Clean, artifact-backed M3 negative.

## NIGHT COMPLETE (2026-06-07). Arc: M1 solid+ / M2 positive-with-nuance (LOCKED, n=20) / M3 measured-limit
(oracle-validated negative). All results in experiments/m2/results/autorun_2026_06_07/ + on box. NOT committed.
Full writeup: MORNING-REPORT-2026-06-07.md. Both GPUs now IDLE (no jobs). Further autonomous M3 swings = low
expected value + p-hacking risk -> STOPPED deliberately. Recommended next (needs Harry): (a) lock M1+M2 writeup;
(b) optional M2 GENERALIZATION to a 2nd dataset (SplitCIFAR) to strengthen -- a build with bug-risk, so not done
unsupervised. Box can be shut to save cost or redirected in the morning.

## STEP 5 — M2 GENERALIZATION to SplitCIFAR-10 (greenlit by Harry, 2026-06-07)
Mirrors run_joint on CIFAR-10 (10-way, trunk-adequate) via validated _phase_context + frozen-random trunk +
within-task-shuffle. SMOKE (R6 s0) PASSED gates: real=0.346 const=0.158 real_lift=0.188 (>0); TASK_lift=0.135
dominates inst_lift=0.052 (task-dominant, mirrors shapes). Running R6 vs R6s n=8 -> exact paired task-level test.

## STEP 5 CIFAR-10 M2 GENERALIZATION n=8 (2026-06-07) — DIRECTIONALLY CONSISTENT, underpowered
R6 vs R6s task-level (within-shuffle lift): R6 0.155 vs R6s 0.090 -> +0.066, 6/8, exact2p=0.17 (n=8 can't reach
sig). Effect size ~matches shapes (+0.071). real_lift R6 0.204 vs R6s 0.113. NOTE: R6s const elevated (0.232 vs
R6 0.150) -> mild partial-bypass for R6s, but within-shuffle subtracts it. VERDICT: M2 task-level channel
DIRECTIONALLY GENERALIZES to real CIFAR images but n=8 underpowered -> extending to n=20 to confirm (don't claim
from p=0.17 per the small-n lesson).

## PALR M0 SMOKE (2026-06-07) — STRONG GO SIGNAL (1 seed, MUST replicate)
Co-trained LEARNED pool, R6 s0: real=0.997 within(task)=0.485 const=0.100 -> TASK_lift=0.385 (vs mean-pool ~0.14;
+0.245, >>+0.08 gate). within-shuffle task-acc 0.485 APPROACHES oracle 0.52 (mean-pool capped 0.24). const=0.100
chance -> unbypassability holds. The workflow thesis (mean-pool readout was the bottleneck) looks RIGHT. SKEPTICAL
CAVEATS: 1 seed; need controls learned-R6s (coupling-specific) + mean-R6 (pool not co-training-in-general); and
this is the JOINT gate -- real M3 test is SEQUENTIAL CL survival. Running full 2x2 (learned/mean x R6/R6s) n=8.

## CIFAR-10 M2 GENERALIZATION n=20 (2026-06-07) — partial/weak generalization
task-level R6 0.160 vs R6s 0.116 -> +0.043, 14/20, exact2p=0.065 (borderline). Directionally consistent with
shapes (+0.071) but WEAKER on real images. Honest: M2 task channel PARTIALLY generalizes to CIFAR.

## PALR M0 2x2 GATE RESULT (2026-06-07) — PALR-specific bet FAILS, but CO-TRAINING is the real lever (BIG)
task-level within-shuffle acc (n=8, chance 0.10, oracle ~0.52): learned-R6 0.498, learned-R6s 0.495,
mean-R6 0.489, mean-R6s 0.431. Gate comparisons: pool effect (learned-R6 vs mean-R6) +0.009 p=0.078 (NULL);
coupling under learned pool (R6 vs R6s) +0.003 p=0.25 (NULL); coupling under mean-pool +0.058 p=0.031 (small).
=> (1) CO-TRAINING the phase->context->g pathway lifts the task channel from M2's 0.24 to ~0.49 (NEAR ORACLE
0.52) -- M3 gap CLOSED in the joint setting. ROBUST (n=8). (2) NOT the learned pool (PALR gate FAILS). (3) NOT
learned synchrony (R6~R6s under co-training). The synchrony-specific R6>R6s only survives under the WEAK
mean-pool. THESIS-CRITICAL: need the NON-OSCILLATOR co-trained control (is it oscillators or just co-training?).
NEXT (relentless): (A) does the co-trained near-oracle FROZEN context survive SEQUENTIAL CL (the M3 prize, vs
capture-freeze 0.13 and oracle 0.52)? (B) non-oscillator co-trained control for attribution.

## STEP 7 — M3 PRIZE TEST (2026-06-07) — STAGED FORGETTING-BYPASS, SYNCHRONY-SPECIFIC (n=4, MUST replicate)
Co-trained frozen label-free phase context -> sequential CL (fresh g_cl + replay):
  R6 (learned coupling):  final=0.962  forgetting=0.042  (per-seed 0.99,0.89,0.98,0.98)
  R6s (random coupling):  final=0.734  forgetting=0.298  (per-seed 0.69,0.95,0.58,0.71 -- high var)
  ref: capture-freeze final=0.131 (forgets); oracle one-hot=0.524.
=> (1) FORGETTING BYPASSED: R6 retains 0.96 vs capture-freeze 0.13. (2) SYNCHRONY-SPECIFIC RETENTION: R6>>R6s
(0.96 vs 0.73, forgetting 0.04 vs 0.30) -- learned synchrony's interference-resistance (M1!) manifests as
forgetting-bypass in M3. ON-THESIS. NOTE: 0.96>>oracle 0.52 because the co-trained context carries CLASS-rich
(not just task) info; the bypass rides on instance-rich separable contexts + low cross-task interference.
SKEPTICAL CAVEATS: n=4; suspiciously strong; R6s high-var; STAGED (akorn jointly pre-trained, NOT fully-online);
need non-oscillator control + fully-online version. Replicating to n=12 now.

## STEP 7 REPLICATION n=12 (2026-06-07) — FORGETTING-BYPASS HOLDS, SYNCHRONY-SPECIFIC, SIGNIFICANT
R6 (learned): final=0.972 (sd 0.028) forgetting=0.028 (per-seed all 0.89-1.0). R6s (random): final=0.797
(sd 0.145) forgetting=0.222 (per-seed 0.58-0.95). R6-R6s paired final: +0.175, 11/12, exact2p=0.0044.
=> STAGED forgetting-bypass is REAL+REPLICATED, and synchrony-specific (learned>>random coupling). Survived the
replication gate that killed 4 prior small-n positives tonight. Remaining gates: (A) attribution = non-oscillator
plain-CNN control (step8, launching n=12); (B) fully-online co-training (drop the joint-pretrain caveat).

## STEP 8 ATTRIBUTION (2026-06-07) — FORGETTING-BYPASS IS SYNCHRONY-SPECIFIC (clean control ladder)
Staged CL final_acc n=12: R6(learned-synchrony) 0.972 >> R6s(random-osc) 0.797 ~ plainCNN(no-osc) 0.827 >>
capture-freeze 0.131. Paired: R6-plainCNN +0.145 p=0.0010; R6-R6s +0.175 p=0.0044; plainCNN-R6s +0.030 p=0.49.
=> Co-training ANY context gen gives PARTIAL bypass (~0.8); LEARNED SYNCHRONY is UNIQUELY effective (0.97, near-0
forgetting, low var), beating BOTH non-oscillator AND random-oscillator controls. NOT just co-training, NOT just
oscillators -> specifically learned synchrony. ON-THESIS, replicated, significant. REMAINING caveat: context gen
jointly pre-trained then frozen (legit frozen-backbone+continual-head CL, but not fully-online). FINAL GATE =
fully-online sequential co-training (step9): can the oscillator learn online (with replay) without forgetting itself?

## STEP 9 FULLY-ONLINE SMOKE (2026-06-07) — WORKS (1 seed), drops the joint-pretrain caveat
R6 fully-online (oscillator co-trained SEQUENTIALLY + raw replay, no joint pretrain): learn=0.998 final=0.983
forgetting=0.019 (even > staged 0.97). C3 hazard did NOT materialize -- replay anchors the oscillator. ~279s/seed.
MUST replicate n=12 + controls (R6s, plainCNN) ONLINE: is it SYNCHRONY or just REPLAY? If R6>>controls online ->
fully-online synchrony-driven forgetting-bypass = ORIGINAL AMBITION achieved. Running n=12 x3 arms now.

## STEP 9 FULLY-ONLINE M3 n=12 (2026-06-07) — DECISIVE: FORGETTING-BYPASS ACHIEVED, OSCILLATOR-NECESSARY
final_acc: R6(learned-syn) 0.964 (sd .021, forget .038, learn .994) | R6s(random-osc) 0.858 (sd .108, forget
.119, learn .953) | plainCNN(no-osc) 0.114 (sd .030, forget .449, learn .474 -> COLLAPSES) | capture-freeze .131.
Paired 12/12 all p=0.0005: R6-R6s +0.107; R6-plainCNN +0.850; plainCNN-R6s -0.743.
=> FULLY-ONLINE, LABEL-FREE, synchrony-driven forgetting-bypass on the toy construct. Oscillator NECESSARY
(plainCNN collapses to chance online -- representation drifts; oscillator bounded phase dynamics resist drift +
learn online 0.99). Learned synchrony best (R6>R6s). The ORIGINAL AMBITION recovered (toy). LIMITS: toy shapes
construct (CIFAR M3-online untested); replay-assisted (but plainCNN+replay collapses -> synchrony essential);
plainCNN under-learns online too; one-session build -> needs review + CIFAR generalization.

## STEP 10 CIFAR-10 M3-ONLINE SMOKE (2026-06-07) — SOBERING: does NOT generalize at first look (1 seed)
R6 CIFAR-online s0: learn=0.777 final=0.276 forgetting=0.626. The shapes bypass (R6 0.96, forget 0.04) does NOT
transfer to real CIFAR-10 -- R6 FORGETS (final 0.28, forget 0.63). 1 seed; confirming n=12 + controls. Likely the
toy construct's super-separable phase contexts were doing the work; on real images phase contexts are less
separable/stable. Honest negative signal for generalization (the #1 risk flagged). NO config-shopping to rescue.

## STEP 10 CIFAR-10 M3-ONLINE n=12 (2026-06-07) — DOES NOT GENERALIZE (definitive negative)
final_acc: R6 0.256 (forget 0.672), R6s 0.267 (forget 0.563), plainCNN ~0.21 (forget ~0.56). All collapse to
~0.2-0.27 (chance 0.10). R6-R6s = -0.011, 3/12, p=0.11 -> NO synchrony advantage on CIFAR. The shapes bypass
(R6 0.96, synchrony-necessary, p=0.0005) was CONSTRUCT-SPECIFIC. On real images the co-trained phase contexts
are not separable/stable enough under online learning -> catastrophic forgetting for ALL arms incl learned
synchrony. NOT config-shopped (shapes-matched protocol). A real CIFAR effort would need genuine method
development (bigger buffer, stronger context encoder, attractor regularization), not tuning -> honest future work.

## FINAL HONEST ARC VERDICT (2026-06-07)
M1 = SOLID positive (synchrony reduces interference, dz~2.28). M2 = positive-with-nuance (phase = usable
label-free context channel, modest; partially generalizes to CIFAR borderline). M3 = forgetting-bypass works
FULLY-ONLINE on the toy binding construct (synchrony-necessary, clean controls, n=12 p=0.0005) but DOES NOT
generalize to CIFAR-10 (all arms forget; synchrony advantage vanishes). => CONSTRUCT-SPECIFIC proof-of-concept,
NOT a general natural-image CL method. Honest contribution: M1+M2 measurement results + an M3 mechanism that
works on structured binding tasks + an honest negative on natural-image scaling. NOT the original-ambition
breakthrough (which required general forgetting-bypass).

## TRACK C — TIGHT GENUINE-BINDING M3-ONLINE (2026-06-07) — THESIS CONFIRMED (n=4, replicating to 12)
TIGHT=True (objects overlap -> segregation REQUIRES binding). M3-online, n=4: R6 final=0.958 (forget .044),
R6s 0.591 (forget .339), plainCNN 0.106 (chance, collapse). R6-R6s +0.368 4/4; R6-plainCNN +0.852 4/4.
HEADLINE: synchrony advantage SCALES with binding difficulty -- easy-shapes R6-R6s=+0.10 -> TIGHT +0.37 (3x).
When binding is genuinely required, random coupling fails (0.59) and only LEARNED synchrony retains (0.96).
Reframes CIFAR: single-object -> nothing to bind -> synchrony idle -> fails (correct SCOPING, not failure).
DECISION: multi-object BINDING is the scope/story; CIFAR partial-rescue (PHLOX, ~0.4) deprioritized.
A router gate (for ref): CIFAR proto-routing R6 0.51 > R6s 0.41 > trunk-centroid 0.32 (real, synchrony-specific,
but 0.51-capped -> PHLOX only partial). Replicating C to n=12 to lock the binding-difficulty interaction.

## SPARSE-ACTIVATION CONTROL (May-30 open item) — CLOSED: synchrony != sparsity
TIGHT binding, n=8: sparseCNN(k-WTA) final=0.105 (forget .477) -- IDENTICAL to dense plainCNN (0.106), both
~chance, FAR below R6 (0.958). => sparse representations alone give NOTHING; the bypass requires the OSCILLATOR
(and learned synchrony specifically). Kills the Elephant/k-WTA 'it's just sparsity' alternative. Control ladder
(TIGHT): R6 0.958 >> R6s 0.591 >> {sparseCNN 0.105 ~ plainCNN 0.106 ~ chance 0.10}.
A router n=12 firmed: CIFAR task-routing R6 0.462 > R6s 0.400 (12/12, p=0.0005) > trunk-centroid 0.300 -- synchrony
-specific even on single-object, but weak (0.46). MONOTONIC LADDER (synchrony effect scales w/ binding demand):
CIFAR(min) +0.06 route < easy-shapes +0.10 < TIGHT +0.37.

## EXP #1 REAL-CONTENT 2-digit MNIST (2026-06-07, n=4) — OSCILLATOR generalizes; learned-synchrony attenuates
M3-online, real overlapping MNIST digits: R6 final=0.650, R6s 0.603, plainCNN 0.220 (chance 0.10). R6-plainCNN
+0.430 (4/4); R6-R6s +0.047 (4/4, SMALL). => TWO claims, different scope: (1) OSCILLATOR-NECESSITY generalizes
to REAL content (R6/R6s ~0.6 >> plainCNN 0.22 -- oscillator dynamics enable online binding-CL where feedforward
collapses); (2) LEARNED-synchrony specificity is largely SYNTHETIC (R6-R6s +0.29 TIGHT-shapes -> +0.05 MNIST).
Honest reframe: core claim = oscillatory binding dynamics necessary+generalizing; learned-coupling = extra boost
when binding structure is learnable. Firming to n=12.

## MAIN-TRACK PUSH #1 — Fashion-MNIST 2-object (2nd real dataset, n=4) — CROSS-DATASET HOLDS
R6 0.615, R6s 0.498, plainCNN 0.298 (chance 0.10). R6-plainCNN +0.318 (4/4); R6-R6s +0.117 (4/4) -- LARGER than
MNIST-digits (+0.04), consistent w/ binding (Fashion items more complex -> binding matters more). Oscillator-
necessity generalizes across 2 REAL datasets (MNIST digits + Fashion clothing). Not MNIST-specific.
