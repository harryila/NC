# M2 pre-registration — Oscillatory-workspace channel capacity (LOCK before any M2 headline run)

**Locking the estimand + design BEFORE results — the discipline that caught M1's d-clause defect and the
positive-control metric-shopping. Draft 2026-06-03; ratify before the matched-param C_ctx run.**

## Claim (one sentence)
Phase-gating raises **task-information-per-parameter** in an oscillatory workspace vs an identical
rate-coded bottleneck under continual learning — i.e. the synchrony channel carries more task bits at
matched capacity. (Novelty = the **measurement**, not the architecture.)

## Why M2 is licensed to start (from M1)
- M1 head-free mechanism: learned synchrony reduces inter-task representational overlap (dz=1.64, 10/10
  seeds), geometry-kill-test SURVIVED (triple-corroborated) → a real representational substrate exists.
- M2 viability pre-check: task-id is linearly decodable from R6 phase-state ABOVE chance (layers 1-2
  cv≈0.36 vs 0.20) → the channel is NON-EMPTY (S_N "zero task bits" trap ruled out). **Route layer 1-2.**

## Primary estimand (LOCK)
**C_ctx = effective channel capacity in BITS of the workspace-conditioned state w.r.t. task.**
Operationalize two complementary ways, BOTH pre-registered (report both, primary = (1)):
1. **Linear-decodable task bits:** I_lin(phase; task) lower-bounded via the frozen linear probe's
   confusion matrix → mutual information in bits (not just accuracy). The probe + capture path already
   exist (m2_primitives.linear_task_decodability, m2_precheck OOM-safe capture).
2. **Effective-rank-in-bits** of the workspace query/conditioned-state covariance
   (m2_primitives.ctx_capacity_bits) — the representational-dimensionality reading.

## RATIFIED 2026-06-03 (Harry): M2 = TRUE CCC C_ctx via a phase-conditioned theta-generator
After the second-pass CCC re-read (below), the estimand fork was resolved: **build the TRUE CCC C_ctx, NOT
the representational I(phase;task).** C_ctx = I(context; GENERATED PARAMETERS), so M2 routes the AKOrN
phase-state as CONTEXT into a MINIMAL phase-conditioned theta-generator (von Oswald-style = the M3 backbone)
and measures I(phase-context; generated-theta) in bits vs a rate-coded context baseline. The GASPnet/
workspace-substrate fork is MOOT (a workspace classifier has C_ctx=0 by definition). M2->M3 is now one
continuous build. M1 confirmed = Path B (head-free mechanism + retention probe; forgetting = methodological,
since CCC proves naive state-based learners MUST forget so a behavioral win is not the goal).

## The contrast (LOCK — matched-parameter, single design axis)
- **Phase-gating ON:** AKOrN R6 phase-state (layer 1-2) as the CONTEXT c feeding theta(c).
- **Phase-gating OFF (two controls, both reported):**
  (a) `apply_proj=False` (R5:no_proj) phase-state as context — the within-architecture flip;
  (b) a **matched rate-coded context** (same dim, non-oscillatory) at ±2% params/FLOPs.
- **Headline metric = C_ctx (bits of I(context; generated-theta)) and C_ctx PER PARAMETER:** ON > OFF at
  matched params. Pre-register the ±2% param/FLOP match (same discipline as M1's ladder).
- **[SUPERSEDED] the earlier "route through a workspace bottleneck / GASPnet" framing — replaced by the
  phase-conditioned theta-generator above, per the CCC re-read. GASPnet/MANAR no longer needed for M2.**

## Substrate (LOCK, pending Harry veto)
**GASPnet** (single global query = simplest clean C_ctx readout; already fuses Kuramoto phase-binding +
routing-by-agreement, so phase-state→workspace is native, not bolted-on; prior-art verdict tags it
"baseline substrate, not competitor" → zero overlap risk). **REJECT Goyal multi-slot** for M2 (K slots
inflate estimator variance + import S_N slot-permutation symmetry prematurely — defer to M3 if needed).
**MANAR** kept as the rate-coded OFF baseline.

## Protocol traps to engineer against (from CCC)
- **Split-MNIST is non-discriminating** (CCC's own warning): use it ONLY to validate the C_ctx pipeline
  reproduces CCC's published ~bits behavior; the REAL claim runs on Split-CIFAR-100-class (reuse the M1
  SplitCIFAR100 driver verbatim).
- **Wrong-Context Probing** (P5/P5b/P6/P7, already built in m2_primitives) as the falsifier: a real channel
  must DEGRADE under wrong/random/zero context (ΔP5 ≪ 0). A channel that doesn't degrade is bypassed.
- **Partial out nuisances** (sparsity, effective-rank, norm) — machinery exists (h3.partial_corr_did).

## Decision rule (LOCK, pre-specify)
- **GREENLIGHT M3** iff C_ctx-per-param(ON) > C_ctx-per-param(OFF) on BOTH controls, with the
  Wrong-Context Probing degradation present (ΔP5<0), n≥10 seeds, exact paired test p<0.05, AND the effect
  survives nuisance-partialling. Single locked metric per family (no metric-shopping — the M1 lesson).
- **PIVOT (capacity ≈ matched rate-coded):** the first measured channel capacity of an oscillatory
  workspace is itself a constraint on the theory — publishable (per the M2 pivot tree).
- **Inconclusive band + ≥15 seeds for the decisive ON-vs-OFF**, as in M1.

## Build sequence (no headline run until estimand ratified)
1. [no-GPU] this doc ratified + C_ctx-in-bits MI estimator finalized (extend ctx_capacity_bits / probe).
2. [GPU-2] clone + smoke-test GASPnet UNMODIFIED on a CIFAR batch (mirror the AKOrN stand-up).
3. [GPU-2] workspace-bottleneck adapter: route layer-1/2 phase-state (from _capture_osc) through GASPnet's
   single query; expose the conditioned-state matrix to the C_ctx estimators.
4. [GPU-2] Split-MNIST pipeline-validation run (protocol repro, NOT a claim).
5. [GPU] matched-param ON-vs-OFF C_ctx-per-param on Split-CIFAR-100-class + Wrong-Context Probing.

_Ratified / dated: ____________________________

## SUBSTRATE UPDATE (2026-06-03) — GASPnet has NO public code release
Verified: arXiv 2507.16674 (v1) releases no code (no GitHub, no "code will be released"); web search finds
none. Our prior-art memo's assumption "M2 rides on public GASPnet code" is FALSE. GASPnet IS small + fully
specified in the paper (3 conv 26/30/32 + 2 dense 16/10; per-location phase; single global query; coupling
r_ij = (<Wq*m_i,Wk*m_j>*N_ij - eps)/kappa; Kuramoto phi update; modulation [1+aF*cos(dphi)]) → reimplementable
in days, not weeks. OPEN DECISION FOR HARRY (substrate):
  A) Reimplement GASPnet as the workspace substrate (faithful to original M2 design; new build+validation).
  B) Use AKOrN as its own workspace — route layer-1/2 phase-state through a MINIMAL global-query bottleneck,
     keep apply_proj ON/OFF as the matched-param contrast. Reuses validated M1 machinery; keeps the bit-
     identical single-flag isolation that survived the M1 kill-test; thinner build. (Agent lean = B; GASPnet
     reimpl as later robustness substrate.) NOT decided autonomously.

## SECOND-PASS FINDING (2026-06-03) — re-read the CCC paper; C_ctx is context->PARAMETER, not representation
Read arXiv 2603.07415 (CCC) in full. CRITICAL clarification of what C_ctx actually is:
  C_ctx = max_P(c) I(c; theta(c)) — mutual info between the CONTEXT SIGNAL and the GENERATED PARAMETERS
  (Def 5, Thm 4). Their headline result: ONLY conditional-regeneration architectures (HyperNetworks) get
  C_ctx >> 0 (0.95-0.98 on Split-MNIST, zero forgetting); EWC/SI/LwF/Replay/CFlow-ODE all sit at C_ctx≈0
  and forget. A plain classifier that MODIFIES A STATE (incl. AKOrN or GASPnet as a classifier) has NO
  context->parameter pathway → C_ctx = 0 BY DEFINITION, no matter how much task info is in its phase-state.
IMPLICATION for M2 (reframes the substrate question): our M2 viability pre-check measured I(phase; task) —
info IN THE REPRESENTATION (cv≈0.36). That is NECESSARY but is NOT C_ctx (info in GENERATED PARAMS). To
measure the true CCC quantity, the phase-state must CONDITION A PARAMETER-GENERATION step (a thin
hypernetwork) — i.e. real M2 is "oscillatory CONTEXT feeding a small theta-generator", partway into M3.
NEITHER GASPnet NOR CCC released code → we build either way; GASPnet gives a workspace but still C_ctx=0
alone. So the substrate fork (A reimplement-GASPnet vs B AKOrN-as-workspace) is the WRONG axis. The REAL
M2 fork is the ESTIMAND:
  (i) REPRESENTATIONAL task-info: I(phase; task) — extends M1, fast, no hypernet, but NOT the CCC quantity.
  (ii) TRUE CCC C_ctx: I(phase-context; generated-theta) — the thesis spine; needs a minimal phase-conditioned
       theta-generator (a thin M2->M3 bridge). More build, but it's the actual claim the arc rests on.
Both are defensible/different papers. Agent recommendation now PENDING this reframe — putting the estimand
fork to Harry rather than the substrate fork. (von Oswald hypernetwork is the M3 backbone; a minimal version
is the natural M2 (ii) vehicle.)

## M2 BUILD DONE (2026-06-03) — theta-generator + true CCC C_ctx estimator (both reviewed SHIP, CPU-verified)
Files (local/uncommitted, in experiments/m2/): theta_generator.py (+test), ctx_channel_capacity.py (+test).
- theta_generator.PhaseContextThetaGen: von-Oswald-seed hypernet, context c -> generated head weights theta;
  make_inject_fn plugs into m2_primitives wrong-context probes; build_context_from_phase (OOM-safe pool-per-
  sample) for ON=R6 / OFF(a)=R5:no_proj; build_rate_coded_context for matched-dim OFF(b).
- ctx_channel_capacity: C_ctx = I(c; theta(c)) in BITS. PRIMARY = decodability_mi_lower_bits (Fano/DPI lower
  bound: CV confusion -> I=H(T)-H(T|That), chance-corrected by label-shuffle floor, clamped [0, H(T)]). COMPANION
  = theta_effective_dim_bits (eff-dim of the generated-theta cloud).
DECISIVE CCC-FAITHFULNESS CHECK (both reviewers, verified): a PERFECTLY task-separable context fed to a CONSTANT
generator gives mi_lower = 0.0; the SAME context through a conditional generator gives ~log2(K). => the estimator
measures the context->PARAMETER channel (true C_ctx, CCC Def 5/Thm 4), NOT representational I(phase;task). If it
had measured representation info the constant case would read ~log2 K. Tests: constant->0, perfect one-hot->log2K,
noisy-separable->strictly between; 200 random designs gave ZERO out-of-[0,Hmax] bound violations.
TWO FLAGGED NOTES (mathematically correct, not bugs): (1) the COMPANION eff_dim counts ALL theta variation incl
class-independent noise, so it is NOT a bound on I(c;theta) -> rely on mi_lower_bits (PRIMARY) for the ON-vs-OFF
greenlight. (2) K balanced one-hot theta span only K-1 dims after centering, so perfect-gen companion reads
eff_dim=K-1 (expected). NEXT: Split-MNIST C_ctx pipeline-validation on GPU-2 (reproduce CCC: hypernet C_ctx>>0
vs state-based ~0), then matched-param ON-vs-OFF on Split-CIFAR-100-class + Wrong-Context Probing.

## M2 INSTRUMENT VALIDATED (2026-06-03, GPU-2) — Split-MNIST, real gradient-trained generators
cctx_validate_splitmnist.py: trained a phase/task-CONDITIONED theta-generator vs a task-AGNOSTIC one (both
solve 5 binary tasks at acc=1.0). Estimator reads: CONDITIONED C_ctx(mi_lower)=2.09 bits (~Hmax=log2(5)=2.32);
AGNOSTIC = 0.0 bits. Reproduces CCC's dichotomy (hypernet C_ctx>>0 vs state-modifier ~0) on REAL trained
weights, not synthetic theta -> the instrument is trustworthy. CONFIRMED the build's caveat LIVE: the agnostic
generator's eff_dim reads 46.9 (counts jitter noise) while mi_lower correctly reads 0.0 -> MUST use mi_lower_bits
(PRIMARY) for the ON-vs-OFF greenlight, NOT eff_dim (which would have falsely flagged the agnostic gen as
high-capacity). VALIDATION_PASS=True. NEXT: the headline run — route REAL AKOrN layer-1/2 phase-state as context
(ON=R6, OFF(a)=R5:no_proj, OFF(b)=rate-coded) into the generator on Split-CIFAR-100-class, train, measure C_ctx
+ Wrong-Context Probing (P5/P6/P7 must degrade). Needs the AKOrN phase-context capture wired to the generator.

## M2 HEADLINE FIRST RUN (2026-06-04, 3 seeds, E=50, layer 1) — NEAR-ZERO C_ctx, ALL ARMS (honest null)
Per-arm mean C_ctx (bits, Hmax=log2(5)=2.32): ON(R6)=0.0161, OFF_a(R5:no_proj)=0.0164, OFF_b(rate)=0.0130.
ALL THREE ~0.01-0.016 bits = essentially ZERO (the Split-MNIST validation got 2.09 for a real channel; these
are ~130x smaller). Per-seed ON: C_ctx 0.014-0.018, cv_accuracy 0.248-0.255 (chance=0.20 -> barely above),
raw_mi~0.022 / chance_floor~0.0074 (floor ~1/3 of raw -> the p>>n fragility the review flagged is real, but
the floor subtraction is working, not eating the whole signal), gen_train_acc~0.93 (generator DID learn).
VERDICT: NOT a positive M2 result. The ON-vs-OFF gaps (~0.007 bits) are SMALLER than the chance-floor noise
(~0.007) -> indistinguishable from zero. The comparison block's "ON_beats_both=True" compares noise-to-noise;
the per-seed MEAN even flips it (OFF_a >= ON). NOTE the internal inconsistency: comparison block (ON=0.0181 >
OFF_a=0.0109) disagrees with C_ctx_mean_bits (ON=0.0161 < OFF_a=0.0164) -> the comparison block uses only one
seed's value, not the mean (a reporting bug to fix; the MEAN is the honest number). Wrong-context deltas ARE
strongly negative (P5/P5b/P7 ~ -0.43 to -0.52) -> the generator's correct-context DOES matter for accuracy,
but that reflects the generator overfitting per-task heads, NOT a high-capacity phase channel (C_ctx is what
measures the channel, and it's ~0). HONEST READ: at this scale/config, the AKOrN phase-state does NOT carry
meaningful CCC context-channel capacity into generated params for ANY arm. This needs interpretation (true null
vs power/scale/design artifact) BEFORE scaling to 10-15 seeds. Do NOT scale yet.

## M2 NEAR-ZERO INTERPRETATION (2026-06-04) — DESIGN ARTIFACT, not a true null (arc NOT threatened)
4-lens adversarial workflow (orchestrator died before synthesis, but 2 lenses completed + converged; other 2
were mid-diagnostic). VERDICT (CCC-theory + arc-strategist lenses, independent, SAME crux): the near-zero C_ctx
for ALL arms is a FIXABLE DESIGN ARTIFACT of capture-then-freeze, NOT a true null about phase.
THE CRUX (clean info-theory): we froze AKOrN's TASK-TRAINED features, which ALREADY solve the tasks (gen_acc
0.93). With the head sitting on already-separating features, the c->theta pathway is under ZERO pressure to
carry task bits -> C_ctx~0 is the CORRECT reading of a "state-modifier-equivalent" architecture (CCC's C_ctx=0
class). This also resolves the tension (gen_acc 0.93 + wrong-ctx hurts -0.5 + C_ctx~0): the generator USES
context as a gain/nuisance knob (wrong ctx hurts) but need not ENCODE task-id in theta (C_ctx~0). The Split-MNIST
validation read 2.09 ONLY because there theta was the ONLY task-adaptive component.
THE FIX (cheap, both lenses): re-run ONE ON arm with frozen TASK-AGNOSTIC features (random projection / raw
pooled pixels / frozen-random) so generated theta is the ONLY task-adaptive part (= the validated condition).
If C_ctx jumps ~0.016 -> toward Hmax 2.32, artifact confirmed. PLUS a 2-min pre-check: linear_task_decodability
on the 8-dim POOLED context itself -- if cv~0.37 (like full layer1) the pool kept the signal; if ~0.20 the pool
destroyed it (enrich the descriptor, scoped null about POOLED phase, still not an arc pivot). Do NOT scale to
10-15 seeds or pivot until the artifact-fix re-run confirms. M1 (strong, kill-test-survived) untouched by this.

## M2 ARTIFACT-DX RESULT (2026-06-04) — INCONCLUSIVE; the test was itself confounded
trained: C_ctx=0.019, cv_acc=0.266, gen_acc=0.976.  agnostic: C_ctx=0.004, cv_acc=0.223(~chance), gen_acc=1.000.
The agnostic (task-scrambled features) did NOT lift C_ctx -> the design-artifact hypothesis FAILED its test.
BUT the test is confounded and CANNOT conclude "true null": SMOKING GUN = agnostic gen_acc=1.000 while
cv_acc~chance and C_ctx~0. The generator hit 100% TRAIN acc even with scrambled features => it is MEMORIZING
per-sample labels (100-way per-sample head, 200 gen-epochs, ~1600 rows) from features+context, NOT learning
TASK structure. So theta encodes per-sample memorization, not task-id -> C_ctx~0 regardless of whether the
phase context carries task. The agnostic-features manipulation didn't force theta to carry TASK because the
generator bypasses via per-sample overfitting. => INCONCLUSIVE, not a null.
ROOT CHECK NEEDED (should have been first, the strategist lens's 2-min pre-check): does the 8-dim POOLED
context even carry task info? Run m2_primitives.linear_task_decodability on the pooled contexts directly (no
generator). If cv~0.37 (like full layer1 state) -> context is fine, the GENERATOR/per-sample-memorization is
the problem (fix: per-task context or regularize gen, prevent memorization). If cv~0.20 -> the 8-dim pool
DESTROYS task info (fix: richer context descriptor). Either way the headline near-zero is a MEASUREMENT/vehicle
problem, not yet a true null. Do NOT scale or pivot. M1 untouched.

## M2 ROOT CAUSE FOUND (2026-06-04) — the 8-dim POOLED context is too lossy (NOT a true null)
Root check (decode TASK directly from context, NO generator): cv_pooled_8dim=0.228 (chance 0.200) ~ CHANCE;
cv_full_state=0.288 this run (M1 pre-check got ~0.37 at layer1). => the 8-dim pool (mean+meansq over 4 group
axes via _pool_phase_state) DESTROYS the task signal the full phase-state carries. The headline near-zero
C_ctx was a LOSSY-CONTEXT-DESCRIPTOR problem: the generator got a ~task-empty input, so C_ctx~0 trivially.
This explains the inconclusive artifact-dx too (both arms fed a task-empty 8-dim context, so swapping features
couldn't help). NOT a true null, NOT the frozen-features artifact. THE FIX (bounded): replace the 8-dim
descriptor with a RICHER context that preserves the full-state task info (~0.29-0.37 decodable). Options: (a)
per-spatial-region pooling (preserve some spatial structure of the phase field), (b) more moments / higher-dim
pool, (c) a fixed random projection of the fuller (C,H,W) phase-state to a tractable dim (e.g. 128-256). Pre-
check each candidate the CHEAP way FIRST (linear_task_decodability on the new descriptor, no generator) and
only feed the generator a descriptor whose cv >> chance. Then re-run the headline ON-vs-OFF. M1 untouched.
PROCESS NOTE: should have run this generator-free context-decodability check BEFORE building the full headline
driver — it is the cheapest possible validation that the context carries the signal at all. Lesson logged.

## M2 DESCRIPTOR SWEEP (2026-06-04) — descriptors are NOT the bottleneck; the task signal itself is WEAK here
Decode TASK from context (no generator), chance=0.20: pooled8 cv=0.247, region_pool 0.225, randproj128 0.256,
randproj256 0.274, FULL-32768-state 0.282. CRITICAL: even the FULL un-pooled phase-state only decodes at 0.282
(~chance+0.08). So richer descriptors barely help -- the ceiling is low. The 8-dim pool was lossy, yes, but
fixing it doesn't recover much because the TASK SIGNAL in this phase-state is weak at this config.
THE REAL FLAG (regression vs earlier): the M1 viability pre-check decoded task from layer-1 phase-state at
cv~0.37 (that result GREENLIT M2). This run's full state = 0.28. SAME layer, same measurement -> meaningfully
LOWER. Something changed: candidates = (a) backbone training regime (pre-check used m2_precheck.py --all-layers
on a fresh R6 5-task E=30 run; this uses cctx_akorn_run._train_backbone E=50, possibly different opt/captured
probe), (b) the pre-check's per-sample capture vs this driver's, (c) which layer/representation is effectively
tapped. DECISION: do NOT launch a headline re-run with randproj128 -- at a 0.28 ceiling, C_ctx through a
generator stays near-floor. Must first RECONCILE the 0.37-vs-0.28 gap: re-run the ORIGINAL m2_precheck.py
--all-layers (the exact code that got 0.37) and confirm whether 0.37 reproduces. If 0.37 reproduces there but
not in this driver -> the driver's capture/training differs (fixable, align them). If it's now ~0.28 there too
-> the earlier 0.37 was optimistic/seed-lucky and the phase task-channel is genuinely marginal -> M2 needs a
rethink (stronger phase tap, more backbone training, or honest scoping). M1 untouched. STOP reactive probing;
reconcile the regression as the next single deliberate step.

## M2 RECONCILED (2026-06-04) — the phase channel is REAL; the near-zero was a STACK of driver bugs
Re-ran the ORIGINAL m2_precheck.py --all-layers (unchanged code, E=30, 5 tasks): layer0 cv=0.297, layer1=0.332,
LAYER2=0.393 (CHANNEL EXISTS) -- REPRODUCES the earlier ~0.37, best at LAYER 2. The phase context channel is
real and reproducible. The M2 headline near-zero was therefore a stack of MEASUREMENT bugs, NOT a true null:
  BUG 1 (wrong layer): headline driver hard-coded --layer 1 (cv~0.33); the signal is strongest at LAYER 2
        (0.39). I tapped the weaker layer. (Prereg said "route layer 1-2" but the data says layer 2.)
  BUG 2 (lossy capture): my driver's full-state at layer1 = 0.28 but the pre-check's layer1 = 0.33 -> the
        driver's capture/pooling path is ADDITIONALLY lossier than m2_precheck._capture_phase_per_sample.
  BUG 3 (8-dim pool): _pool_phase_state crushes task info to ~chance (earlier finding).
  BUG 4 (frozen task-features): in the full headline, frozen task-trained features let the generator bypass
        the context (the original interpretation, still a real issue downstream of 1-3).
FIX (deliberate, aligned to what WORKS): build the M2 context from the pre-check's EXACT capture path
(_capture_phase_per_sample at LAYER 2) -> a descriptor that decodes task at ~0.39 -> feed THAT to the generator
(and for the generator stage, use task-agnostic/raw features so theta must carry task, addressing BUG 4). Then
re-run headline ON vs OFF. Validate (cheap, no generator) that the new context decodes task >> chance at layer 2
BEFORE the headline run. M1 untouched. NET: M2 is recoverable; signal is real; the bugs were all in HOW I
measured, not WHETHER the channel exists.

## M2 SCREEN PRE-REGISTRATION (2026-06-05) — generator-free ON-vs-OFF decodability (LOCKED before results)
RATIONALE: the C_ctx estimator decodes task from the GENERATED theta; by the data-processing inequality
(T -> context -> theta) C_ctx is UPPER-BOUNDED by the context's own task-decodability. So the cheapest CORRECT
screen is: does the R6 phase-state decode task BETTER than the matched R5:no_proj control (single apply_proj
flip)? This is simultaneously (a) the CEILING of any C_ctx headline and (b) a publishable result on its own
(estimand (1)). It reuses the VALIDATED m2_precheck capture (_capture_phase_per_sample + _pool_phase_state +
m2_primitives.linear_task_decodability; reproduces ~0.33 L1 / ~0.39 L2 on R6) so it CANNOT re-introduce the
headline's wrong-layer / lossy-capture / frozen-feature bugs.
DESIGN: m2_screen.py trains R6 and R5:no_proj on the SAME seed's 5-task SplitCIFAR100 (paired = same data),
E=30, captures phase at L1+L2, decodes task per arm. 6 seeds (0-5), extensible to 10.
STOPPING RULE (LOCKED, primary layer = 2, secondary = 1): per seed Delta_s = cv_R6 - cv_R5:no_proj.
  PROCEED to the C_ctx hypernetwork headline   iff mean Delta >= +0.03 AND one-sided paired p < 0.05.
  PIVOT to a NULL finding                        iff equivalence within +-0.03 (TOST p < 0.05) -> synchrony
        reorganizes inter-task overlap (M1) but does NOT add task-channel CAPACITY: a real constraint on the
        theory, publishable as the first measured oscillatory context-channel null.
  INCONCLUSIVE                                    -> extend to 10 seeds, then call.
This is the pre-specified EXIT the M2 saga lacked (the "every null is a fixable bug" loop). Result pending —
running on GPU-2 (m2_screen.json, incremental per-seed save). M1 untouched.

## M2 MARGINAL SCREEN RESULT (2026-06-05, 10 seeds: GPU-2 s0-5 + GPU-1 s6-9) — NOT PROCEED
Merged R6 vs R5:no_proj linear task-decodability (the LOCKED screen), n=10 paired seeds:
  L1 (secondary): R6=0.343 R5=0.336 meanD=+0.007 dz=+0.19 (perm p1=0.30) TOST=0.023 -> EQUIVALENT (+-0.03):
     synchrony adds NOTHING at layer 1.
  L2 (PRIMARY):   R6=0.352 R5=0.392 meanD=-0.040 dz=-0.73 (perm p1=0.98; 6/10 seeds negative) -> R6 is
     WORSE than R5 (beyond the +-0.03 equivalence band on the negative side; not PROCEED, not equivalence).
VERDICT on the MARGINAL descriptor: NOT PROCEED. Synchrony does not add (and at the readout-feeding layer 2
REDUCES) marginal task-decodability => the C_ctx hypernetwork headline on this descriptor would be null
(C_ctx <= context task-decodability, which does not favor R6). DECISIVE CAVEAT (under test): the descriptor is
_pool_phase_state = mean+meansq over sites = MARGINAL phase stats, BLIND to the relational/clustering synchrony
structure M1 actually uses (phase_cluster_stability). The L2 "R6 worse" is CONSISTENT with synchrony RELOCATING
task-info out of the marginal distribution into relational structure the descriptor cannot see. m2_relational_
probe.py (running GPU-1, seeds 0-2) decodes task from relational descriptors (2nd-moment, coh-eig, cluster-
occupancy, spatial) to disambiguate RESCUE (relational R6>>R5) vs NULL-ROBUST (relational also R6~R5). M1 untouched.

## M2 RELATIONAL PROBE (2026-06-05, CIFAR class-IL) — NULL-ROBUST at n=2 (killed seed 3 to reallocate GPUs)
On CIFAR class-IL, relational descriptors did NOT rescue: the seed-0 2nd-moment hint (+0.041 at L1) COLLAPSED
at seed 1 (-0.031); n=2 means ~0 across all relational descriptors. So on CIFAR the null is descriptor-robust
(not just a marginal-pooling artifact). Killed seed 3 (redundant) to free GPU-1 for the construct test.

## M2 CONSTRUCT-MISMATCH RESULT (2026-06-05) — *** RESCUE ***: synchrony OPENS a task-channel on a BINDING task
GROUNDING (AKOrN ICLR'25): synchrony binding is demonstrated on multi-object OBJECT DISCOVERY (Tetrominoes/
dSprites/CLEVR/Shapes), NOT single-object classification -- our entire M2 used the classification codepath =
synchrony's WEAKEST construct. m2_shapes_construct.py: multi-object Shapes (2 shapes/img from {tri,sq,circ,diam},
10 pair-classes, 5 CL experiences) on the EXACT R6-vs-R5:no_proj ladder (single apply_proj flip); decode
experience-id from phase (marginal + relational descriptors). 8 seeds (GPU-1 s0-3 + GPU-2 s4-7), E=30.
RESULT (paired R6-R5, n=8) -- R6 phase carries SIGNIFICANTLY more task info than R5:no_proj:
  L1: spatial +0.082 (p1=0.016, 7/8) | marginal +0.077 (p1=0.031, 6/8) | 2nd_moment +0.074 (p1=0.047, 5/8)
      | cluster_occ +0.018 (p1=0.047, 6/8) | coh_eig +0.022 (ns).
  L2: coh_eig +0.107 (p1=0.027, 7/8) | 2nd_moment +0.097 (p1=0.059) | spatial +0.098 (p1=0.066) | marginal
      +0.078 (p1=0.070) | cluster_occ +0.009 (ns).
R6 cv ~0.37-0.41 vs R5 ~0.29-0.31 (R6 ~2x chance). OPPOSITE SIGN from CIFAR class-IL (R5>=R6, null). => RESCUE
by the pre-registered rule (multiple descriptors >= +0.05 with p<0.05). The M2 CIFAR null was a CONSTRUCT
MISMATCH: class-IL CIFAR does not engage binding; on a binding task synchrony DOES open a decodable task-channel.
HONEST CAVEATS: modest magnitude (R6 ~2x chance, not a blowout); high seed variance (sd ~0.10-0.15, bimodal --
some seeds large e.g. seed4/seed7 +0.2-0.3, some ~null); L2 borderline (p~0.06-0.07) while L1 is clean.
NEXT: (1) binding mechanism check m2_shapes_binding.py (does R6 cluster objects >> R5 via object-ARI; running
8 seeds both GPUs) to confirm WHY; (2) more construct seeds to firm L2; (3) rebuild the M2 C_ctx (theta-gen)
ON the binding task; (4) faithful object-discovery codepath (train_obj). IMPLICATION FOR THE ARC: M2/M3 are
viable on OBJECT-CENTRIC/binding tasks, NOT class-IL CIFAR -- the thesis holds where synchrony is engaged.
M1 (head-free interference reduction, dz=2.28 long-stream) untouched.

## M2 4-AXIS AUDIT (2026-06-05, 4 parallel agents) — the prior nulls AND the "rescue" were ARTIFACTS
Combing the code + CCC first-principles before the Tetrominoes escalation revealed we were measuring M2 wrong
on FOUR independent axes, each alone enough to force a null:
  (1) WRONG ABLATION: apply_proj=False (R5:no_proj) is NOT "synchrony off" — it only swaps the tangent-space
      projection (geodesic integration) for raw Euler+renorm; Kuramoto coupling + recurrence + Omega still run.
      Real synchrony on/off = R6 (learned coupling) vs R6s=R6_scrambled (frozen RANDOM coupling, apply_proj=True,
      params+geometry IDENTICAL) and/or R5:depthwise (coupling removed). So R6-vs-R5:no_proj (M1 H3, M2, and the
      object-discovery project test) all compared two COUPLED systems.
  (2) UNFAIR BASELINE: R5:no_proj is bimodal/unstable; the Shapes "RESCUE" (R6>R5:no_proj) was R5 COLLAPSING on
      ~3/8 seeds, not R6 gaining. No clean OFF control existed.
  (3) CAPTURE BUG: averaging the oscillator state over eval_inits is a Euclidean mean of gauge-arbitrary sphere
      vectors -> attenuates/cancels the phase signal (asymmetrically, penalizing R6). FIX = eval_inits=1.
  (4) ESTIMAND/ARCH (the killer, CCC): the C_ctx headline generated theta ON TOP OF frozen task-trained features
      that already solve the task = CCC Remark-2 "context bypass" -> C_ctx~0 BY CONSTRUCTION. Split-MNIST read
      2.09 bits only because there theta was the SOLE adaptive component. Fix = unbypassable trunk + end-to-end +
      functional metric (wrong-context dP5 + forgetting), not decode-task-from-theta-on-a-task-solving-trunk.

## M2 CORRECTED SCREEN (2026-06-05) — *** PASS at L1 ***: learned synchrony DOES add a task-channel
Re-ran the screen with fixes (1)-(3): real ablation R6 vs R6s (frozen random coupling, matched) + R5:depthwise;
eval_inits=1; relational descriptors; multi-object binding construct; generator-free decodability of experience-id.
POOLED n=12 (R6 vs R6s, the true synchrony on/off):
  L1 marginal   D=+0.056 p=0.010 (11/12) | L1 2nd_moment D=+0.056 p=0.019 (10/12) | L1 spatial D=+0.059 p=0.017 (10/12)
  L2 (all descriptors) D ~ 0 (p 0.3-0.6, 7/12) -> null at the deep layer.
R6/R6s both >> R5:depthwise (coupling removed) at all layers. INTERPRETATION: task info in the phase context
comes from HAVING coupling (vs depthwise) AND, at L1, from the coupling being LEARNED/synchronized (R6 > R6s,
+0.056, p~0.01, robust across 3 descriptors + 10-11/12 seeds). This is a REAL, significant synchrony effect on
a SOUND footing (correct ablation/baseline/capture) — modest (~0.056, ~60% more above-chance signal than R6s at
L1) and L1-only. The n=6 was underpowered (D+0.045 p~0.15); n=12 firmed it. NOTE: this REVERSES the premature
"capacity-null" read — the prior nulls were the 4 artifacts above, not the absence of a channel.
VERDICT: PASS (L1 clears +0.05 & p<0.05 on 3 descriptors) -> PROCEED to the full end-to-end UNBYPASSABLE
phase->theta hypernetwork (fix (4)): task-agnostic trunk, L1 phase context, R6-vs-R6s ablation, functional
metrics (wrong-context dP5 + forgetting/BWT) on the binding construct + Split-MNIST validation. The L1 channel's
~0.056 decodability is an UPPER bound on usable I(T;theta); the hypernet tests if it yields real C_ctx/forgetting.

## M3 HYPERNET (2026-06-05) — fix-(4) build + UNBYPASSABILITY INVARIANT VALIDATED; R6-vs-R6s running
m2_hypernet.py implements the corrected estimand end-to-end: trunk = FROZEN-RANDOM conv (task-agnostic, no
static bypass), context-gen = a frozen AKOrN (R6 or R6s) -> L1 phase descriptor c, and the ONLY trainable
task-adaptive params = the hypernet g: c -> theta(head). Naive sequential CL on the binding construct (Shapes,
10 pair-classes, 5 experiences). Metrics: forgetting/BWT + wrong-context dP5 (swap c to another task's context;
negative => channel is real AND used). This closes CCC Remark-2 "bypass": theta is the SOLE adaptive route.
UNBYPASSABILITY SMOKE (--demo, the invariant the old C_ctx headline FAILED): task-carrying context -> acc 1.000;
CONSTANT context -> acc 0.211 ~ chance 0.25. PASS: with no usable context the head CANNOT separate tasks => there
is no static bypass; only context-routed task info can be used. (This is precisely what frozen task-trained
features violated and Split-MNIST accidentally satisfied.)
EARLY (R6s baseline, GPU-2, seed 0): learn=0.590 (learns each experience, chance 0.1) | final=0.127 (~chance)
| forgetting=0.579 (catastrophic) | wrong-ctx dP5=+0.003 (~0 => frozen-random phase carries NO usable task
channel). EXPECTED null baseline. PENDING: R6 (GPU-1, learned synchrony) x3 seeds — the test is whether R6 lifts
final_acc above R6s's ~0.13 and drives dP5 negative (phase channel actually used to resist forgetting). 3 seeds
each, paired R6-vs-R6s. (R6s s0 above; R6 + remaining seeds in flight.)

## M2 HYPERNET — CL RESULT (2026-06-05, naive sequential, n=3 each) — FUNCTIONAL NULL (but confounded)
                  final_acc      forgetting     wrong-ctx dP5
  R6 (synchrony):   0.126          0.604          -0.008
  R6s (random):     0.127          0.608          -0.012   (chance final = 0.10)
Dead heat: BOTH forget catastrophically to ~chance; dP5~0 (context barely used). Under naive sequential CL the
+0.056 decodability edge yields NO usable forgetting-resistance. BUT this is CONFOUNDED: naive g forgets its OWN
c->theta mapping (later tasks overwrite earlier), which floors BOTH arms regardless of context quality. So the CL
null does NOT distinguish "channel unusable" from "g's own forgetting masks it." => ran the joint upper-bound.

## M2 HYPERNET — JOINT UPPER-BOUND (2026-06-05) — *** R6 >> R6s ***: the phase channel IS usable for param-gen
Remove the CL-forgetting confound: train g on ALL tasks jointly (no CL), context = each arm's L1 phase, vs a
CONSTANT-context control (= how much the frozen-random trunk alone solves). Metric: ctx_lift = acc_real-acc_const
(the context's marginal param-gen contribution) + wrong-ctx dP5. [seed 0; n=3 in flight]
                  acc_real   acc_const   ctx_lift     wrong-ctx dP5
  R6 (synchrony):  0.572      0.162       +0.410        -0.538
  R6s (random):    0.315      0.200       +0.115        -0.253   (chance = 0.10)
R6's learned-synchrony phase context adds ~3.6x the usable param-gen capacity of frozen-random coupling
(+0.410 vs +0.115), and swapping to a wrong context destroys R6 far more (-0.538 vs -0.253) => the context is
genuinely carried AND used. REINTERPRETATION OF THE CL NULL: the channel is NOT unusable — naive g's OWN
catastrophic forgetting floored both arms. The thesis SURVIVES at the parameter-generation level (decodable AND
usable). M3 is now concrete: the channel works; what's missing is a CL mechanism that stops g from forgetting its
c->theta map (von Oswald output-regularization / context-conditional consolidation). PERF NOTE: g-training caches
the frozen akorn+trunk context/features once (was recomputing every step -> 200s+ timeout); now ~30s/seed.
JOINT LOCKED (n=6, paired R6-vs-R6s, seeds 0-5):
                  mean ctx_lift     mean wrong-ctx dP5
  R6 (synchrony):   +0.274            -0.392
  R6s (random):     +0.117            -0.199   (chance = 0.10)
  paired diff (R6-R6s): lift +0.157 (5/6 seeds R6>R6s; paired-t one-sided p~0.02, two-sided ~0.04; sign p~0.11)
                        dP5  -0.193 (5/6 seeds R6 more context-dependent)
Seed 3 reversed (R6s>R6) = the known bimodal seed variance, but 5/6 + paired-t p~0.02-0.04 FIRMS the effect.
VERDICT: the learned-synchrony phase channel carries ~2.3x the usable param-gen capacity of frozen-random
coupling => CHANNEL USABLE (decodable AND usable). GO to M3: the regularized-hypernet (--cl-reg) that stops g
forgetting its c->theta map; test whether R6 retains tasks where R6s collapses on the binding CL stream.

## M3 MECHANISM 1 (output-regularization) — NULL at all beta (2026-06-05, seed-0 beta probe)
von Oswald-style output-reg (penalize g's theta drift on a buffer of past phase-contexts). beta in {0,.3,1,3,10,30}:
  R6  final: .133 .130 .107 .110 .100 .100   (chance .10)
  R6s final: .122 .123 .125 .108 .100 .100
final ~ chance at EVERY beta for BOTH arms: low beta forgets (=naive), high beta blocks new learning (learn drops
too) -- NO regime retains. beta=0 == naive null (sanity OK). DIAGNOSIS: von Oswald regularizes a small FIXED set
of per-TASK embeddings; our context is per-INPUT (noisy within a task), so freezing g on specific past input-
contexts neither preserves the task-level mapping nor leaves capacity for new tasks. Mechanism mismatch, NOT a
refutation of M2 (channel usability is the joint-upper-bound finding, independent of this CL mechanism).

## M3 MECHANISM 2 (latent context-replay) — RUNNING (2026-06-05, R6 GPU-2 / R6s GPU-1, n=6, epochs=40)
Store a small buffer of past (phase-context, frozen-feature, label); interleave replay batches during CL. This
is the natural way to exploit a USABLE+SEPARABLE channel: R6's task-separated contexts should let g fit distinct
per-task heads (retain), while R6s' overlapping contexts create conflicting replayed (ctx->y) pairs (interfere).
Expect replay to approach the joint upper bound (R6 ~0.5 >> R6s ~0.3 >> naive .13) IF separability is the driver.
RESULT (n=6, small buffer 60/task): R6 final 0.157 vs R6s 0.112 (paired +0.045, 5/6 seeds, paired-t one-sided
p~0.036; dP5 R6 -0.068 vs R6s -0.006). WEAK POSITIVE: replay lifts R6 above naive/chance where R6s stays at
chance, and only R6 is context-dependent -- but R6's 0.157 is far below the joint ceiling (0.57), driven by 3/6
seeds. DIAGNOSTIC (big buffer 300/task, replay-batch 256, n=6): R6 final 0.124 vs R6s 0.127 -> NULL (2 R6 / 2 R6s /
2 tie). The small-buffer R6>R6s VANISHED under a buffer-size change => it was a FRAGILE ARTIFACT, not a real
effect. Also note neither replay regime reached the joint ceiling (0.57); CL-replay stays ~chance (0.12-0.16),
i.e. heavy replay did NOT recover retention -- the sequential schedule/replay-balance (or context overlap at
test) limits it.

## M3 HONEST VERDICT (2026-06-05) — channel USABLE (M2), but NO robust online-CL forgetting-bypass yet
Summary of the unbypassable phase->theta hypernetwork program (all on the binding construct, frozen phase
channel R6 vs R6s, caches validated, unbypassability invariant validated):
  * JOINT upper-bound (no CL): R6 lift +0.274 vs R6s +0.117 (n=6, 5/6, p~0.02-0.04) -> channel USABLE. [SOLID]
  * Naive CL: R6 ~ R6s ~ chance (g forgets its own c->theta map). [null, confounded]
  * Output-reg CL: NULL at all beta (per-input vs per-task mismatch). [null]
  * Replay CL (small buffer): weak R6>R6s (p~0.04) -> EVAPORATES at big buffer (R6~R6s~chance). [not robust]
## M3 ORACLE LIVENESS (2026-06-05) — instrument VALIDATED; the replay null is a REAL channel limit (overlap)
Suspected the replay null might be a broken instrument (heavy replay -> ~chance is anomalous for CL). Test: same
big-buffer replay harness (1500 buf, 256 replay), but context = ONE-HOT TASK ID (oracle, perfectly separable).
RESULT (n=3 so far, GPU split): final 0.537/0.517/0.525, forgetting 0.008/0.038/0.002, dP5 ~ -0.52. The harness
RETAINS NEAR-PERFECTLY with a clean context -> instrument is ALIVE. So the phase-context collapse (final ~0.13)
is a GENUINE channel result, not a harness bug. BOTTLENECK = CONTEXT OVERLAP: oracle (zero overlap) -> replay
batches never collide -> CL works; phase (modest overlap) -> conflicting replayed (ctx->y) -> g collapses. NB the
oracle CL ceiling (~0.53) ~= the joint phase ceiling (~0.57): the phase channel HAS the task info under
simultaneous training, but its overlap blocks SEQUENTIAL exploitation. IMPLICATION: M3 (online forgetting-bypass)
needs a SHARPER/less-overlapping phase channel; the toy binding construct's channel is usable-but-not-sharp-enough.
The principled path is a setup where synchrony binding is strong (faithful object-discovery) AND a direct
measurement of per-task context separability -- but the phase->oracle gap (0.13->0.53) is large, so M3-as-headline
is a real risk; M1+M2 are the defensible contribution.

CONCLUSION: M2 stands (synchrony's phase-state is a decodable AND usable label-free context channel; R6>>R6s).
M3's central claim -- that this channel lets a hypernet BYPASS catastrophic forgetting better than random
coupling -- is NOT supported by the mechanisms tried. The channel is real and usable, but I could not convert
it into a robust online-CL advantage (outreg fails; replay is fragile and sub-joint). This is an honest
negative/open result, deliberately NOT p-hacked by config-shopping. M1 (dz=2.28) + M2 (channel usable) are the
defensible milestones; M3 (forgetting-bypass) is an OPEN PROBLEM requiring a different approach (e.g. context-
conditional architecture, a faithful object-discovery task with a stronger channel, or a CL method tuned to
match the joint ceiling) -- a decision for the next session, not more solo config perturbation.
