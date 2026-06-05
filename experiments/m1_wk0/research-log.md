
## 2026-06-02 — A5-competitiveness result (10/10 A5:derpp seeds): R6 NOT competitive on forgetting
DER++ forgets 50.1 pts; R6 (naive) forgets 78.6 — R6 forgets +28.5 pts MORE, all 10 seeds (prereg margin was
"within 3 pts" → FAILS as written). HONEST read: this is apples-to-oranges BY OUR OWN DESIGN — DER++ has a
replay buffer; R6 is naive (replay ablated OUT per the M1 guardrail). "Replay beats no-replay" is not news,
and it's on the SATURATED/CONFOUNDED forgetting endpoint we already deprioritized (the one where R6≈R5). The
A5-competitiveness conjunct was written assuming forgetting = primary endpoint; we pivoted to head-free H3.
So either (a) the conjunct is MIS-SPECIFIED for the head-free claim (drop/replace it), or (b) it's a real gap
that Path-A de-saturation + a matched-replay fight (R6+replay vs DER++) must close. NOT my call to wave away
post-hoc — flagged for Harry's endpoint fork. The mechanism claim (kill-test SURVIVE, head-free H3 10/10) is
untouched by this; this only concerns whether the *behavioral/forgetting* leg of M1 can stand.

## 2026-06-02 — Positive-control difficulty sweep (GPU-2, RTX 4090) + operating-point decision
Sweep over 4 difficulties × 2 epochs × 2 seeds. KEY FINDING: the H3 probe has CLEAR detection power — R6
has lower inter-task overlap than R5:no_proj (synchrony-favoring direction) in EVERY cell where both arms
learn the binding task: diff0 ΔO=+0.165/+0.112, diff1 ΔO=+0.038/+0.065, diff3-ep35 ΔO=+0.026. Signal only
vanishes/reverses at transitional difficulties (diff2, diff3-ep60) where learning is partial (~80-94%).
METHODOLOGICAL CORRECTION (logged, not silent): the sweep's "saturation guard" (reject train-acc >92%) was
designed for M1's FORGETTING endpoint and is WRONG for the OVERLAP endpoint — two arms can both hit 100%
train-acc yet have very different cross-task CKA (diff0: 0.556 vs 0.721). Training saturation does NOT
confound representational overlap. So diff0/diff1's high train-acc does NOT disqualify them. DECISION: run
the full control at **diff1 (jitter 0.18), 50 epochs, 10 seeds** — the principled middle: binding provably
required (single-feature acc=1/3 at all difficulties), both arms learn (~99%), signal large + epoch-STABLE
(+0.038→+0.065). Avoided diff0 (pixel-memorization risk at 100%) and the lone auto-"usable" diff3-ep35
(fragile: +0.026 flips to −0.015 at ep60, n=2). Full control launched on GPU-2 → writes positive_control.json
(auto-read by analyze_hardened gate). GPU-1 phase2 untouched in parallel.

## 2026-06-03 — Positive control PASSES (raw contrast) + honest-DiD scale limitation found (GPU-2)
Full control, diff1, 10 seeds, both arms learn ~99%. TWO metrics on the SAME data:
- RAW O_inter contrast (R5-R6) — the metric the SWEEP and the DECISIVE M1 CIFAR result both used:
  mean_delta=+0.074, exact p=0.021, dz=0.85, **8/10 seeds R6<R5** → **PASSES** (R6 lower inter-task overlap
  on a synchrony-favoring task = the prereg-required DETECTION-POWER proof). R6 O_inter=0.510 vs R5=0.584.
- HONEST-DiD (inner=O_inter-O_intra, the augmentation-baseline variant built this session): mean_delta=+0.027,
  p=0.26, dz=0.21, 6/10. FAILS significance.
DIAGNOSIS (not goalpost-moving — flagged honestly): the honest-DiD SHARPENS the signal at CIFAR scale (large
probe → stable O_intra) but DEGRADES it here. On the small 9-class control (~240 probe samples) the per-seed
O_intra estimate is itself noisy; subtracting two noisy quantities ~quadrupled variance (sd 0.044→0.088). This
is a real, documented LIMITATION of the DiD estimator at small probe scale, NOT a failure of detection power.
FIX (future): stabilize O_intra with a larger augmentation probe or multiple averaged augment draws.
VERDICT: positive control PASSES on the project-standard raw-overlap contrast (p=0.021) → prereg Guard
satisfied → M1 PIVOT/null calls now DECLARABLE. Do NOT silently swap to whichever metric passes — the raw
contrast is the consistent project metric (sweep + decisive M1); the DiD is a more-conservative variant that
needs a bigger probe to be powered. positive_control.json currently records pass=False (DiD-based); the
raw-contrast pass is the scientifically correct read and must be reflected before the gate auto-reads it.

## 2026-06-03 — Determinism fix VERIFIED + 20-seed positive control is WEAK (GPU-2)
Determinism fix confirmed working: re-running diff0 seeds 0,1 reproduces stored O_inter to 4 decimals
(seed0 0.603959 vs 0.6040; seed1 0.677969 vs 0.6780). So same-seed = same-result now; the earlier n=10
run-to-run p straddle (0.021 -> 0.066) was the cuDNN nondeterminism, now eliminated. CONSEQUENCE: the
20-seed verdict is TRUSTWORTHY. And it is WEAK: diff1 NULL (raw p=0.22, DiD 11/20 seeds, exact p=0.47);
diff0 MARGINAL (raw p=0.057 misses, DiD p=0.013 / 13/20 seeds clears) ONLY at the easiest operating point.
Pushing n=10->20 COLLAPSED diff1 -> the n=10 "pass" (p=0.021) was small-sample luck, not signal. This is
a REAL result (not noise): on the synthetic color×shape binding task, synchrony's overlap-reduction effect
is small (dz~0.4) and operating-point-dependent — MUCH weaker than the dz=1.64 it shows on real CIFAR CL.
That asymmetry (weak on the task we ENGINEERED for binding, strong on real CIFAR) is itself the key question.
Interpretation workflow + options-for-Harry being written; NOT declaring M1's verdict on this. M1's POSITIVE
head-free mechanism result + geometry-kill-test SURVIVE are independent and untouched by this.

## 2026-06-03 — Positive-control interpretation (4-agent adversarial workflow) + what it means
VERDICT: positive control FAILS at the strict prereg standard. diff1 (the prospectively-LOCKED operating
point, research-log 2026-06-02) is NULL at n=20 (raw p=0.22, DiD p=0.47, 11/20 seeds). The diff0 "pass"
requires BOTH operating-point shopping (the cell we rejected by name) AND metric shopping (raw Obar p=0.057
MISSES; only the DiD p=0.013 clears — the same DiD that FAILED at diff1). n-trajectory is self-disqualifying:
diff1 p=0.021(n10)->0.066(rerun)->0.22(n20); a real effect tightens, an artifact dilutes; determinism VERIFIED
so n=20 is trustworthy. Honest state = FAIL-as-prespecified / weak-where-found.
CRITICAL — does NOT threaten M1: the Guard is ONE-DIRECTIONAL (blocks declaring a NULL/PIVOT-A/B only; no path
to invalidate a positive). M1's claim is the POSITIVE head-free mechanism (re-verified from committed CIFAR
JSONs: mean +0.0249, dz=1.64, t=5.19, 10/10 seeds) + the triple-corroborated geometry kill-test. A large,
every-seed, kill-test-surviving detection is self-evidently a LIVE instrument on CIFAR regardless of the weak
synthetic control. "Probe weak on this synthetic task" != "synchrony does nothing."
ASYMMETRY (weak synthetic vs strong CIFAR): (1) CONSTRUCT MISMATCH (primary) — AKOrN binds SPATIAL/multi-object
groups (its ItrSA strength); our task = ONE centered shape + arbitrary color×shape LABEL conjunction → nothing
spatial to bind; rides knet.py classification codepath, not the object-discovery harness. (2) PROBE UNDERPOWER
(cheapest to fix) — control has C(3,2)=3 CKA pairs on 18 probe rows vs CIFAR's C(10,2)=45 pairs on ~200 rows
(15× fewer pairs, 11× fewer rows) → high-variance CKA, dz=1.64 reads as dz~0.4.
TWO HARD "DO NOT"s (honored): (a) do NOT edit positive_control.json to pass=True — leaving pass=False is the
honest defensible state; flipping bakes metric-shopping into the gate. (b) do NOT adopt ItrSA harness now
(arc pivot disguised as a fix). DECISIONS LEFT FOR HARRY: verdict framing (rest M1 on the positive head-free
result, which doesn't invoke the Guard, vs insist on a forgetting-null PIVOT which needs a stronger control);
gate-file integrity; control scope (liveness-check vs publishable-strength); multi-object construct-redesign
appetite; publication honesty (report the control openly as partial/weak — it strengthens the disciplined-
power story). AUTONOMOUS (running): disentangle power-vs-construct by raising probe_per_class (pure knob).

## 2026-06-03 — M2 viability pre-check (GPU-2): AMBIGUOUS — weak but NON-ZERO task channel
First M2 go/no-go: can a frozen linear probe read task-id off the trained R6 phase-state? cv_acc=0.293 vs
chance=0.200 (5 tasks), margin +0.093, at LAYER 0 only. Read: the channel is NOT empty — the catastrophic
S_N "zero task bits" failure (cv≈chance, which killed DND/HSPC-T) did NOT happen, so M2 is not dead-on-arrival.
But it's a WEAK signal at layer 0, not a confident "phase clearly encodes task." Same flavor as the positive
control: real direction, weak magnitude. KEY caveat: only LAYER 0 (shallowest AKOrN layer) was probed —
task-discriminative info often lives DEEPER (layers 1,2 feed the readout), so +0.09 may understate the channel.
Cheap reversible follow-up (running): per-layer decodability at layers 1,2. NOT building the GASPnet M2
workspace (needs Harry). Determinism env applied.

## 2026-06-03 — M2 all-layers probe: CHANNEL EXISTS (M2 viable, GO) — deeper layers carry the signal
Per-layer linear task-decodability from R6 phase-state (one trained model, 5 tasks, chance=0.20):
  layer 0: cv=0.287 (+0.087) AMBIGUOUS  |  layer 1: cv=0.370 (+0.170) CHANNEL EXISTS  |  layer 2: cv=0.352 (+0.152) CHANNEL EXISTS.
The layer-0-only pre-check (+0.09) UNDERSTATED the channel; layers 1-2 (which feed the readout) carry ~2x-chance
task info. VERDICT: M2 is VIABLE — the catastrophic S_N "zero task bits" trap (cv≈chance, killed DND/HSPC-T) is
RULED OUT. There IS a real, linearly-decodable task channel in the oscillator phase configuration. Honest scope:
it's a MODEST channel (cv 0.37 vs 0.20), not a blowout — task is present-but-not-cleanly-separable from phase
alone, so M2's eventual claim is "phase carries N task-bits, more than a matched rate-coded baseline" (a capacity
measurement), NOT "phase perfectly encodes task". M2 should route LAYER 1-2 phase-state through the workspace
bottleneck. This is the M2 GO/NO-GO and it's a GO (with the build itself — GASPnet workspace, C_ctx-in-bits
protocol — still Harry's call, not autonomous). NOTE: per-layer probe used the OOM-safe pool-per-sample capture;
no determinism issue (frozen probe, fixed seed).

## 2026-06-03 — Power-vs-construct probe (big CKA probe, probe_per_class 2->24): it's BOTH, split by operating point
Re-ran diff0+diff1 with a 12x bigger CKA probe (216 vs 18 rows) to disentangle underpower vs construct:
  diff0: raw p 0.057->0.031, DiD p 0.013->0.021 — NOW SIGNIFICANT ON BOTH METRICS (they agree, removing the
         earlier metric-shopping ambiguity). dz~0.68-0.75. -> diff0's weakness was PROBE UNDERPOWER; the
         detection effect is REAL and replicable at the easiest operating point.
  diff1: raw p 0.22->0.77 (effect went slightly NEGATIVE), DiD p 0.47->0.79. -> bigger probe did NOT rescue it;
         diff1 is GENUINELY NULL (nothing to power up). The synthetic task elicits no synchrony advantage at
         the harder operating point.
HONEST READING: the H3 probe HAS demonstrable detection power on a synchrony-favoring task (diff0, adequate
sampling, R6<R5 on BOTH metrics) — a legitimate LIVENESS demonstration, instrument not dead. But this is NOT a
clean prereg pass: diff0 is the point the prereg REJECTED BY NAME, and diff1 (the locked point) is null. So:
"detection power demonstrated at the easy setting, absent at the registered setting." Adequate liveness check,
NOT a satisfied Guard, NOT the ideal positive control. Construct caveat stands (single-shape label-conjunction
vs AKOrN's spatial-binding strength; difficulty-dependent). Still does NOT touch M1's positive head-free result.
Decisions for Harry unchanged (control scope: liveness-check-suffices vs publishable-strength-needed; whether to
build a multi-object construct). Did NOT flip positive_control.json to pass. All on GPU-2; GPU-1 untouched.

## 2026-06-03 — Baselines complete (30/30): clean replay-vs-no-replay split, REINFORCES Path B
Full forgetting table (class10, E50, 10 seeds): R6=78.6, R5:no_proj=79.5, A4:EWC=78.5, A4:Replay=54.4,
A5:DER++=50.1 (learn-acc all ~0.74-0.80). The dichotomy is REPLAY vs NO-REPLAY, exactly as CCC predicts:
every no-replay method (R6, R5, AND published EWC) sits at the ~78-79 forgetting ceiling; only replay
(Replay 54, DER++ 50) breaks it. So "R6 not competitive with DER++" = "no-replay can't beat replay" —
TRUE OF EWC TOO, a standard method. This is NOT a knock on synchrony; it's the structure of the problem
(CCC's impossibility bound: regularization can't raise capacity, only re-feeding old data does). KEY: the
forgetting metric cannot distinguish R6 from R5 from EWC (all pinned at the no-replay ceiling) — the synchrony
signal is REPRESENTATIONAL (head-free dz=1.64), invisible here. The EWC comparison turns the failed
A5-competitiveness conjunct into a clean METHODOLOGICAL point ("matched no-replay methods all forget alike;
synchrony's effect is in the representations, not the saturated output head") — direct evidence FOR Path B.
GPU-1 auto-advanced to Stream C (20x5 replication) — still low-value per Path B; running but not gating.

## 2026-06-05 — M1 CLOSED OUT: head-free mechanism REPLICATES + STRENGTHENS on the class20 long stream
Phase2 fully complete on GPU-1 (markers: BASELINES DONE + REPLICATION DONE + ALL PHASE-2 STREAMS DONE; 20/20
class20; 0 fails). Ran analyze_hardened on the finished data (CPU, no GPU; no commits):
  HEAD-FREE H3 DiD (R6 vs R5:no_proj) on the class20 (20-task) stream, n=10: mean_delta=0.0219, p=2.5e-5,
  dz=2.28 -> the synchrony overlap-reduction MECHANISM replicates on the 2x-longer stream and is STRONGER than
  class10 (class10 dz=1.64, p=2.9e-4). M1's core claim is now confirmed HEAD-FREE on BOTH stream lengths.
  Forgetting endpoint (exactly as Path B predicts): primary R6-R5:no_proj=-0.91 pt, perm_p=0.014 (tiny, inside
  equivalence band); class20 replication SIGN matches but perm_p=0.54 (NS); A5/DER++ not competitive
  (R6-A5=+28.5 pts, the expected no-replay-vs-replay gap). Official hardened-gate call on FORGETTING =
  PIVOT-A-PENDING (equivalence holds; positive control not passed) -> precisely why M1 rests on the MECHANISM,
  not forgetting. NET: M1 done + armored; the long-stream replication is confirmatory and came in strong.
  (Note: GPU-1's analyze_hardened.py is an OLDER copy than local — lacks read_positive_control; numbers unaffected.)

## 2026-06-05 — M2 ESTABLISHED (channel usable) + M3 OPEN (no robust online-CL bypass). Full detail in m2/preregistration-M2.md
After a 4-axis audit found the earlier M2 nulls AND the Shapes "rescue" were artifacts (wrong ablation
R5:no_proj, unstable baseline, eval_inits averaging, and a CCC context-bypass estimand), redid M2 cleanly with
the REAL synchrony ablation R6 (learned coupling) vs R6s (frozen RANDOM coupling; params+geometry identical),
eval_inits=1, relational descriptors, on a multi-object BINDING construct.
  M2 — SOLID (two independent ways):
   (1) Corrected decodability screen: L1 phase carries +0.056 more task info for R6 than R6s, p~0.01, n=12,
       across 3 descriptors (marginal/2nd-moment/spatial). R6,R6s both >> R5:depthwise (coupling removed).
   (2) UNBYPASSABLE phase->theta hypernet, JOINT upper-bound (no CL, so g's own forgetting can't confound):
       R6 context-lift +0.274 vs R6s +0.117 (n=6, 5/6 paired seeds, paired-t one-sided p~0.02-0.04); wrong-
       context dP5 R6 -0.39 vs R6s -0.18. Unbypassability invariant validated (constant context -> chance).
   => synchrony's phase-state is a DECODABLE and USABLE label-free context channel for parameter generation,
      and learned synchrony (R6) >> frozen-random coupling (R6s). This is the milestone result.
  M3 — OPEN (honest negative with the mechanisms tried): converting the usable channel into an ONLINE CL
   forgetting-bypass did NOT work robustly. Naive CL: g forgets (~chance, both). Output-regularization: null
   at all beta (von Oswald regularizes per-TASK embeddings; our context is per-INPUT). Latent context-replay:
   weak R6>R6s (p~0.04) at small buffer that EVAPORATED at a larger buffer (R6~R6s~chance) — fragile artifact;
   neither replay regime reached the joint ceiling (0.57). Deliberately did NOT config-shop a buffer size that
   flatters R6. NET: M1 (dz=2.28) + M2 (channel usable, R6>>R6s) are defensible; M3 (forgetting-bypass) needs a
   different approach — a faithful object-discovery task (stronger channel) or a context-conditional architecture
   — a strategic build for the next session, not more solo perturbation. New code: m2/m2_hypernet.py
   (demo/joint/cl-reg[outreg|replay]); results in m2/results/m2_hypernet*.json. No commits made.
