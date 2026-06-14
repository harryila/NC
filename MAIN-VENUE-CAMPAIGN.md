# Main-venue campaign — "Recurrence, not synchrony" (NeurIPS/ICLR) — gated plan (2026-06-12)

DECISION (Harry): the paper's goal is a MAIN ML VENUE (NeurIPS/ICLR), not TMLR/workshop. This plan beats the three
honest blockers a main-track reviewer raises (SCOPE, SCALE, positive-novelty), gate-first so a dead crux is cheap.
Detailed landscape + gate spec come from research workflow w75pvmaaw (folded in on completion). Pre-register every
gate's decision rule BEFORE seeing results (project lesson). NO commit/push without say-so.

## The result we are elevating (confirmed, parameter-matched, adversarially self-verified, n=12)
In AKOrN (ICLR-2025 Oral) used as a label-free CL context channel on binding tasks: the Kuramoto COUPLING is
CAUSALLY INERT (R5d severs all neuron-neuron coupling -> accuracy + object-phase structure unchanged, p=0.59-0.81).
The work is done by: spherical/divisive NORMALIZATION = RETENTION/stability (remove -> diverge + forget; not
formation); RECURRENCE helps binding (T-sweep p=0.001) but is NOT a fixed point (oscillatory refiner, not DEQ).
Synchrony is a bystander -> a clean Roelfsema/Shadlen-Movshon synchrony-skeptic result inside a pro-synchrony model.
[Finishing now: R7clampNP (competition-vs-projection) + single-task (formation-vs-retention) controls; then E1 geometry.]

## The three main-track blockers (must beat all three)
1. SCOPE — results are on a NON-NATIVE repurposing of AKOrN (frozen-random trunk, label-free CL, synthetic tasks).
   We never tested AKOrN's OWN claim (object discovery). "We corrected AKOrN" is really "in a setting it wasn't built for."
2. SCALE — synthetic shapes / MNIST-scale, <=6 classes. No real object benchmark or large model.
3. POSITIVE IS OWNED — "iterative normalized refinement binds" (slot-attention/IODINE/DEQ) + "divisive norm stabilizes
   recurrence" are existing lit; only the dissociation-inside-AKOrN is ours -> a single-model refutation.

## GATE A (THE crux, do FIRST) — coupling-severance on AKOrN's NATIVE object-discovery benchmark
Run our coupling-severance on AKOrN's own object-discovery task, metric FG-ARI / MBO (eval_obj.py /
source/evals/objs/mbo.py). The objs KNet uses the SAME KLayer with J="attn"; sever the coupling by replacing
KLayer.connectivity (the Attention all-pairs mixer) with a PARAM-MATCHED per-token no-coupling map (Identity / 1x1
conv / per-token MLP, zero cross-token interaction), holding the stimulus c, project() and normalize() fixed. The
per-layer readout re-injects c=ro(x) each block, so severance removes ONLY the cross-token coupling channel — the
model is not crippled. REPRODUCE-then-ABLATE.
  ⚠ DATASET CORRECTION (w75pvmaaw native-gate lens): **TETROMINOES IS A TRAP, not "cheapest faithful."** AKOrN's
  OWN appendix Table 10 shows AKOrN^attn FG-ARI 86.19 does NOT beat its ItrSA baseline 86.81 there — the paper text
  flags Tetrominoes as the sole "except for" dataset where synchrony fails to help. A null severance on Tetrominoes is
  a KNOWN reproduction, a reviewer-fatal own-goal. Use Tetrominoes ONLY as a near-zero-cost pipeline smoke test
  (ch=128,psize=4,L=1,epochs=50). **CLEVRTex-full is the ONLY decisive venue** — that's where AKOrN credits the
  coupling: AKOrN^attn 75.79 FG-ARI / 54.94 MBO at L=1 vs ItrSA 66.07 / 43.41 = a ~9.7pt FG-ARI / ~11.5pt MBO gap.
  NO public checkpoints exist (must retrain). Repro target: 75.6–75.8 FG-ARI within ~1–2pt BEFORE severing, else STOP.
  PRE-REG RULE (on CLEVRTex, n>=3 seeds, identical budget, TOST/equivalence bounds + CIs so null = evidence-of-absence):
  - INERT (refutation, BEST): severed FG-ARI stays ~75 (gap survives) -> "the headline mechanism of an ICLR Oral is
    causally inert on its OWN benchmark" -> main-venue-shaped, push hard.
  - MATTERS (context-dependent, still good): severed collapses toward ItrSA ~66 -> "synchrony is load-bearing for
    native object discovery but inert as a label-free CL context channel" -> sharper, honest, still publishable.
  - Either outcome converts narrow->significant. This gate decides the whole framing.
  PARAM-MATCH AUDIT (highest-leverage inoculation): dump full-vs-severed param counts, re-allocate freed Attention
  params onto the per-token path, pre-register the audit — else the pro-synchrony camp reads any FG-ARI drop as "you
  removed capacity, not coupling."
[Compute: CLEVRTex-full ~38GB download (v1+OOD+CAMO); L=1 SimCLR 500ep multi-DAYS on one 4090 at reduced batch+grad-accum.
AKOrN-LARGE (ch=512,L=2) = rental-only strength rung, NOT gate-clearance.]

## GATE A-mech (the WHY, runs alongside — cheap, no extra training) — tangent-space decomposition
Not content with "coupling inert (black box)": OPEN it. The Kuramoto step is Riemannian gradient flow on the product
of spheres; Proj is linear, so the tangent update splits EXACTLY: dxdt = Ωx + g_J + g_c, where g_J=Proj_x(Jx)
(coupling drive) and g_c=Proj_x(c) (stimulus drive). MECHANISTIC HYPOTHESIS: coupling is inert because ||g_J||≪||g_c||
— the oscillators just align to their feed-forward stimulus on the sphere; the lateral coupling never steers the
trajectory. step42 measures, on the TRAINED model (no retrain): per-step ||g_J||/||g_c||, cos(g_J,g_c), align(x_t,c),
and a FROZEN-WEIGHTS J-zero counterfactual (force Jx=0, same trained weights -> compare obj_ami). This converts the
empirical null into a mechanistic LAW and is the analytic backbone of the credit-assignment framing. Independent
theory corroboration: Heeger/Rawat/Martiniani (2409.18946) prove Lyapunov stability of an oscillatory recurrent circuit
with the coupling matrix = IDENTITY (no neuron-neuron coupling needed). Port the SAME decomposition to native CLEVRTex
to PREDICT the gate before the full severance retrain.

## GATE B (scale) — minimum scale that disarms "toy" for a refutation on the model's own benchmark
Reproduce AKOrN-base on the native benchmark at the scale needed (likely Tetrominoes full + CLEVRTex-base; larger if
feasible). Refutations on the original model's own benchmark are held to a lower scale bar than SOTA claims, but we
still need >= the original's evaluation scale. [Scale ladder from w75pvmaaw scale-path lens.]

## GATE C (elevate the positive) — the cross-domain META-THESIS + a THIRD dissociation
The meta-thesis (state as a FALSIFIABLE law, not an anecdote): "under parameter-matched severance, the biologically-
named structural flourish (synchrony, hierarchy) is causally inert in binding/reasoning regimes where iterative
recurrence is present; recurrence is the workhorse." Leg 1 = our AKOrN coupling-severance (binding). Leg 2 = ARC-Prize
HRM re-analysis + TRM (2510.04871) "iteration not hierarchy" in reasoning — CITE, don't run (independent third-party
corroboration, an ASSET). Leg 3 (the one we RUN, picked by w75pvmaaw): a param-matched coupling/phase severance INSIDE
the synchrony-binding family — **Complex AutoEncoder (2204.02075) or SynCx (2405.17283)**, with a χ→cosine swap à la
"Binding Dynamics in Rotating Features" (2402.05627), reporting FG-ARI on the SAME Tetrominoes/multi-object data.
WHY this leg: cheapest+cleanest (tiny models, ~100× faster than slot-attn, one-4090), directly metric-comparable
(FG-ARI not a cross-domain analogy), highest-credibility (same Löwe lineage as AKOrN; they already showed χ↔cosine
interchangeable), and it's a SECOND independent pro-synchrony model (answers "one model isn't a principle").
REJECTED legs + why: Muzellec/Serre complex+Kuramoto (coupling HELPS +10-15% → would falsify); Mamba selectivity
(load-bearing); complex-net phase (carries binding info). Spiking spike-vs-rate = discussion-level corroborator only.
PRE-REGISTER THE FALSIFIER: a significant matched FG-ARI drop under leg-3 severance KILLS the cross-domain law →
retreat to single-model + cited corroborator. Honest scope (mandatory): we do NOT contradict Rotating Features, where
binding IS load-bearing (ARI 0.987→0.059); claim is narrowly AKOrN's coupling term.

## GATE D (write) — main-venue paper, only after A (+B/C as they land)
FRAME = **credit-assignment DISSOCIATION** ("Synchrony is a bystander: where the work actually happens in oscillatory
binding networks"), NOT pure refutation, NOT a new-method paper. (1) LEAD POSITIVE: the tripartite causal decomposition
by ablation not assertion — coupling=inert (proven analytically by GATE A-mech: g_J≪g_c), normalization=retention/
stability, recurrence=binding-but-not-a-converging-fixed-point. (2) GATE the refutation on CLEVRTex FG-ARI/MBO; show
AKOrN's own ItrConv/ItrSA ablation is a CONFOUND (bundles coupling+norm+proj+recurrence) and ours is the decomposition
they never ran. (3) CLOSE with the falsifiable 3-leg meta-law. Write for TWO reviewer pools: object-centric benchmark-
skeptics (Dittadi/Locatello metrics+datasets) and neuro-AI synchrony (Shadlen-Movshon "Synchrony Unbound" + Roelfsema
2023). RHETORICAL LINEAGE (cite as the accept-able shape): Santurkar "How Does BatchNorm Help Optimization?" (1805.11604,
the exact "works for a different reason than claimed" template), "Attention is Not All You Need" (2103.03404), MLP-Mixer
(2105.01601), DyT (2503.10622), ResNet-strikes-back (2110.00476). Cede the positive mechanism explicitly to slot-
attention/IODINE/DEQ + Heeger-DN (2409.18946, stability with coupling=identity). Fix the repo README/research-log
"synchrony isolated" overclaim. 2nd independent review pass.

## DEADLINES + ODDS + KILL-RISKS (w75pvmaaw framing lens)
- DEADLINES (today 2026-06-12): NeurIPS 2026 + ICML 2026 PASSED. **Primary target: ICLR 2027** (~late-Sept 2027
  deadline; natural home — AKOrN is an ICLR Oral → referee-pool overlap raises refutation salience; ~3.5mo suffices for
  the gate if compute starts now). NeurIPS 2027 (~mid-May 2027) fallback.
- HONEST ODDS (decisive result = the CLEVRTex severance gate): pure refutation on CL repurposing only ~10-20% main /
  60-70% TMLR-CoLLAs; gate PASSES + decomposition + leg-3 + cite ARC/HRM ~35-45% ICLR/NeurIPS; gate INVERTS → pivot to
  context-dependent ~30-40% (more defensible). Subtract ~15pp if the param-match isn't bulletproof.
- TOP KILL-RISKS: (1) SCOOP by the Welling/Geiger/Löwe group — they have code+infra to run CLEVRTex severance fastest
  (Goldstone-modes 2605.14685, Kuramoto-Attention 2606.11585 already from them) → move on P1/P2 NOW, timestamp early.
  (2) Param-match hole → any FG-ARI drop read as "you crippled it." (3) CLEVRTex repro failure (no checkpoints).
  (4) Tetrominoes-trap own-goal. (5) "Attention already IS Kuramoto" (2606.11585) → frame as empirical-causal on object
  discovery, not a re-derivation. (6) Over-claim desk-reject → scope narrowly. (7) Meta-thesis-as-anecdote → leg-3 is
  load-bearing. (8) Absence-of-evidence → pre-register TOST. (9) Bar-raising by ImageNet-scale oscillator papers
  (WONN 2605.20922, KoPE 2604.07904) → argue native-fidelity > raw scale.

## Guardrails (non-negotiable)
NO commit/push without say-so. Memory-frugal GPU (no OOM). Pre-register gate rules. Adversarially verify every
positive (the verification already saved us from over-claiming "normalization PRODUCES binding"). Reproduce-before-ablate.
Pull box results local before analysis. Honest scope at every step (the result has only gotten TRUER as we control harder).

## STATUS
- 2026-06-12: campaign opened (main-venue goal). Dissociation result confirmed (controls + E1 finishing). Landscape
  research w75pvmaaw running (native-gate spec, frontier, prior-art de-risk, scale, meta-thesis, framing). GATE A next.
- 2026-06-12 (GATE A in flight): severance VALIDATED on box (param-matched -0.0389%, zero-cross-token, #1 kill-risk
  closed); train_obj/eval_obj patched (--J none); bs=256 fits 4090 (faithful repro, no A100); c_norm=gn confirmed;
  CLEVRTex-full downloaded (50k imgs). L=1 AKOrN^attn reproduction RUNNING (pid 12386, ~9h, loss 4.99->0.75 by ep100,
  FG-ARI 65.2 at ep100 climbing toward 75.6 target).
- **INTERIM NATIVE READ (predictor on ema_99, epoch 100, n=24 -- PRELIMINARY, frozen J-zero != retrain):** the
  coupling MECHANISM generalizes cleanly to native CLEVRTex -- g_J dominates (2.2x), is 94.6% COMMON-MODE, raises
  R_global (0.06->0.67 over T), orthogonal to g_c -- SAME signature as the CL task. BUT the CONSEQUENCE differs:
  frozen J-zero COLLAPSES FG-ARI 0.652->0.347 (-30pt) + MBO 0.51->0.25. So on its own benchmark the trained model
  RELIES on the coupling, UNLIKE the inert CL-channel role. => Likely framing PIVOT from "inert everywhere refutation"
  to the pre-registered **CONTEXT-DEPENDENT story**: the same common-mode/global-sync coupling is causally INERT as a
  label-free CL context channel but LOAD-BEARING for native object-discovery readout. More defensible (avoids the
  Rotating-Features-says-binding-is-load-bearing desk-reject) and mechanistically explained. CAVEATS: ep100/500 (20%
  trained), n=24, and FROZEN J-zero is an UPPER BOUND on severance cost -- the param-matched RETRAIN is decisive.
  NEXT: finish repro (ep500) -> re-run predictor at convergence -> severance RETRAIN (--J none) -> eval both (does a
  param-matched no-coupling model RE-LEARN to bind? recovers=inert/refutation; stays low=coupling matters/context-dep).
