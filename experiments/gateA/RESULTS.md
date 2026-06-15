# GATE A — native AKOrN synchrony dissociation (CLEVRTex-full, FG-ARI/MBO)

Status: 2026-06-13. n=1 seed (seed=1234) per arm so far; **n>=3 + equivalence stats are the next step** (rigor campaign).
All arms L=1, ch=256, psize=8, T=8, c_norm=gn, bs=256, 500 epochs (the README CLEVRTex command). Eval = eval_obj.py
(agglomerative clustering of readout features, n_clusters=11).

## 2026-06-14c — CANONICAL: downstream retrieval decider + Slot-Attention cross-model + adversarial-review DOWNSCOPE
The where/what reallocation was converted into the pre-registered DOWNSTREAM DECIDER (PREREG-downstream.md): cross-scene
object retrieval by attribute (leave-one-scene-out cosine; gallery=other scenes; relevance=same attribute), on CLEVRTex-OOD
(held-out materials+shapes), n=2000 scenes / nq=12617 objects. A 2nd architecture (Slot Attention, pretrained CLEVRTex
ckpt, eval-only; trust gate FG-ARI 0.62 reproduces published) was run on the IDENTICAL object set (raw-pixel arm = shared
model-independent floor). A 5-lens adversarial review (workflow wm01luswy) then forced a DOWNSCOPE; numbers self-verified.

LEAD METRIC = mAP@R (prereg-primary); R@1 secondary. OOD retrieval (mAP / R@1):
| arm | role | material (LOCAL) | size (GLOBAL) |
|---|---|---|---|
| raw pixels | shared input floor | 0.553 / 0.712 | 0.343 / 0.346 (=chance) |
| AKOrN severed | ungrouped (per-token MLP) | **0.343 / 0.763** | 0.346 / 0.398 |
| AKOrN ItrSA | attention grouping | 0.103 / 0.487 | 0.352 / 0.475 |
| AKOrN full | synchrony grouping | **0.096 / 0.434** | 0.355 / 0.500 |
| SA encoder | ungrouped substrate | 0.063 / 0.415 | 0.347 / 0.473 |
| SA slots_i3 | grouped slots | 0.047 / 0.133 | 0.349 / 0.412 |

**LOCKED CLAIM (one-directional):** object-centric GROUPING (attention OR synchrony) causally DISCARDS LOCAL per-object
appearance (material) info that the ungrouped substrate retains above every model-agnostic floor (severed material L2-probe
0.469 > raw 0.334 > conv-stem 0.239, CI-disjoint); FG-ARI is anti-correlated with this local-attribute utility. Causal in
AKOrN (param-matched severance −0.039%); reproduces in DIRECTION in Slot Attention (destruction leg only). Novelty vs
Oh-A-DINO (2503.09867, descriptive) = the causal param-matched intervention.

**RETIRED (refuted by our own data) — do NOT claim:**
- "size goes UP / local→global REALLOCATION / double-dissociation crossover": SPATIAL-CONTEXT CONFOUND. L2-probe size:
  raw 0.358, randinit (UNTRAINED full arch) 0.447, severed 0.483, ItrSA 0.604 ≈ full 0.613 → any cross-token op (even
  untrained) recovers size; not grouping-specific. Prereg size-gate (full≥severed) does not isolate grouping. Demote to a
  one-directional claim; the global axis is at most a caveated aside.
- "synchrony-specific" anything: ItrSA (no Kuramoto) == full on every probed attribute (material kNN 0.120 vs 0.121; size
  0.604 vs 0.613). Synchrony = "a pure grouping operator: +12 FG-ARI at ~no representational cost" (gate the +12 on n≥3/TOST).

**SA scope:** SA's effect is R@1-only / near-chance on the primary mAP, and SA FAILS the prereg size-build gate → SA
corroborates the DESTRUCTION LEG ONLY, NOT "principle confirmed across two architectures." n_iters is non-monotone
(slots_i1 material < slots_i3) → the clean axis is ungrouped-substrate→grouped, not a grouping-strength dose-response.
TODO: harmonize the two decision-rule scripts (native_retrieval.py size-CI-gate vs crossmodel_slotattn.py bare ≥) and
rename SA_DIRECTION_CONFIRMED.

**Hard-mask aggregation control (RESOLVED, n=2000 OOD):** pooling the SAME 64-d encoder features by the assigned slot's
HARD winning region (sa_enchard_i3) collapses material the SAME as the SOFT mask and as the GRU slot vector —
encoder 0.063 mAP / 0.415 R@1 → enchard 0.046 / 0.110 ≈ encmask 0.045 / 0.099 ≈ slots 0.047 / 0.134. So the loss is the
grouped SPATIAL FOOTPRINT (the model's imperfect object assignment, FG-ARI 0.52 OOD), NOT soft-mask off-object dilution and
NOT the GRU bottleneck. The dilution confound (adversarial review) is KILLED. (n=12 hard≈encoder hint was noise.)

**Corrected verdict (effect-size + R@1-agreement gate, not bare CI-sep):** SA_DESTRUCTION_LEG_CONFIRMED = TRUE on OOD
(decider; material mAP delta 0.016, R@1 4×) but FALSE in-dist (mAP delta 0.007 < 0.01 floor → near-chance; R@1 still
0.151→0.055). size_build_gate = FALSE and n_iters_monotonicity = FALSE on both (confounds correctly retired). The
OOD>in-dist gap is a live caveat (could be held-out-class separability) — the prereg decider is OOD where it holds.

**Controls queued:** raw/patchfy/randinit/ItrSA MATERIAL floors IN the retrieval metric; scene-level CLUSTER bootstrap
(i.i.d.-query CIs anti-conservative — CI-separation is WEAK; model-seed variance is the binding bar, n≥3 campaign running).
Features saved to results/crossmodel_sa_{full,outd}_features.npz (gitignored) for offline re-scoring (--rescore_npz).

**T-SWEEP DOSE-RESPONSE (t_sweep.py, eval-only, single trained full ckpt, OOD n=320):** vary AKOrN's native Kuramoto
recurrence T∈{1,2,4,6,8} at eval time (grouping-STRENGTH knob, no retraining) — turns the discrete ladder into a
continuous curve, addressing "3 dots not a curve". Canonical FG-ARI (eval_obj, n_clusters=11) + GT-pooled material/size
retrieval from the SAME activation per T:
| T | FG-ARI | material mAP | material R@1 | size mAP |
|---|---|---|---|---|
| 1 | 0.025 | 0.139 | 0.279 | 0.355 |
| 2 | 0.255 | 0.199 | 0.468 | 0.348 |
| 4 | 0.472 | 0.176 | 0.481 | 0.358 |
| 6 | 0.582 | 0.150 | 0.439 | 0.362 |
| 8 | 0.733 | 0.103 | 0.372 | 0.357 |
TWO-PHASE (honest): FG-ARI rises monotonically with T (0.025→0.733). Material utility PEAKS at low grouping (T=2 mAP
0.199 / T=4 R@1 0.481) then DECLINES monotonically as FG-ARI climbs — **ρ(FG-ARI, material mAP)=−1.0 over the converged
regime T≥2** (and ρ=−1.0 for R@1 over T≥4). T=1 is degenerate/under-iterated (FG-ARI≈0, both metrics low → including it
makes the all-5 ρ=−0.40); report all points, highlight the mature-regime trade-off. size mAP FLAT (~0.355, all T) —
consistent with the retired size leg. Reading: early recurrence BUILDS the representation; past T≈2-4, further Kuramoto
grouping TRADES local material for segmentation. The strongest cheap within-model version of the causal anti-correlation.
(t_sweep.py validates against the headline: T=8 OOD FG-ARI 0.726 ≈ canonical in-dist 75.5; material mAP 0.103 ≈ full 0.096.)

**Honest ceiling (review):** as-is TMLR/workshop; after fix-set (n≥3 seeds + drop size leg + SA=destruction + mAP-led +
hard-mask + 2nd dataset family) = borderline-to-solid ICLR poster; NOT spotlight/oral (Oh-A-DINO overlap; both distinctive
selling points refuted). Decisions: ONE-DIRECTIONAL framing; HOLD SynCx (rebuttal-phase); GPU → n≥3 seeds + 2nd dataset.
Artifacts: results/{retrieval_outd.json, crossmodel_sa_full.json, crossmodel_sa_outd.json} (committed @ d1def04).

## 2026-06-14 UPDATE — where/what usefulness probe + controls + ItrSA reframe (n=1, decisive direction)
GATE-A "coupling load-bearing" was reframed via a GT-mask-pooled attribute-decode probe (native_usefulness.py,
PREREG-usefulness.md): does an arm high on FG-ARI (where) differ on per-object property decoding (what)? Severing
coupling halves FG-ARI (75.5->38.5) yet per-object decode does NOT drop. Validated against the validation-workflow's
full confound list (native_usefulness_controls.py + analyze_usefulness.py):
- FLOOR controls: severed material > raw-pixel/random-init/stem floors -> genuine representation, not de-processing.
- L2-NORM control: material gap full->severed identical raw vs cosine-norm (0.207 vs 0.204) -> CONTENT, not a
  common-mode variance artifact.
- PROBE-FREE: kNN-retrieval + KMeans-AMI agree (severed material kNN 0.39 / AMI 0.39 vs full 0.12 / 0.13).
- Verdict W1_REALLOCATION: coupling reallocates capacity -> grouping (FG-ARI) + global size, away from per-object
  material/shape. Mechanism = the 94-96% common-mode smoothing (predicted a priori).
- **ItrSA FLOOR (the reframe):** ItrSA (attention, NO Kuramoto) ~= full on representation (material kNN 0.120 vs 0.121;
  size 0.500 vs 0.516). So the reallocation is driven by cross-token ATTENTION GROUPING, NOT synchrony specifically:
  severed->ItrSA (add attention) crashes material -0.27; ItrSA->full (add synchrony) adds +12 FG-ARI at ~no material
  cost. => HONEST HEADLINE shifts from "synchrony buys segmentation not representation" to "object-centric GROUPING
  (attention or synchrony) trades per-object representation for segmentation; synchrony confers no representational
  advantage over attention." Broader / field-level / subsumes synchrony.
Probe-free 6-arm ladder (material kNN / size kNN): severed 0.39/0.39 | itrsa 0.12/0.50 | full 0.12/0.52 (FG-ARI
38.5 < 63.5 < 75.5). CAVEAT: all n=1 model seed; single benchmark L=1; color dead (dropped). Next gates: n>=3 seeds;
cross-model (SynCx) for the family-level claim. Files: results/{native_usefulness.json, native_usefulness_controls.json,
usefulness_analysis.json}.

## Ladder (n=1)
| arm | what it removes | FG-ARI | MBO | R_global | reading |
|---|---|---|---|---|---|
| full AKOrN^attn | — (reproduction) | **75.5** | 56.5 | 0.68 | faithful (paper 75.6/55.0) |
| J=none | cross-token coupling (param-matched per-token MLP, retrained) | **38.5** | 30.4 | — | coupling/ROUTING load-bearing |
| A1 proj-off | tangent projection (apply_proj=False) | **76.7** | 56.2 | — | projection INERT |
| A3 norm-clamp | unit-sphere normalize -> bounded clamp (+ proj off) | **80.9** | 56.0 | 0.76 | sphere/projection IMPLEMENTATION replaceable |
| A4 ItrSA | whole Kuramoto block (model=vit, T=8) | ~65.7 (paper; ours training) | — | — | no-oscillator floor |

Param-match audit (native_severance.py): J=none mixer vs Attention delta = -0.0389% (exact_match -> 0). Verified on box.

## Frozen counterfactuals (corroboration only; NOT the causal test)
- Frozen J-zero (full ema_499): FG-ARI 0.76 -> 0.36, MBO 0.57->0.24. (native_decompose_ema499.json)
- Native tangent decomposition (ema_499): g_J/g_c = 2.2x, common-mode 94.6%, R_global 0.06->0.68 (same signature as CL).

## DESYNC PROBE — RETRACTED as synchrony-evidence (CIRCULAR)
native_desync_probe_full.json shows FG-ARI craters under phase noise while R_global is robust, BUT the synthesis
adversary correctly flagged this as CIRCULAR: the probe perturbs state x and the readout reads f(x), so it only shows
"corrupting the readout input hurts", not "synchrony matters"; and "R_global robust" is a noise-geometry artifact
(global mean averages out i.i.d. token noise). DO NOT cite as evidence. Replaced by the TRAINED desync arm
(native_phase_noise.py, --phase_noise) which is non-circular.

## Honest standing (per synthesis verdict w2juub3bv)
- CLEAN: CL coupling-inert (n=12, separate setting); the tangent decomposition (positive mechanism); faithful native
  repro; native coupling load-bearing; A3 implementation-replaceable.
- OVERREACH (dropped): "synchrony inert everywhere"; "phase inert natively"; "global sync not the binding variable"
  (circular desync); "oscillator formalism dispensable" -> rescoped to "sphere/projection implementation replaceable".
- NEXT (rigor campaign, path A): n>=3 all arms + TOST/equivalence; the TRAINED desync arm (decisive); an eval-time
  global-rotation isometry control; an isolating control for the CL-vs-native context-dependence; rescope language.

## Files
- native_severance.py (J=none, param-matched), native_norm_ablate.py (A3 sphere ablation),
  native_phase_noise.py (trained desync), native_decompose.py (mechanism + R_global), native_desync_probe.py (RETRACTED).
- akorn_gateA.patch = the GATE-A wiring added to external/akorn/{train_obj,eval_obj}.py (external/akorn is gitignored).
- orchestrate_*.sh / launch_train.sh / watch_a3_rglobal.sh = box-side orchestration.
- results/ = eval logs, decomposition JSONs, clean per-epoch loss trajectories.
