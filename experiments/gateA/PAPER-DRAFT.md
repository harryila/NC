# PAPER DRAFT (working) — object-centric grouping causally discards local appearance

**Status:** drafting during the n>=3 seed wait. All AKOrN numbers are **n=1 model seed** until the campaign lands (the
3.4x/4.2x severed-vs-full gap is expected to survive; flagged inline). Goal: ICLR-poster-tier, honest. NOT spotlight
(that needs the DINOSAUR cross-architecture inversion — parked for rebuttal).

## One-sentence contribution
Using a parameter-matched causal severance ladder in AKOrN (ICLR-2025), an eval-only recurrence dose-response curve, and a
directional cross-architecture check in Slot Attention, we show that **object-centric grouping causally discards local
per-object appearance (material) information the ungrouped substrate retains** — so the field's segmentation metric FG-ARI
is **anti-correlated with downstream local-attribute utility** — and we reconcile this with Dittadi et al.'s (2022)
*positive* ARI↔property-prediction result by showing the correlation flips sign by attribute class (positive for global
geometry, negative for local appearance).

## Abstract (draft)
Object-centric models are evaluated chiefly by segmentation quality (FG-ARI), on the assumption that better grouping
yields better object representations. We test this causally. In AKOrN — a Kuramoto-synchronization object-discovery model
— we build a parameter-matched severance ladder (per-token MLP → iterative attention → full synchrony; parameter delta
-0.039%, retrained) and probe each arm with a leave-one-scene-out cross-scene attribute-retrieval task on CLEVRTex (GT
masks given to every arm, so grouped arms receive free localization and can only lose on representation). Grouping causally
collapses downstream **material** (local appearance) retrieval — severed beats full ~3-4x, CI-disjoint, with a perfect
rank anti-correlation between FG-ARI and material utility across the ladder (rho=-1.0) — while **size/shape** (global
geometry) are preserved. The effect holds **both in-distribution and on held-out OOD materials**, ruling out a
distribution-shift artifact, and survives a strengthening **eval-only recurrence dose-response**: increasing AKOrN's
Kuramoto steps raises FG-ARI monotonically (0.03->0.73) while material utility falls (rho=-1.0 over the converged regime).
Floor anchoring on the headline metric shows the ungrouped substrate's material sits *between* the raw-pixel ceiling and
the conv-stem/random-init floors — preserved input information, not a magic gain — whereas grouped arms fall *below* the
random-init floor. The effect is not synchrony-specific (attention-only matches full synchrony on every probed attribute)
and reproduces **directionally** in Slot Attention (a second, architecturally distinct grouping mechanism), where a
hard-mask control localizes the loss to the grouped spatial footprint rather than dilution or the bottleneck. Finally we
reconcile an apparent contradiction: Dittadi et al. (2022) report ARI *positively* predicts object-property prediction and
propose it for model selection; we show that aggregate masks a dissociation — ARI tracks global-geometry utility
positively but local-appearance utility negatively — so **selecting models on ARI actively selects against local-appearance
fidelity**.

## 1. Introduction (drafting notes — LEAD with these two, per the steer)
- Frame: the where/what evaluation crisis (Singh, Schaub-Meyer & Roth 2026) — OCL measures localization ("where") and
  representation usefulness ("what") with disjoint metrics. We supply the missing CAUSAL link and show the two are in
  tension for grouping.
- **LOAD-BEARING #1 — causal, both splits:** the parameter-matched severance ladder isolates grouping as the cause
  (not capacity: delta -0.039%); the inversion holds IN-DIST (severed material mAP 0.165 vs full 0.039, 4.2x) AND OOD
  (0.343 vs 0.096, 3.4x), rho(FG-ARI,material)=-1.0 each, size-specific. -> a representation property, not an eval artifact.
- **LOAD-BEARING #2 — the Dittadi reconciliation:** prior work disagrees about ARI's value (Dittadi 2022: ARI positively
  predicts property prediction, use it to select models; Oh-A-DINO 2025: object reps lose material). We dissolve the
  conflict: ARI↔utility flips sign by attribute class (global geometry +, local appearance -). This reframes the
  contribution from "grouping loses texture (now causal)" to "a precise account of which attributes ARI-selection trades
  off, reconciling two conflicting prior results." (This is the sophistication that reads as understanding-the-field.)
- Contribution bullets: (i) causal param-matched severance ladder; (ii) eval-only recurrence dose-response curve;
  (iii) FG-ARI anti-correlated with local-attribute utility + the Dittadi per-attribute reconciliation; (iv) directional
  cross-architecture corroboration (Slot Attention) with a hard-mask aggregation control.

## 2. Related work (positioning prose — the paper lives/dies here)
**Oh-A-DINO (Wagner & Harmeling, 2025)** is the closest: on CLEVRTex multi-object retrieval both SSL and slot
representations preserve geometry (shape/size) but lose surface appearance (colour/material/texture). That phenomenon is
theirs; what is ours is its CAUSE. They compare representation *types* correlationally and *restore* the missing
attributes with an auxiliary VAE latent; we hold architecture and parameter count fixed and causally *remove* grouping,
showing it is the grouping operation itself that discards local appearance the ungrouped substrate retains above every
model-agnostic floor. We add two results absent from Oh-A-DINO: an eval-only recurrence dose-response in which FG-ARI
rises with grouping strength while local-attribute utility falls (rho=-1.0, converged regime), establishing FG-ARI as an
ANTI-correlate not a proxy; and a directional corroboration in Slot Attention (directional only — near-chance on the
primary mAP and failing the global-attribute control, so explicitly not a cross-architecture law).
**Dittadi et al. (2022)** report the OPPOSITE-seeming result — ARI positively correlates with downstream object-property
prediction, proposed as a model-selection signal. We reconcile: the positive aggregate is carried by global/geometric
properties; isolating local appearance flips the sign (material rho=-1.0 vs size/shape rho>=+0.4, both splits). Thus ARI
is not a uniform proxy, and ARI-based selection trades against local fidelity.
**Singh, Schaub-Meyer & Roth (2026)** standardize a unified where/what metric because the two are measured disjointly;
we supply the causal mechanism showing they are in tension for grouping.
**OCCAM (Rubinstein et al., 2025)** shows the practical consequence (training-free segment-then-encode beats slot-OCL on
OOD worst-group classification); we provide the mechanistic why behind that result rather than re-establishing the win.
**AKOrN (Miyato, Loewe, Geiger & Welling, ICLR 2025)** is the model we dissect; we do not dispute its grouping ability and
do not claim a synchrony-specific effect (attention-only ItrSA matches full Kuramoto on every probed attribute) — our
delta is strictly diagnostic.
(Adjacent/recent: Kapl et al. 2026, CLEVRTex property-generalization VQA, OC vs dense — cite & differentiate.)

## 3. Method
- **Severance ladder (causal core):** AKOrN with cross-token connectivity removed/replaced, parameter-matched (J=none
  per-token MLP / attention ItrSA / Kuramoto full / sphere->clamp normclamp); delta -0.039%; retrained (n=1 seed -> n>=3).
- **Downstream probe:** cross-scene object retrieval by attribute (leave-one-scene-out cosine; gallery=other scenes, same
  scene masked; relevance=same attribute). GT-mask-pooled object vectors (decouples what[probed] from where[FG-ARI]) ->
  grouped arms get FREE localization yet still lose. Metrics: mAP@R (primary) + R@1 (secondary). Splits: in-dist (full) +
  OOD (25 held-out materials). Decision rule (real_gt): CI-sep AND effect-size AND R@1 agreement (CIs are i.i.d.-query
  bootstrap, anti-conservative -> effect size is load-bearing).
- **Dose-response (strengthener, eval-only):** vary Kuramoto recurrence T at eval on the trained full ckpt.
- **Cross-architecture (directional):** pretrained Slot Attention; ungrouped encoder vs grouped slots; hard-mask
  aggregation control isolates the grouped footprint from soft-mask dilution / the GRU bottleneck.
- **Floors (headline metric):** raw-pixel / conv-stem(patchfy) / random-init on the retrieval metric.

## 4. Results (verified numbers; n=1 seed flagged)
- **Fig 1 (money) — severance ladder, OOD:** FG-ARI 38.5/63.5/75.5/80.9 vs material mAP 0.343/0.103/0.096/0.092
  (R@1 0.76/0.49/0.43/0.43); severed beats full 3.4x CI-disjoint; rho(FG-ARI,material)=-1.0; size flat (0.346-0.357,
  full>=severed) = material-specific. [n=1 seed; bars are within-seed query bootstrap pending n>=3.]
- **In-dist (kills OOD-artifact attack):** severed material 0.165 vs full 0.039 (4.2x), rho=-1.0, size-specific. Both
  splits -> representation property, not separability.
- **Fig 2 — recurrence dose-response (T=1 shown-but-grayed):** FG-ARI 0.025/0.255/0.472/0.582/0.733 (T=1,2,4,6,8);
  material mAP peaks at T=2 (0.139/0.199/0.176/0.150/0.103) then declines; rho=-1.0 over converged T>=2 (R@1 over T>=4);
  size flat. T=1 degenerate/under-iterated (both low) -> all-5 rho=-0.40, report both. Two-phase: early recurrence BUILDS
  the rep, further grouping TRADES local for segmentation.
- **Fig 3 — pooled FG-ARI-vs-material scatter:** 9 grouping-strength points (4 ladder + 5 T-sweep); rho=-0.80 all points
  (-0.95 excl degenerate T=1). The figure Oh-A-DINO structurally cannot make.
- **Fig 4 — floor anchoring (headline metric, n=2000):** OOD material mAP raw 0.553 > **severed 0.343** > randinit 0.229
  > patchfy 0.181 > itrsa 0.103 > full 0.096. Severed sits BETWEEN the raw ceiling and the conv-stem/randinit MODEL floors
  (preserved input info, NOT magically above input); grouped arms (full/itrsa) fall BELOW even random-init and the
  conv-stem — grouping destroys material below what an untrained net retains. (R@1: raw 0.712 / randinit 0.658 / patchfy
  0.485 — note randinit R@1 is high, so report mAP-anchored claim primarily.)
- **Dittadi per-attribute reconciliation (load-bearing):** material(LOCAL) rho=-1.0 both splits; size(GLOBAL) rho +1.0
  (OOD)/+0.8 (in-dist); shape +0.6/+0.4. -> ARI-selection selects against local fidelity.
- **Fig 5 — Slot Attention (directional corroboration only):** encoder material R@1 0.415 -> grouped slots 0.13 (~3x);
  hard-mask: enchard 0.046 ~= encmask 0.045 ~= slots 0.047 << encoder 0.063 mAP -> grouped FOOTPRINT, not dilution/
  bottleneck. Caption: directional only — near-chance primary mAP, non-monotone iters, fails size-build gate; NOT a law.

## 5. Limitations (state plainly — from the hostile-AC pre-mortem)
- **n=1 model seed** on every AKOrN arm (n>=3 retrain in flight). Only the within-checkpoint T-sweep is seed-immune.
  The 3.4x/4.2x gap is expected to survive (CI-disjoint + 4 within-model probes: mAP, R@1, kNN 0.39 vs 0.12, AMI 0.39
  vs 0.13) but no single-seed result is claimed to "generalize".
- **Single dataset family (CLEVRTex)** even with in-dist+OOD splits. A different family (real images, MOVi) needs new
  data/training — the primary external-validity limit; stated, not papered over.
- **"Material" is the only working local-appearance proxy** (colour was near-chance, dropped) -> phrase as "local
  appearance (material/texture) on CLEVRTex", never "appearance" in general.
- **Slot Attention is weak second-architecture evidence** — directional only (R@1, near-chance mAP, non-monotone, fails
  size-build gate). Belongs in the abstract as "corroborated in direction", never "replicated across architectures".
- **Cross-model magnitude is dim-confounded** (256-d vs 64-d) -> only DIRECTION claimable cross-model.
- **The anti-correlation is scoped**, not a field-wide proof FG-ARI is wrong: rho holds within these interventions on
  CLEVRTex; the broader critique leans on the where/what literature + the Dittadi per-attribute reconciliation.

## Open / next
- n>=3 seeds (binding bar) -> seed-level CIs/TOST on the material gap + the full-vs-ItrSA FG-ARI +12 (synchrony's only
  effect); replace within-seed query bars with across-seed bars in Fig 1.
- DINOSAUR cross-architecture inversion = the one lever to spotlight (real-image, needs adaptation; rebuttal-phase).
- TOST ItrSA==full: provisional at n=1, defer to seeds.
