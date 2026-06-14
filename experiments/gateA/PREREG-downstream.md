# PRE-REG — downstream-consequence test: is FG-ARI ANTI-correlated with downstream utility? (2026-06-14)

Locked BEFORE pulling any OOD/CAMO retrieval number. This is the main-venue DECIDER: it converts the decodability
reallocation (severed kNN material 0.39 vs full 0.12; size full 0.52 vs severed 0.39) into a real DOWNSTREAM TASK where
the field's headline metric (FG-ARI) is anti-correlated with utility.

## Scoop position (why the framing is narrow + dated)
Oh-A-DINO ([2503.09867]) already showed slots lose material/texture, keep geometry, via CLEVRTex multi-object
retrieval. The where/what-crisis paper ([2602.07532], Roth group) standardized a unified where/what metric. The BARE
retrieval finding is TAKEN. Our novelty (lead with ONLY this): the **causal, parameter-matched severance LADDER**
(no-grouping[severed] -> attention[ItrSA] -> synchrony[full], param delta -0.0389%) **+ the FG-ARI-vs-downstream-utility
INVERSION curve**, replicated across architecturally-distinct grouping mechanisms. Neither prior paper has the causal
ladder or the anti-correlation curve. Cite both as corroboration. Move on the eval-only AKOrN result this week.

## Task (PRIMARY decider): cross-scene object retrieval by MATERIAL on CLEVRTex-OOD
- OOD by construction (25 held-out materials + 4 held-out shapes) -> a high score CANNOT be memorized texture identity;
  it must be a transferable surface representation. Eval-only (no training); loaders accept data_type='outd'.
- Per arm: GT-mask-pooled object vectors (reuse pool_objects VERBATIM -> decouples what[probed] from where[FG-ARI]),
  L2-normalize. For each query object, gallery = all objects in OTHER scenes (mask same-scene to kill co-occurrence),
  rank by cosine; relevance = same MATERIAL. Report mAP@R + Recall@1, 1000x query-bootstrap 95% CI, n>=3 seeds.
- GT masks given to BOTH arms => the comparison is CONSERVATIVE (the grouped arm gets free localization yet still loses
  on material). That is the point, not a weakness.

## Decision rule (LOCKED)
INVERSION-CONFIRMED (AKOrN) iff ALL three:
  (i)   mAP_material(severed) > mAP_material(full) AND > mAP_material(ItrSA), CI-separated.
  (ii)  Spearman rho(FG-ARI, mAP_material) <= -0.5 over {severed, ItrSA, full, normclamp}.
  (iii) SIZE-CONTROL opposite rank: mAP_size(full) >= mAP_size(severed), CI-separated. (Confound-killer: proves
        material-SPECIFIC reallocation, not "severed is globally a better encoder.")
PRINCIPLE-CONFIRMED iff (i)-(iii) hold on >= 2 of {AKOrN, SynCx, Slot-Attention} (each on its own monotone
  grouping-strength axis; anchored to the shared model-agnostic floors rawpixels/conv-stem/randinit; claim "same
  DIRECTION", not an identical intervention across models).
NULL iff mAP_material(severed) NOT > full beyond CI, OR rho >= 0 -> downstream-decider NOT supported; report honestly,
  retreat to the decodability-only reallocation (clean TMLR-grade result; the n>=3 reallocation is the bankable floor).

## Confound guards (locked)
- Floor anchoring: report retrieval for rawpixels / conv-stem(patchfy) / random-init too; severed must sit between
  patchfy and raw on material (destroyed-then-restored framing, NOT "severed magically better"). full<patchfy~=severed
  already holds at n=1 (confound_verdict=COUPLING_DESTROYS_MATERIAL).
- Min-support >= the same token-count gate across models (neutralize 16x16 vs full-res resolution mismatch).
- color DROPPED (dead: near-chance for raw pixels too). Report material(local) + size(global) + shape.

## Supporting (Task 2, NOT decisive): CLEVRTex-CAMO material robustness
- Train material classifier on full-split GT-pooled vectors per arm, eval on camo-split (object+bg share material =
  the spurious cue common-mode smoothing blends into). retention = acc_camo/acc_full; severed > full CI-separated.
- Keep SUPPORTING-only (OCCAM 2504.07092 owns "segment-then-encode beats slot-OCL on worst-group"); headline = the
  OOD retrieval inversion tied to the causal ladder.

## Honest caveats
n=1 model seed at first read (directional); HEADLINE gated on n>=3 from the rigor campaign. Single dataset family
(CLEVRTex). Cross-model severance is grouping-STRENGTH-at-fixed-params (SynCx iters/bottleneck, SA slots/iters), NOT
AKOrN's operator-swap -> frame as "same direction within each model's own no-grouping->grouped axis".
