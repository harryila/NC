# PRE-REG — FG-ARI × downstream-usefulness dissociation on the GATE-A ladder (2026-06-14)

Locked BEFORE looking at any usefulness number. Tests the paper's sharpest surviving hook (the OCL
evaluation-crisis angle, [2602.07532](https://arxiv.org/abs/2602.07532) / [2504.07092](https://arxiv.org/abs/2504.07092)):
**can a CLEVRTex AKOrN arm be matched/high on FG-ARI (localization, "where") yet differ on representation
usefulness (object-property decoding, "what")?** If yes, the ladder is a controlled generator of the disjoint
where/what regimes the field just declared an open problem — and "high FG-ARI can be mechanistically hollow"
becomes a demonstrated claim, not an assertion. If no (usefulness tracks FG-ARI monotonically), the sharpest
claim dies and we fall back to the narrower "AKOrN-without-the-sphere" simplification paper.

## Design (clean by construction)
- **Arms (n=1 seed each, the existing GATE-A checkpoints):** full (FG-ARI 75.5), proj-off / A1 (76.7),
  norm-clamp / A3 (80.9), severed J=none (38.5). [FG-ARI from experiments/gateA/RESULTS.md, recomputed-as-available.]
- **Representation probed = the EXACT feature eval_obj clusters for FG-ARI:** the readout map at `model.out[0]`
  input, [B, ch, Hf, Wf]. So "what" and "where" are read off the SAME tensor — no representation mismatch.
- **GT-mask-pooled object vectors (decouples what from where):** for each GT object (flat-mask id v, co-registered
  128×128), mean-pool the readout feature over that object's cells (mask nearest-downsampled to Hf×Wf). The probe
  uses GROUND-TRUTH masks, so it measures property-encoding INDEPENDENT of the model's own segmentation quality
  (a hollow-but-well-segmenting model and a good-rep-but-poorly-segmenting model are both detectable).
- **Targets (object-intrinsic, resize-invariant):** shape, size, color, material — categorical, from the raw scene
  JSON. Position/pixel_coords DROPPED (native-frame coords confounded by the crop+resize). 
- **Probe:** StandardScaler → multinomial LogisticRegression (max_iter 2000). Split BY SCENE 60/40 (no object
  leakage). Report per-attribute TEST accuracy + 1000× object-bootstrap 95% CI + majority-class chance.
- **Usefulness scalar `U` per arm = mean over the 4 attributes of (acc − chance)/(1 − chance)** (chance-normalized,
  so high-cardinality material and 3-way size are comparable). Report raw accuracies too.

## Decision rule (pre-registered)
Let U(arm) be the chance-normalized usefulness; FG(arm) the FG-ARI. Bootstrap CIs define "differ".
- **D1 — Hollow-high-FG-ARI:** norm-clamp has FG > full (80.9 vs 75.5) by ~5pt. If U(clamp) is NOT > U(full)
  (CI overlaps or is below) → the clamp's FG-ARI gain is **hollow** (buys localization score, not better
  object representations). SUPPORTS the hook.
- **D2 — Where/what split under severance:** severed loses ~37 FG-ARI. If U(severed) retains **≥ 60%** of full's
  usefulness (U_sev ≥ 0.60·U_full, CI-separated from a collapse-to-chance) → severing coupling **destroys
  localization but largely preserves property-encoding** → where and what dissociate. SUPPORTS the hook strongly.
- **PRIMARY verdict = DISSOCIATION-REAL if (D1 OR D2) holds beyond CIs.** Then: the "high-FG-ARI hollow / disjoint
  where-what" result is real on AKOrN alone → justifies building the cross-family version (Fork 2/3) and gives the
  paper its field-level altitude. 
- **NULL verdict = NO-DISSOCIATION if** U is rank-consistent with FG across all 4 arms (Spearman ρ(U,FG) ≥ 0.9 AND
  neither D1 nor D2 fires). Then: usefulness just tracks FG-ARI → the "hollow" hook is dead → fall back to the
  single-model "AKOrN-without-the-sphere" simplification paper (no OCL-crisis altitude), and do NOT spend GPU-weeks
  on the cross-family benchmark on this basis.
- **AMBIGUOUS** otherwise (e.g. one of D1/D2 fires but within-CI) → report honestly, treat as weak-positive, decide
  with the n≥3 replication.

## Honest caveats (state in any writeup)
- n=1 model seed per arm — this is a CHEAP DIRECTIONAL GATE, not a final number; probe-level bootstrap CIs only
  quantify object-sampling noise, not model-seed noise. A real claim needs the n≥3 retrain (already planned).
- Single benchmark (CLEVRTex), L=1, ch=256. GT-mask pooling is the clean isolation but discards within-object
  feature structure. Material is high-cardinality (~60) so its probe is the noisiest leg — weight the verdict on
  shape/size/color + material as corroboration.
