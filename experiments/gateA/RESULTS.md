# GATE A — native AKOrN synchrony dissociation (CLEVRTex-full, FG-ARI/MBO)

Status: 2026-06-13. n=1 seed (seed=1234) per arm so far; **n>=3 + equivalence stats are the next step** (rigor campaign).
All arms L=1, ch=256, psize=8, T=8, c_norm=gn, bs=256, 500 epochs (the README CLEVRTex command). Eval = eval_obj.py
(agglomerative clustering of readout features, n_clusters=11).

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
