# GATE A — PRE-REGISTRATION + EVAL MECHANICS (LOCK before any CLEVRTeX retrain)

**Paper:** "Recurrence, not synchrony" — ICLR 2027 main venue.
**Claim under test (native venue):** In AKOrN (Miyato et al., ICLR-2025 Oral) the Kuramoto **coupling term**
`J` is *causally inert for object binding* on AKOrN's OWN unsupervised object-discovery benchmark — the exact
task where the Oral credits coupling. Severing `J` (param-matched, retrained) leaves FG-ARI / MBO unchanged.
**Estimand:** ΔFG-ARI = FG-ARI(full AKOrN) − FG-ARI(coupling-severed AKOrN), at matched capacity / budget /
eval. A tight equivalence interval around 0 = *evidence of absence* (refutes the coupling-credits-binding
reading); a drop toward the ItrSA baseline = coupling MATTERS (refutes our thesis).

**Discipline note.** This doc is locked BEFORE results, in the lineage that caught M1's d-clause defect and
M2's metric-shopping (`experiments/m2/preregistration-M2.md`). Once ratified, the decision rule, equivalence
bounds, seed count, and param-match audit are frozen; deviations must be logged as amendments with date.

> **Source-tree caveat (must-read for reproducibility).** The authoritative trunk + layers + dataset + metric
> code is synced at `/tmp/akorn_src` (exact code that runs on the GPU box) and is cited below by `file:line`.
> The **eval driver itself (`eval_obj.py`) is NOT in that sync** — only the metric leaves
> `source/evals/objs/fgari.py` and `source/evals/objs/mbo.py` are present. The driver text quoted in §1 is the
> upstream `autonomousvision/akorn:eval_obj.py` (fetched from GitHub `main`), and it is **byte-consistent with
> a prior local reconstruction** we already ran (`experiments/m2/eval_obj_shapes.py:35` calls
> `eval_obj.eval_dataset(net, ..., method="kmeans", saccade_r=1, pca=...)`, and that path imports the same
> `model.out[0]` hook + `F.normalize` + `clustering` + `calc_fgari_score`). **GATE-A ACTION ITEM A0 (blocking):
> before the headline run, `diff` the box's `external/akorn/eval_obj.py` against the upstream text in §1 and
> paste the diff into the run log.** If they differ, the line numbers in §1 must be re-pinned. Everything we
> change for the ablation lives in the **trunk** (`KLayer.connectivity`); the readout + clustering + metric
> pipeline (§1) is held byte-identical across arms.

---

## §1 — EXACTLY how FG-ARI and MBO are computed (the fixed deterministic readout)

The result is bulletproof only if the severance changes **only the trunk** and the feature→label→metric
pipeline is a fixed deterministic function of the trunk's output. Below is that pipeline, end to end, with
line cites. Upstream line numbers (`eval_obj.py:NN`) are from the GitHub `main` text fetched for this doc and
**must be re-pinned via A0**; all `klayer.py` / `knet.py` / `fgari.py` / `mbo.py` / `utils.py` cites are from
the synced `/tmp/akorn_src` and are authoritative now.

### 1.1 Which feature is clustered (the "readout z")
- The clustered feature is the input to the **first module of `model.out`**, captured by a forward hook:
  `model.out[0].register_forward_hook(get_activation("z"))` (`eval_obj.py:62`). `get_activation` stores
  `output.detach()` (`eval_obj.py:57-58`).
- For AKOrN, `self.out = nn.Sequential(nn.Identity(), pool, Reshape, Linear, ReLU, Linear)`
  (`knet.py:131-138`). `model.out[0]` is the **`nn.Identity()`** at index 0, so its *output* equals its
  *input* = the per-patch feature map `c` produced by `forward()` right before pooling
  (`knet.py:175-176`: `c, x, xs, es = self.feature(input); c = self.out(c)`). Concretely `c` is the
  **last-layer ReadOut** of the oscillator state: `c = ro(x)` where `ro = ReadOutConv` (`knet.py:113-117, 167`).
  So the feature being clustered is `ReadOutConv(KLayer-final-oscillator-state)`, shape `[B, ch, H/psize, W/psize]`
  = `[B, 256, 16, 16]` for CLEVRTeX (imsize 128, psize 8).
  - **ReadOutConv** (`common_layers.py:64-89`): `x = invconv(x)` (Conv2d `ch→ch*N`) → `unflatten(1,(ch,N))` →
    `x = ||x||_2` over the N (rotating) dim `+ bias`. So the readout is the **per-channel magnitude of the
    oscillator's N-dim group response** — a *phase-invariant* magnitude readout. (This matters mechanistically:
    the clustered feature is a magnitude readout of the oscillator state, §3.)
- The forward is called with `return_xs=True` (`eval_obj.py:72`): `output, _xs = model(imgs, return_xs=True)`.
  `_xs` is unused for clustering; only the hooked `z` is used (`eval_obj.py:76`).

### 1.2 Normalization (fixed)
- `v = F.normalize(v, dim=1)` (`eval_obj.py:79`) — L2-normalize each spatial token's channel vector to the unit
  sphere **before** clustering. Applied for both AKOrN and ViT (so it is arm-invariant). No other scaling.

### 1.3 Saccade tiling (sub-patch upsampling of the feature grid)
- `_imgs, _ = gen_saccade_imgs(images, model.psize, model.psize // saccade_r)` (`eval_obj.py:121`).
- `gen_saccade_imgs(img, psize, r)` (`utils.py:146-153`): bicubically upsamples the image by `psize-r` px, then
  emits the tiles `img[:, :, h:h+H, w:w+W]` for `h,w in range(0, psize, r)`. With **`saccade_r=1` (our default,
  matching `eval_obj_shapes.py:35`), `r = psize//1 = psize`**, so `range(0, psize, psize)` yields exactly ONE
  offset → **a single image, `nh=nw=1`, no uptiling.** The per-tile features are stitched into `nimg` of shape
  `[N, ch, ho*nh, wo*nw]` (`eval_obj.py:127-132`); at `saccade_r=1` this is just `[N, ch, 16, 16]`.
  **LOCK: `saccade_r = 1` for all GATE-A arms** (it is identical across arms, so it cannot confound ΔFG-ARI;
  we fix it to keep the feature grid = the native 16×16, which is what the Oral reports).

### 1.4 PCA (config flag — must be held identical across arms)
- If `pca=True`: `nimg = apply_pca_torch(nimg, n_components=pca_dim)` with `pca_dim=128` (`eval_obj.py:137-141,
  227`). `apply_pca_torch` (`utils.py:107-143`) is **per-sample** PCA: center per image, covariance `[C,C]`,
  `torch.linalg.eigh`, take top-128 eigenvectors, project. Note `n_components(=128) < C(=256)` so PCA is active.
  If `pca=False`, the raw 256-d features are clustered.
- **LOCK: `pca` is a frozen hyperparameter set ONCE and held byte-identical across full vs severed arms.**
  Default for the headline = the value that reproduces the Oral target (see §2.1); record it in the run log.
  Because PCA is deterministic (`eigh`, fixed sign-flip convention `flip` at `utils.py:131-132`) and
  arm-invariant, it cannot confound ΔFG-ARI. (Reproduction sweep in §2.1 pins `pca ∈ {False, True}`.)

### 1.5 Clustering (fixed algorithm + fixed n_clusters per dataset)
- `clustering(x, h, w, method, n_clusters)` (`eval_obj.py:85-106`):
  - **`method="agglomerative"` (the upstream `__main__` default, `eval_obj.py:381`, and the headline method):**
    `x.view(C,-1).T` → `Z = fastcluster.average(x)` (average-linkage, Euclidean) →
    `fcluster(Z, t=n_clusters, criterion="maxclust")` → reshape to `(h,w)` (`eval_obj.py:88-96`). Deterministic.
  - **`method="kmeans"`:** `KMeans(n_clusters, random_state=0, n_init="auto")` (`eval_obj.py:97-103`).
    `random_state=0` ⇒ deterministic. (This is the method `eval_obj_shapes.py` used.)
- **n_clusters is dataset-fixed** (`eval_dataset`): **CLEVRTeX (full/camo/outd) → `n_clusters=11`**
  (`eval_obj.py:241-245`); tetrominoes → 4 (`:235-237`); clevr → 11; dsprites → 7; coco → 7; pascal → 4.
- **LOCK: headline `method="agglomerative"`, `n_clusters=11` for CLEVRTeX-full.** Both are arm-invariant.
  We will *also* report `method="kmeans"` as a robustness column (the inertness should not depend on the
  clusterer); but the **pre-registered primary is agglomerative/11** to match the Oral's reported numbers.

### 1.6 Upsample predicted labels to image resolution (fixed)
- Each per-image label map `pred (h,w)` is upsampled to the input resolution by
  `nn.Upsample(scale_factor=(H/h, W/w), mode='nearest')` (`eval_obj.py:148-151`), then stacked + `.long()`
  (`eval_obj.py:154`). Nearest-neighbor ⇒ label-preserving, deterministic.

### 1.7 FG-ARI (foreground Adjusted Rand Index) — step by step
- Foreground GT is formed as `_gt = ((gt > 0).float() * gt).long()` (`eval_obj.py:163`) — i.e. keep GT label
  values where `gt>0`, zero elsewhere (this only zeroes background; the *per-pixel foreground mask* is applied
  inside the metric).
- `scores["fgari"] = np.array(calc_fgari_score(_gt, preds))` (`eval_obj.py:164`).
- `calc_fgari_score(gt_labels, pred_labels)` (`fgari.py:4-27`): for each image `idx`,
  `area_to_eval = np.where(gt_labels[idx] > 0)` (`fgari.py:21`) selects **only foreground pixels** (drops both
  `-1` ignore and `0` background), then
  `ari = adjusted_rand_score(gt_labels[idx][area], pred_labels[idx][area])` (`fgari.py:23-24`). Returns the
  **list of per-image ARIs** (`fgari.py:26-27`). The headline **FG-ARI = mean over all images** (`print_stats`:
  `np.concatenate(fgaris,0).mean()`, `eval_obj.py:285-286`). This is the standard foreground-ARI used by the
  object-discovery literature (matches `clevr_tex.py:249-260`'s `skip_0=True` ARI semantics).

### 1.8 MBO (Mean Best Overlap) — step by step
- `score, _scores = calc_mean_best_overlap(gt.numpy(), preds.numpy())` (`eval_obj.py:165`).
- `mean_best_overlap_single_sample` (`mbo.py:24-64`): drop `-1` ignore from GT uniques; set `pred=-1` where
  `gt<0`; **drop background label `0` from the GT set** (`mbo.py:48-49`) — so MBO is computed over **foreground
  GT objects only** (this is "MBO", the instance/object MBO; the class-MBO `mbo_c` path needs a semantic GT and
  is not used for CLEVRTex). Build one-hot GT masks and one-hot PRED masks (over ALL pred labels incl.
  background), `iou_matrix = intersection/union` (`mbo.py:4-21, 59-62`), `best_iou = max over pred per GT`
  (`mbo.py:63`), return `mean(best_iou)` over GT objects (`mbo.py:64`). If no FG GT, return `-1` (skipped).
- Batch: `calc_mean_best_overlap` (`mbo.py:67-89`) averages per-sample MBO over samples with value `≠ -1`;
  `print_stats` re-aggregates over the dataset dropping `-1` (`eval_obj.py:287-288`).
- **Headline MBO = dataset mean of foreground-object best-IoU.**

### 1.9 What is held FIXED vs what the ablation changes
- **FIXED (byte-identical across full vs severed):** `model.out` readout head, `ReadOutConv`, `F.normalize`,
  saccade tiling (`saccade_r=1`), `pca` flag + `apply_pca_torch`, `method`, `n_clusters=11`, the upsample,
  `fgari.py`, `mbo.py`, the dataset + crop/resize (`clevr_tex.py:97-212`, `imsize=128`, center-crop 0.8), the
  EMA-weight selection (`eval_obj.py:372-373` loads `["model_state_dict"]` into `EMA` then uses `.ema_model`),
  and `n_clusters` foreground GT construction.
- **CHANGED (the only intervention):** inside the trunk, `KLayer.connectivity` (the coupling map `J`) is
  replaced by a param-matched per-token map (§2.4). `c` (the stimulus bias), `c_norm`, `omg`, `project()`,
  `normalize()`, `gamma`, and the whole Kuramoto iteration structure (`klayer.py:152-165`) are **held fixed**.

---

## §2 — PRE-REGISTERED DECISION RULE

### 2.1 GATE 0 — reproduce the Oral target BEFORE trusting any ablation (blocking)
We do not interpret any ΔFG-ARI until **full AKOrN reproduces the published CLEVRTex-full number** within
tolerance. This protects against "our trainer is broken" reading both arms as noise.
- **Target (AKOrN^attn, L=1, CLEVRTex-full, from the Oral):** **FG-ARI 75.6–75.8, MBO ~55** (the value the
  paper credits to coupling; ItrSA L=1 baseline = 66.07 / 43.41, the ~9.7-pt gap we are interrogating).
- **Config (LOCK, from `train_obj.py` + Oral):** `model=akorn`, `J="attn"`, `L=1`, `T=8`, `ch=256`, `N=4`,
  `psize=8`, `heads=8`, `gta=True`, `c_norm="gn"`, `use_omega=False`, `project=True`, `maxpool=True`,
  `no_ro=False`, `data=clevrtex_full`, `imsize=128`, SimCLR loss `temp=0.1` (`train_obj.py:115,251-256`),
  Adam `lr=1e-3 wd=0` (`train_obj.py:337`), EMA `beta=0.998` (`train_obj.py:348`). Epochs/batch = the Oral's
  CLEVRTeX recipe; **effective batch held constant via grad-accum** under the no-OOM constraint (§4).
- **PASS criterion:** full-AKOrN EMA-model FG-ARI within **±1.5 pt** of 75.7 (i.e. **74.2–77.2**) AND MBO within
  **±2.5 pt** of 55, on the locked eval (§1). If FAIL → fix the trainer/eval (re-pin A0, check `pca`, check
  data crop) and re-run; **no ablation numbers are reported until GATE 0 passes.** Record the exact `pca` value
  that achieves target as the frozen headline `pca`.
- The reproduction run is `n≥3` seeds; report mean±sd; the **gate is on the mean.**

### 2.2 The severance comparison protocol (LOCK)
- **Arms:** (A) **FULL** = AKOrN^attn L=1 as above; (S) **SEVERED** = identical, with `KLayer.connectivity`
  replaced by the param-matched per-token map (§2.4), `J`-output carrying **zero cross-token interaction**.
- **Seeds:** **n ≥ 3 per arm for the headline; n ≥ 5 if the result lands in the inconclusive band (§2.3).**
  (Our mechanistic prior is high-confidence inertness; n=3 is the floor, escalate on ambiguity — same
  inconclusive-band-→-more-seeds rule as M1/M2.) Mechanistic n=12 already exists on the CL context channel;
  GATE A is the native-benchmark confirmation, so seeds buy precision on the equivalence test, not discovery.
- **Budget parity (identical, non-negotiable):** same #epochs, same optimizer (Adam lr=1e-3 wd=0), same LR
  schedule (`LinearWarmupScheduler`, `warmup_iters` as set for FULL), same EMA (beta=0.998, update_every=10,
  update_after_step=200, `train_obj.py:348`), same effective batch (via identical batch×grad-accum), same data
  + augmentation seed policy, same #recurrent steps `T=8`. The ONLY difference between arms is the connectivity
  module class. Log both arms' full argv.
- **Eval parity:** both arms evaluated by the identical §1 pipeline (same `pca`, `method=agglomerative`,
  `n_clusters=11`, `saccade_r=1`, same EMA-model selection). FG-ARI and MBO both reported.
- **Paired reporting:** seeds are paired by index (same data seed) where feasible; report per-seed
  FG-ARI/MBO for both arms.

### 2.3 Equivalence / TOST bounds — what counts as INERT vs MATTERS (LOCK)
We pre-register a **two-one-sided-tests (TOST) equivalence** decision on **ΔFG-ARI = FULL − SEVERED** and on
**ΔMBO**, plus effect size and bootstrap CIs. Bounds are set by the *scientifically meaningful* scale: the gap
the Oral credits to coupling is **FULL(75.7) − ItrSA(66.07) ≈ 9.7 pt FG-ARI** (and 55 − 43.4 ≈ 11.5 pt MBO).

- **Equivalence margin (LOCK): Δ_eq = ±2.0 pt FG-ARI** (and **±2.5 pt MBO**). Rationale: ±2 pt is ~20% of the
  9.7-pt credited gap — comfortably inside run-to-run reproduction noise (our GATE-0 tolerance is ±1.5 pt) yet
  ~5× smaller than the effect coupling would need to "buy binding." A null inside ±2 pt means coupling explains
  at most ~20% of its credited gap, i.e. it is **not** the mechanism.
- **INERT (refutes coupling-credits-binding — our claim CONFIRMED):** TOST rejects both one-sided nulls at
  α=0.05, i.e. the **90% CI of ΔFG-ARI lies entirely within (−2.0, +2.0)** AND severed FG-ARI stays in the
  FULL reproduction band (≥ ~73.7, i.e. does NOT regress toward 66). MBO must likewise sit within ±2.5 pt.
- **MATTERS (refutes our thesis — coupling is load-bearing):** severed FG-ARI **drops by > 2.0 pt** with the
  95% CI of ΔFG-ARI excluding 0, **directionally toward the ItrSA floor (66)**. A drop of ≥ ~5 pt (≥half the
  credited gap) is an unambiguous "coupling matters." (Symmetric: a >2 pt *increase* when severed would also be
  "coupling not needed / harmful" — still consistent with inertness-of-binding, reported honestly.)
- **INCONCLUSIVE band:** |ΔFG-ARI| point estimate ≤ 2 pt but the 90% CI is NOT fully inside ±2 (underpowered)
  → **escalate to n≥5 (then n≥8) seeds** until TOST resolves or MATTERS triggers. Pre-registered: we do not
  declare INERT from a wide CI.
- **Statistics to report (all pre-registered, no post-hoc swapping):**
  1. Per-seed FG-ARI & MBO, both arms; mean ± sd.
  2. ΔFG-ARI, ΔMBO with **bootstrap 95% CI** (10k resamples over the per-image score arrays pooled across
     seeds) AND **90% CI for the TOST**.
  3. **Effect size:** Cohen's d / Hedges' g on the per-seed arm means (and a per-image Cliff's δ as
     distribution-free backup).
  4. **Equivalence verdict (TOST p-values for both one-sided tests).**
  5. A **collapse-distance** number: `(FULL − SEVERED)/(FULL − ItrSA)` — fraction of the credited gap that
     severance erases (0 ⇒ fully inert; 1 ⇒ coupling explains the whole gap). This single number is the
     headline-figure summary.
- **Single locked metric family:** FG-ARI is **primary**; MBO is the **co-primary corroborator** (both must
  agree for an INERT verdict — guards against metric-shopping, the M1/M2 lesson). kmeans + pca-flip are
  robustness columns, not decision metrics.

### 2.4 PARAM-MATCH AUDIT — the #1 kill-risk (BLOCKING; must pass before any Δ is trusted)
If SEVERED is not provably capacity-matched to FULL, a drop reads as "you crippled it," not "coupling is
inert." So the severance must replace the coupling with a **param-matched, per-token (zero cross-token
interaction) map**, holding `c`, `c_norm`, `omg`, `project()`, `normalize()` FIXED.

**What `J` is (the thing we replace), for L=1 CLEVRTex (`J="attn"`):** `self.connectivity = Attention(ch=256,
heads=8, weight="conv", kernel_size=1, gta=True, hw=[16,16])` (`klayer.py:97-108`). Its learnable params
(`common_layers.py:276-328`): `W_qkv` = Conv2d(256→768, k=1) = 256·768 + 768 = **197,376**; `W_o` =
Conv2d(256→256, k=1) = 256·256 + 256 = **65,792**; GTA position matrices `mat_q,mat_k,mat_v,mat_o`, each
`[h·w, head_dim/2, 2, 2]` = `[256, 16, 2, 2]` = **16,384** each → **65,536** total. The coupling is invoked as
`_y = self.connectivity(x)` (`klayer.py:128`) and is the ONLY cross-token mixing in the Kuramoto step (the rest
— `+c`, `omg`, project, normalize — is strictly per-token).

**Severance design (LOCK):** replace `Attention(...)` with a **per-token map of equal parameter count and zero
cross-token interaction**. Concretely a pointwise (1×1) channel MLP applied independently per spatial token,
e.g. `Conv2d(256→768,1)→(nonlinearity-free linear recombination)→Conv2d(...)→Conv2d(256→256,1)` sized so the
**total trainable param count equals FULL's connectivity (W_qkv+W_o+GTA mats) to the parameter**. Because a
1×1 conv has receptive field 1, it **cannot move information between tokens** ⇒ no coupling, by construction,
while matching capacity. Position-dependence that the GTA matrices provided is preserved by giving the severed
map a **per-position affine table of identical shape to the GTA mats** (same 65,536 params, added per-token),
so the severed map has the *same* parameter budget AND the *same* access to absolute position — it only loses
**token–token interaction**. (Rationale: this isolates the single causal axis "cross-token coupling," exactly
as M1/M2 isolated a single flag.)

**Audit protocol (must ALL pass, logged before reading Δ):**
1. **Total-param equality:** `sum(p.numel() for p in FULL.parameters())` ==
   `sum(p.numel() for p in SEVERED.parameters())` **exactly** (print both; the only module that changed is
   `layers[l][0].connectivity`, so the per-module diff must be 0). FAIL ⇒ resize severed map until equal.
2. **Per-module param table:** print the named-parameter shapes of FULL.connectivity vs SEVERED.connectivity;
   assert identical aggregate count and that NO param outside `connectivity` changed shape.
3. **Zero-cross-token proof (mechanical, not by training):** feed a batch where token `j` is perturbed; assert
   SEVERED's connectivity output at token `i≠j` is **unchanged** (Jacobian `∂out_i/∂in_j = 0`), confirming no
   coupling. (A 1×1 conv passes this by construction; the test is the receipt.) Symmetrically confirm FULL's
   attention DOES change at `i≠j` (the thing we removed exists).
4. **Everything-else-fixed proof:** assert `c_norm`, `omg`, `ReadOutConv`, `out`, `gamma`, `project`,
   `normalize` modules are byte-identical objects/shapes across arms (state_dict key set equal except under
   `connectivity`).
5. **FLOP/throughput sanity (secondary):** report train-step wall-clock + FLOPs for both arms; large asymmetry
   is logged but param-equality (1) is the binding criterion.

**Optional analytic corroborator (no retrain) — the frozen-weights J-zero counterfactual.** Independently of
the retrained severance, run the *trained FULL* model with `connectivity`'s output forced to 0 via a forward
hook (`out → torch.zeros_like(out)`), exactly as `experiments/m2/scripts/step42_tangent_decompose.py:81-87`
does. If the object-phase / FG-ARI is unchanged with byte-identical trained weights minus the coupling output,
coupling is shown inert *analytically* (stronger than retrain, because nothing else can compensate). This is a
corroborating prediction in §3, not the headline (the headline is the param-matched retrain).

---

## §3 — MECHANISTIC PREDICTOR (pre-registered: which pattern predicts INERT vs MATTERS)

The mechanism (proved on the CL context channel, n=12; `step42_tangent_decompose.py`) is a **tangent
decomposition of the Kuramoto step** (`klayer.py:152-165`). Since `project()` is LINEAR (`klayer.py:121-124`),
the tangent update splits EXACTLY:
`x ← normalize(x + γ·[Ω x + Proj_x(Jx + c)])`, with `Proj_x(Jx + c) = Proj_x(Jx) + Proj_x(c) ≡ g_J + g_c`,
i.e. a **coupling drive `g_J = Proj_x(Jx)`** + a **stimulus drive `g_c = Proj_x(c)`**. We measure, per step /
per group (`step42_tangent_decompose.py:42-64`): `||g_J||`, `||g_c||`, `ratio_JC = ||g_J||/||g_c||`,
`cos(g_J,g_c)`, `align(x_t,c)`, the global order parameter `R_global`, and FG-ARI under the J-zero
counterfactual.

**Pre-registered predictor mapping (LOCK — stated BEFORE the CLEVRTex numbers):**
- **Predicts INERT (our claim) iff, on FULL CLEVRTex:** (a) `g_J` DOMINATES magnitude (`ratio_JC` large, the
  3–22× we saw on the CL channel — at least `>2×`), BUT (b) `g_J` is **spatially common-mode** (≳90% of its
  energy is the spatial mean / a single shared direction — the "global sync, not binding" signature), AND
  (c) `cos(g_J, g_c) ≈ 0` (coupling pushes orthogonal to the stimulus that actually carries object identity),
  AND (d) `R_global` RISES with coupling on (≈0.80→0.89) — coupling buys **global synchronization**, AND
  (e) **FG-ARI(J-zero) ≈ FG-ARI(full)** within the §2.3 equivalence margin. This conjunction = "coupling raises
  global sync but carries no binding" ⇒ severance should be INERT. **This is the falsifiable forward
  prediction: the retrained ΔFG-ARI in §2.3 should be inside ±2 pt.**
- **Predicts MATTERS (refutes us) iff:** `g_J` is **spatially structured / object-aligned** (low common-mode
  fraction), OR `cos(g_J,g_c)` is appreciably non-zero in an object-consistent way, OR **FG-ARI(J-zero) drops**
  toward the ItrSA floor. Any of these ⇒ expect the retrained severance to also drop (MATTERS).
- **Consistency check (must hold for the paper's story):** the *sign and size* of the analytic J-zero ΔFG-ARI
  should **track** the retrained-severance ΔFG-ARI. If they disagree (e.g. J-zero inert but retrained severance
  collapses), we DO NOT claim inertness — we report the discrepancy and investigate (it would mean retraining
  the severed map recovered binding by a non-coupling route, which is itself interesting but not our claim).

---

## §4 — NO-OOM / COMPUTE CONSTRAINTS (box: 1× RTX 4090, 24 GB)

- **Effective batch held constant across arms** by `batch × grad-accum`; reduce per-step batch and raise
  grad-accum to fit 24 GB. Both arms use the identical effective batch (parity, §2.2).
- **Eval memory:** `eval_dataset` shuffles + batches at `batchsize` (`eval_obj.py:184-205`); clustering is
  per-image on CPU (`fgari`/`mbo` are numpy). **Pool/cap probe size**: cap eval `batchsize` so the per-tile
  feature stack `nimg [N,256,16,16]` + per-sample PCA covariance `[256,256]` fit; PCA runs on GPU
  (`USE_GPU_FOR_PCA=True`, `eval_obj.py:42`) so cap N accordingly. Accumulate scores per batch
  (`scores.append`), never hold the whole dataset's features in memory.
- **Tetrominoes = smoke test ONLY** (cheap end-to-end). It is a KNOWN trap as a headline: AKOrN appendix Table
  10 shows AKOrN^attn (86.19 FG-ARI) does NOT beat its ItrSA baseline (86.81) there — a null on Tetrominoes is
  a reproduction, not evidence. Use it to validate the pipeline (data load → train a few epochs → eval → metric
  runs), never to decide the claim. **CLEVRTeX-full is the decisive venue.**

---

## §5 — HONEST SCOPE (LOCK — what we do and do NOT claim)

- **We claim, narrowly:** in **AKOrN's specific Kuramoto coupling term `J`** (the `connectivity` map,
  `klayer.py:94-110, 126-150`), on **AKOrN's own unsupervised object-discovery benchmark (CLEVRTeX-full,
  L=1, the configuration the ICLR-2025 Oral credits)**, the coupling is **causally inert for object binding**:
  severing it (param-matched, retrained) leaves FG-ARI and MBO within ±2 pt, while the mechanism shows the
  coupling buys **global synchronization (R_global ↑)**, not spatial binding.
- **We do NOT claim that synchrony/binding is universally inert.** In particular we **do not contradict
  Rotating Features** (Löwe et al.) or related slot/phase models where the binding mechanism IS load-bearing —
  those use a *different* binding operator and objective; our result is about *AKOrN's coupling term*, not
  "synchrony in general." The contribution is a **precise, mechanism-level negative result on one celebrated
  model**, isolating *which* component does the work (recurrence + per-token stimulus drive + readout) vs which
  does not (cross-token Kuramoto coupling).
- **We do NOT claim the coupling is useless for everything** — it demonstrably raises global synchronization and
  may matter for other AKOrN tasks (e.g. Sudoku) or other readouts; the locked scope is object binding on this
  benchmark via FG-ARI/MBO.
- **Reproduction honesty:** GATE 0 (reproduce 75.6–75.8) must pass or no claim is made; a failure to reproduce
  is reported as such, not laundered into a null.

---

## §6 — RATIFICATION CHECKLIST (sign before the headline CLEVRTeX run)

- [ ] **A0** `diff external/akorn/eval_obj.py` (box) vs §1 upstream text; line numbers re-pinned; diff in log.
- [ ] **GATE 0** full AKOrN^attn L=1 reproduces FG-ARI 74.2–77.2 / MBO ~52.5–57.5 (n≥3); frozen `pca` recorded.
- [ ] **Param-match audit §2.4** all 5 checks pass; total param count equality printed; J-zero/J-nonzero
      cross-token Jacobian receipts attached.
- [ ] Severance config = FULL config with ONLY `connectivity` swapped; both argvs logged.
- [ ] Equivalence margin ±2.0 FG-ARI / ±2.5 MBO, TOST α=0.05, bootstrap 10k, n≥3 (≥5 if inconclusive) — locked.
- [ ] Mechanistic predictor (§3) measured on FULL and its INERT/MATTERS mapping recorded BEFORE reading Δ.
- [ ] Scope sentences (§5) included verbatim in the paper's claims.

_Ratified / dated: ____________________________   (do not start the headline run until signed)_
