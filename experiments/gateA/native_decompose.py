"""
GATE A — NATIVE MECHANISTIC PREDICTOR for the AKOrN object-discovery model (J="attn").
======================================================================================

WHAT THIS IS
------------
A pre-retrain *predictor*. It ports our tangent-space decomposition + global-sync probe
(experiments/m2/scripts/step42_tangent_decompose.py) from the M2/sudoku arch onto AKOrN's
OWN native object-discovery network (source/models/objs/knet.py, J="attn", the ICLR-2025
Oral setting that CREDITS the Kuramoto coupling for binding). Given a TRAINED objs
checkpoint it measures, with frozen weights and no retrain, whether severing the coupling
will be inert on FG-ARI / MBO — BEFORE we spend the expensive coupling-severed retrain.

THE MECHANISM WE ARE PORTING (read from the actual code, cited)
--------------------------------------------------------------
The Kuramoto step is Riemannian gradient flow on the product of unit n-spheres
(klayer.py:152-165):
        c = self.c_norm(c)                                  # klayer.py:155  (done ONCE)
        x = normalize(x, n)                                 # klayer.py:156
        for t in T:
            dxdt, _ = self.kupdate(x, c)                    # klayer.py:160
            x = normalize(x + gamma*dxdt, n)                # klayer.py:161
and kupdate (klayer.py:126-150) is:
        _y = self.connectivity(x)                           # klayer.py:128  == J x
        y  = _y + c                                         # klayer.py:130
        y, x = reshape(y,n), reshape(x,n)                   # klayer.py:137-138  (group dim = 2)
        y_yxx, sim = self.project(y, x)                     # klayer.py:142
        dxdt = omg_x + reshape_back(y_yxx)                  # klayer.py:147
project() (klayer.py:121-124) is the per-group tangent projection
        Proj_x(y) = y - (sum_dim2(x*y)) * x
which is LINEAR in y. Therefore the tangent update splits EXACTLY:
        Proj_x(Jx + c) = Proj_x(Jx) + Proj_x(c)
        g_J = Proj_x(J x)   (lateral COUPLING drive; J="attn" => J x = Attention(x), the
                             all-pairs output, common_layers.py:343-386)
        g_c = Proj_x(c)     (feed-forward STIMULUS drive; c is already c_norm'd because
                             c_norm runs ONCE in forward at klayer.py:155, NOT inside kupdate)
NOTE on J="attn" (knet default, train_obj.py:155 with --J attn): self.connectivity is an
Attention module (klayer.py:97-108). connectivity(x) is the all-pairs attention output, i.e.
exactly "sum_j J_ij x_j" in the Kuramoto sense; g_J = Proj_x(attention(x)). (klayer.py:127 comment)

WHAT WE MEASURE (per KLayer, per Kuramoto step t, per n-sphere group)
---------------------------------------------------------------------
  ||g_J|| = ||Proj_x(Jx)||              (coupling drive magnitude)
  ||g_c|| = ||Proj_x(c)||               (stimulus drive magnitude)
  ratio   = ||g_J|| / ||g_c||           (does coupling dominate the step?)
  cos(g_J, g_c)                         (orthogonal noise vs aligned steering?)
  common-mode fraction = ||avgpool3x3(g_J)|| / ||g_J||   (spatially shared push?  high => carries
                                         no per-location binding signal; ~1 => pure common-mode)
  R_global = global Kuramoto order parameter of the layer's oscillator state x
                                         (mean resultant length across ALL spatial tokens per
                                          group; the "global synchrony" the coupling buys)
PLUS the decisive frozen-weights counterfactual:
  FG-ARI(full) and FG-ARI(J-zero), MBO(full) and MBO(J-zero) computed with eval_obj.py's
  EXACT readout+clustering: feature = the spatial map fed into model.out (hooked at model.out[0],
  an nn.Identity, knet.py:131-138), F.normalize over channels, then per-image clustering into
  n_clusters, FG-ARI via source/evals/objs/fgari.py and MBO via source/evals/objs/mbo.py.
J-zero = a forward-hook on each KLayer.connectivity that forces its output to 0 (klayer.py:128
=> _y=0 => g_J=0), with EVERY other trained weight (omega, c_norm, project, readout, out) byte
-identical. This isolates the coupling OUTPUT only; it does NOT change capacity, so it is the
analytic upper bound on what coupling-severance can cost.

PRE-REGISTERED PREDICTION  (write this down BEFORE looking at the numbers)
-------------------------------------------------------------------------
  IF on the trained objs checkpoint we observe:
     (a) g_J DOMINATES g_c (ratio >> 1, matching the 3-22x we saw on M2), AND
     (b) g_J is COMMON-MODE-dominant (common_mode_frac(g_J) high, e.g. >= ~0.9, i.e. >=90%
         spatially shared), AND
     (c) the coupling RAISES global synchrony (R_global increases over the T steps and is
         higher than the J-zero run), AND
     (d) FG-ARI(J-zero) ~= FG-ARI(full)  and  MBO(J-zero) ~= MBO(full)
         (no material drop, e.g. |dFG-ARI| within a few points, well under the ~9.7pt
          AKOrN^attn-vs-ItrSA gap that coupling is credited for),
  THEN we PREDICT the parameter-matched coupling-severance retrain will be INERT on the
  native benchmark: the coupling buys GLOBAL sync, not binding. The J-zero number is the
  cheap leading indicator; the param-matched retrain is the confirmatory test (Gate A body).
  FALSIFIER: if FG-ARI(J-zero) collapses (>> few-pt drop) while g_J is NOT common-mode, the
  coupling is carrying binding and the severance is expected to hurt — prediction refuted.

NB this is a frozen-weights ablation, NOT the param-matched severance retrain. A J-zero drop is
an UPPER bound on the severance cost (severance gets to re-learn a per-token replacement); a
J-zero null is therefore strong evidence the retrain will also be null. The #1 kill-risk
(capacity-matching) lives in the retrain, not here.

OOM-SAFETY (single RTX 4090, 24GB; hard constraint: NO OOM)
----------------------------------------------------------
  * small probe batch (--bs default 8), --n_images default 32, torch.no_grad everywhere,
  * clustering done per-image on CPU after pooling features off-GPU,
  * torch.cuda.empty_cache() between the full and J-zero passes and between batches,
  * the decomposition records only scalar means per step (no per-token tensors retained).

USAGE (run on the GPU box where torch + /tmp/akorn_src deps live)
----------------------------------------------------------------
  python experiments/gateA/native_decompose.py \
      --ckpt /path/to/ema_model.pth \
      --src /tmp/akorn_src \
      --data tetrominoes            # cheap smoke test; use clevrtex_full for the decisive run
      --n_images 32 --bs 8 --device cuda \
      --out experiments/gateA/native_decompose_tetro.json
  # decisive venue:
  python experiments/gateA/native_decompose.py --ckpt .../ema_model.pth --data clevrtex_full \
      --data_root /path/to/clevrtex_full --n_clusters 11 --out .../native_decompose_clevrtex.json

  Model hyperparams must match the checkpoint's training run (defaults below mirror the
  AKOrN^attn objs config: N=4, ch=256, L=1, T=8, psize=8 [Tetro psize 4], J=attn, gta=True).
  Pass --ch/--L/--T/--N/--psize/--J/... to override if your checkpoint differs.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# tangent-projection helpers (mirror klayer.py:121-124 project(); group/within-group dim=2)
# --------------------------------------------------------------------------------------
def _proj(v_g, x_g):
    """Per-group tangent projection  Proj_x(v) = v - <v,x> x , summed over within-group dim 2.
    v_g, x_g are reshaped oscillator tensors [B, G, n, H, W] (klayer reshape, kutils.py:9-13)."""
    sim = (x_g * v_g).sum(2, keepdim=True)
    return v_g - sim * x_g


def _gnorm(v_g):
    """Per-group vector norm over the n-sphere dim 2 -> [B, G, H, W]."""
    return v_g.norm(dim=2)


def _common_mode_frac(g_g, ksize=3):
    """Common-mode fraction of a tangent drive: ||avgpool_{k x k}(g)|| / ||g||, averaged.
    g_g is reshaped [B, G, n, H, W]. We avg-pool the raw oscillator field spatially (the shared
    'global' component) and compare its energy to the total. ~1 => the drive is almost entirely a
    spatially-common push (carries no per-location/binding signal); low => spatially structured.
    Pooling is applied per (group,n)-channel over H,W with reflect padding so edges are handled."""
    B, G, n, H, W = g_g.shape
    flat = g_g.reshape(B, G * n, H, W)
    pad = ksize // 2
    pooled = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="reflect"), ksize, stride=1)
    pooled = pooled.reshape(B, G, n, H, W)
    num = _gnorm(pooled)               # [B,G,H,W] energy of the locally-shared component
    den = _gnorm(g_g).clamp_min(1e-9)  # [B,G,H,W] total energy
    return float((num / den).mean())


def _order_param(x_g):
    """Global Kuramoto order parameter R_global of an oscillator state.
    x_g reshaped [B, G, n, H, W]; each (b, g, :, h, w) is a unit n-vector (post-normalize,
    klayer.py:161). R = || mean over all spatial tokens of the unit vectors || per (b,g),
    i.e. the mean resultant length, in [0,1]. 1 => globally phase-locked, ~1/sqrt(N_tokens)
    => incoherent. We average over groups and batch to a single global-synchrony scalar."""
    B, G, n, H, W = x_g.shape
    unit = F.normalize(x_g, dim=2)                 # ensure unit (state is already normalized)
    mean_vec = unit.mean(dim=(3, 4))               # [B,G,n] resultant vector across tokens
    R = mean_vec.norm(dim=2)                        # [B,G] resultant length
    return float(R.mean())


# --------------------------------------------------------------------------------------
# checkpoint loading — like eval_obj.py: build net, wrap EMA, load, use .ema_model
# --------------------------------------------------------------------------------------
def build_net(args, imsize):
    from source.models.objs.knet import AKOrN
    net = AKOrN(
        n=args.N,
        ch=args.ch,
        L=args.L,
        T=args.T,
        gamma=args.gamma,
        J=args.J,                       # "attn" for the native objs / ICLR-Oral setting
        ksize=args.ksize,
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        c_norm=args.c_norm,
        psize=args.psize,
        imsize=imsize if args.model_imsize is None else args.model_imsize,
        autorescale=args.autorescale,
        init_omg=args.init_omg,
        learn_omg=args.learn_omg,
        maxpool=args.maxpool,
        project=args.project,
        heads=args.heads,
        use_ro_x=args.use_ro_x,
        no_ro=args.no_ro,
        gta=args.gta,
    )
    # A3: rebuild the sphere-norm-ablated architecture so the checkpoint loads cleanly + R_global is measured under it
    if getattr(args, "norm_ablate", "none") != "none":
        if "/root/NC" not in sys.path:
            sys.path.insert(0, "/root/NC")
        from experiments.gateA.native_norm_ablate import norm_ablate_akorn
        net = norm_ablate_akorn(net, variant=args.norm_ablate, no_proj=True, verbose=False)
    if getattr(args, "phase_noise", 0.0) > 0:
        if "/root/NC" not in sys.path:
            sys.path.insert(0, "/root/NC")
        from experiments.gateA.native_phase_noise import phase_noise_akorn
        net = phase_noise_akorn(net, args.phase_noise, verbose=False)
    return net


def load_checkpoint(net, ckpt_path, device):
    """Load a trained objs checkpoint the way eval_obj.py does: wrap the net in an EMA, load the
    EMA state_dict, and read the smoothed weights out of ema.ema_model. Supports the formats
    train_obj.py writes (training_utils.save_model -> {'model_state_dict': EMA.state_dict()} and
    the bare ema.state_dict() in ema_model.pth), plus a plain model_state_dict fallback."""
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob

    # Is this an EMA state_dict (ema_pytorch prefixes keys with 'ema_model.' / 'online_model.')?
    is_ema = isinstance(sd, dict) and any(
        k.startswith("ema_model.") or k.startswith("online_model.") or k.startswith("initted")
        for k in sd.keys()
    )
    if is_ema:
        from ema_pytorch import EMA
        ema = EMA(net, beta=0.998, update_every=10, update_after_step=200)
        missing, unexpected = ema.load_state_dict(sd, strict=False)
        model = ema.ema_model
        load_kind = "ema.ema_model"
    else:
        # plain model state dict (e.g. checkpoint_*.pth / model.pth)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        model = net
        load_kind = "plain model_state_dict"

    # Guard the silent-random-weights trap: strict=False leaves mismatched params at init, which
    # would corrupt c/g_c/R_global/FG-ARI with NO error. Abort loudly if real (non-EMA-bookkeeping)
    # param keys are missing -> architecture/config disagrees with the trained checkpoint.
    real_missing = [k for k in missing if not (k.endswith("initted") or k.endswith("step"))]
    if real_missing:
        raise RuntimeError(
            "load_checkpoint: %d param keys MISSING (architecture/config likely mismatches the "
            "trained checkpoint): %s ... check --ch/--T/--psize/--c_norm/--gta match the training run."
            % (len(real_missing), real_missing[:8]))

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(load_kind=load_kind,
                       n_missing=len(missing), n_unexpected=len(unexpected),
                       missing_head=list(missing)[:8], unexpected_head=list(unexpected)[:8])


# --------------------------------------------------------------------------------------
# instrumented forward: monkeypatch each KLayer.kupdate to record the decomposition.
# The native objs block is nn.ModuleList([klayer, readout, linear_x]) (knet.py:123) so the
# KLayer is net.layers[l][0]  (cf. step42 used [2] on a different arch).
# --------------------------------------------------------------------------------------
def make_patched_kupdate(klayer, layer_idx, rec):
    """Returns a function bound as klayer.kupdate that records the per-step decomposition AND
    reproduces the EXACT original update (klayer.py:126-150) so the forward trajectory is
    byte-identical to the unpatched model."""
    from source.layers.kutils import reshape, reshape_back

    def patched(x, c):
        # --- exact original quantities (klayer.py:128-135) ---
        _y = klayer.connectivity(x)                       # J x  (Attention all-pairs output)
        y = _y + c                                        # klayer.py:130
        if hasattr(klayer, "omg"):
            omg_x = klayer.omg(x)                          # klayer.py:132-133
        else:
            omg_x = torch.zeros_like(x)

        xg = reshape(x, klayer.n)
        yg = reshape(y, klayer.n)

        # --- DECOMPOSITION (exact linear split of Proj_x(Jx+c)) ---
        gJ = _proj(reshape(_y, klayer.n), xg)              # g_J = Proj_x(Jx)
        gc = _proj(reshape(c, klayer.n), xg)               # g_c = Proj_x(c)   (c already c_norm'd)
        nJ = _gnorm(gJ)
        nc = _gnorm(gc)
        cosJc = (gJ * gc).sum(2) / (nJ * nc).clamp_min(1e-9)
        rec[layer_idx].append(dict(
            gJ_norm=float(nJ.mean()),
            gc_norm=float(nc.mean()),
            ratio_gJ_gc=float((nJ / nc.clamp_min(1e-9)).mean()),
            cos_gJ_gc=float(cosJc.mean()),
            common_mode_frac_gJ=_common_mode_frac(gJ),
            common_mode_frac_gc=_common_mode_frac(gc),
            R_global_pre=_order_param(xg),                 # R of state x BEFORE this step's update
        ))

        # --- exact original update (klayer.py:141-150) ---
        if klayer.apply_proj:
            y_yxx, sim = klayer.project(yg, xg)
        else:
            y_yxx = yg
            sim = yg * xg
        dxdt = omg_x + reshape_back(y_yxx)
        sim = reshape_back(sim)
        return dxdt, sim

    return patched


def run_decomposition(model, imgs, device):
    """Run a single instrumented full forward; return per-layer per-step decomposition records
    and the final-state R_global per layer (post last update)."""
    rec = {l: [] for l in range(model.L)}
    originals = {}
    for l in range(model.L):
        klayer = model.layers[l][0]
        originals[l] = klayer.kupdate
        klayer.kupdate = make_patched_kupdate(klayer, l, rec)

    from source.layers.kutils import reshape
    with torch.no_grad():
        c, x, xs, es = model.feature(imgs.to(device))

    # restore
    for l in range(model.L):
        model.layers[l][0].kupdate = originals[l]

    # final-state R_global per layer (xs[l+1] is the list of states for layer l; take last)
    final_R = {}
    with torch.no_grad():
        for l in range(model.L):
            xfin = xs[l + 1][-1]
            final_R[f"L{l}"] = _order_param(reshape(xfin, model.n))
    return rec, final_R


# --------------------------------------------------------------------------------------
# eval_obj.py-exact readout + clustering, for FG-ARI / MBO.
# Readout feature = the spatial map fed into model.out (hooked at model.out[0]=nn.Identity,
# knet.py:131-138). That is c = ro(x_final), shape [B, ch, H, W] (knet.py:167,175). We hook
# out[0] so we capture the same feature whether or not pooling/MLP run.
# --------------------------------------------------------------------------------------
def _capture_readout_feature(model, imgs, device):
    feat_holder = {}

    def hook(_mod, inp, _out):
        # input to nn.Identity == the spatial readout feature c before pooling
        feat_holder["f"] = inp[0].detach()

    h = model.out[0].register_forward_hook(hook)
    with torch.no_grad():
        _ = model(imgs.to(device))     # full forward; populates the hook
    h.remove()
    return feat_holder["f"]            # [B, ch, H, W]


def _cluster_feature_to_labels(feat, n_clusters, out_hw):
    """eval_obj.py-style readout: L2-normalize the [B,ch,H,W] feature over channels, then cluster
    each image's H*W token vectors into n_clusters with agglomerative clustering on cosine
    distance, upsample the label map to out_hw (image resolution) with nearest interpolation.
    Returns int label maps [B, out_h, out_w]. Done on CPU to stay OOM-safe."""
    from sklearn.cluster import AgglomerativeClustering
    feat = F.normalize(feat, dim=1)                       # F.normalize over channels (eval_obj.py)
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C).cpu().numpy()
    labels = np.zeros((B, H, W), dtype=np.int64)
    for b in range(B):
        ncl = min(n_clusters, H * W)
        cl = AgglomerativeClustering(n_clusters=ncl, metric="cosine", linkage="average")
        lab = cl.fit_predict(feat[b])
        labels[b] = lab.reshape(H, W)
    # upsample to image resolution (nearest) so it lines up with GT masks
    lab_t = torch.from_numpy(labels).float()[:, None]
    lab_up = F.interpolate(lab_t, size=out_hw, mode="nearest")[:, 0].long().numpy()
    return lab_up


def eval_fgari_mbo(model, imgs, gt_labels, n_clusters, device):
    """Compute FG-ARI (mean) and MBO on a probe batch using the readout+clustering above and the
    repo's own metric functions (source/evals/objs/fgari.py, .../mbo.py). gt_labels: [B,H,W] int."""
    from source.evals.objs.fgari import calc_fgari_score
    from source.evals.objs.mbo import calc_mean_best_overlap

    out_hw = (gt_labels.shape[1], gt_labels.shape[2])
    feat = _capture_readout_feature(model, imgs, device)
    pred = _cluster_feature_to_labels(feat, n_clusters, out_hw)
    aris = calc_fgari_score(gt_labels, pred)               # list, one per image (fgari.py:18-27)
    fgari = float(np.mean(aris)) if len(aris) else float("nan")
    mbo_mean, _ = calc_mean_best_overlap(gt_labels, pred)  # (mbo.py:67-89)
    return dict(fgari=fgari, mbo=float(mbo_mean)), pred


# --------------------------------------------------------------------------------------
# J-zero counterfactual: forward-hook each KLayer.connectivity to force output -> 0.
# A forward hook that RETURNS a value replaces the module output, so connectivity(x)->0 =>
# _y=0 (klayer.py:128) => g_J=0, with every other trained weight byte-identical.
# --------------------------------------------------------------------------------------
class JZero:
    def __init__(self, model):
        self.model = model
        self.handles = []

    def __enter__(self):
        for l in range(self.model.L):
            conn = self.model.layers[l][0].connectivity
            self.handles.append(
                conn.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
            )
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------------------
# probe-batch loaders for the native eval datasets (GT masks at image resolution)
# --------------------------------------------------------------------------------------
def load_probe_batch(args):
    """Return (imgs [N,3,H,W] in [0,1], gt_labels [N,H,W] int, imsize, meta).
    Uses the repo's own eval loaders (load_data is_eval=True) so images/masks match training."""
    from source.data.datasets.objs.load_data import load_data

    dataset, imsize, collate_fn = load_data(
        args.data, args.data_root, args.data_imsize, is_eval=True
    )

    imgs, gts = [], []
    if args.data == "tetrominoes":
        # NumpyDataset: __getitem__ -> (img_tensor[3,H,W], {"pixelwise_instance_labels": [H,W]})
        for i in range(min(args.n_images, len(dataset))):
            img, lab = dataset[i]
            imgs.append(img)
            gts.append(np.asarray(lab["pixelwise_instance_labels"]).astype(np.int64))
        imgs = torch.stack(imgs)                                  # [N,3,H,W] in [0,1]
        gt = np.stack(gts)                                        # [N,H,W]
    elif args.data.startswith("clevrtex"):
        # CLEVRTEX eval: __getitem__ -> (ind, img[3,H,W], msk[1,H,W], meta?)  (clevr_tex.py:172-212)
        for i in range(min(args.n_images, len(dataset))):
            item = dataset[i]
            img = item[1]
            msk = item[2]                                         # [1,H,W] flat instance ids
            imgs.append(img)
            gts.append(np.asarray(msk)[0].astype(np.int64))
        imgs = torch.stack(imgs)
        gt = np.stack(gts)
    else:
        raise NotImplementedError(
            f"--data {args.data}: only tetrominoes / clevrtex_* wired for this probe"
        )
    return imgs.float(), gt, imsize, dict(n=len(imgs))


# --------------------------------------------------------------------------------------
def summarize_decomp(rec):
    """Per-layer per-step trajectories + scalar summaries for the prediction rule."""
    out = {}
    for l, steps in rec.items():
        if not steps:
            continue
        def col(k):
            return [round(s[k], 5) for s in steps]
        ratios = np.array([s["ratio_gJ_gc"] for s in steps])
        cmJ = np.array([s["common_mode_frac_gJ"] for s in steps])
        Rs = np.array([s["R_global_pre"] for s in steps])
        out[f"L{l}"] = dict(
            gJ_norm=col("gJ_norm"),
            gc_norm=col("gc_norm"),
            ratio_gJ_gc=col("ratio_gJ_gc"),
            cos_gJ_gc=col("cos_gJ_gc"),
            common_mode_frac_gJ=col("common_mode_frac_gJ"),
            common_mode_frac_gc=col("common_mode_frac_gc"),
            R_global_per_step=col("R_global_pre"),
            # scalar summaries
            ratio_gJ_gc_mean=float(ratios.mean()),
            ratio_gJ_gc_max=float(ratios.max()),
            common_mode_frac_gJ_mean=float(cmJ.mean()),
            R_global_first=float(Rs[0]),
            R_global_last=float(Rs[-1]),
            R_global_rise=float(Rs[-1] - Rs[0]),
        )
    return out


def evaluate_prediction(decomp_summary, full_metrics, jzero_metrics, final_R_full, final_R_jzero,
                        cm_thresh=0.85, ratio_thresh=1.5, fgari_tol=3.0):
    """Apply the PRE-REGISTERED rule. Returns the verdict dict (prediction: 'severance_inert' /
    'severance_harmful' / 'ambiguous') and the per-clause booleans, using the LAST native KLayer
    (the one whose readout feeds the metric)."""
    last_key = sorted(decomp_summary.keys())[-1] if decomp_summary else None
    s = decomp_summary.get(last_key, {}) if last_key else {}

    ratio_dominant = s.get("ratio_gJ_gc_mean", 0.0) >= ratio_thresh
    common_mode = s.get("common_mode_frac_gJ_mean", 0.0) >= cm_thresh
    raises_sync = (s.get("R_global_rise", 0.0) > 0.0) or (
        final_R_full.get(last_key, 0.0) > final_R_jzero.get(last_key, 0.0)
    )
    # metrics_full / metrics_jzero are RAW ARI in [-1,1] (fgari.py returns adjusted_rand_score);
    # fgari_tol is given in POINTS (paper-scale, x100), so convert to the raw scale for comparison.
    dfgari = full_metrics["fgari"] - jzero_metrics["fgari"]
    dmbo = full_metrics["mbo"] - jzero_metrics["mbo"]
    fgari_unchanged = abs(dfgari) <= (fgari_tol / 100.0)

    clauses = dict(
        a_gJ_dominates=bool(ratio_dominant),
        b_gJ_common_mode=bool(common_mode),
        c_raises_R_global=bool(raises_sync),
        d_fgari_unchanged=bool(fgari_unchanged),
        delta_fgari_full_minus_jzero=round(float(dfgari), 5),
        delta_mbo_full_minus_jzero=round(float(dmbo), 5),
    )
    if ratio_dominant and common_mode and raises_sync and fgari_unchanged:
        prediction = "severance_inert"
    elif (not fgari_unchanged) and (not common_mode):
        prediction = "severance_harmful"
    else:
        prediction = "ambiguous"
    return dict(prediction=prediction, clauses=clauses, decomp_layer_used=last_key,
                thresholds=dict(cm_thresh=cm_thresh, ratio_thresh=ratio_thresh,
                                fgari_tol=fgari_tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained objs checkpoint (EMA ema_model.pth etc.)")
    ap.add_argument("--src", default="/tmp/akorn_src", help="AKOrN source root to import from")
    ap.add_argument("--data", default="tetrominoes",
                    help="tetrominoes (smoke) | clevrtex_full (decisive) | clevrtex_camo | clevrtex_outd")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--data_imsize", type=int, default=None)
    ap.add_argument("--n_images", type=int, default=32, help="probe-set size (OOM-safe small)")
    ap.add_argument("--bs", type=int, default=8, help="forward batch size (OOM-safe small)")
    ap.add_argument("--n_clusters", type=int, default=None,
                    help="clusters for readout; default 6 (tetro) / 11 (clevrtex)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="experiments/gateA/native_decompose.json")

    # model hyperparams (must match the checkpoint's training config; defaults = AKOrN^attn objs)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--J", type=str, default="attn")
    ap.add_argument("--norm_ablate", type=str, default="none",
                    help="A3: clamp|soft|layernorm sphere-norm ablation (must match the checkpoint)")
    ap.add_argument("--phase_noise", type=float, default=0.0,
                    help="trained-desync: per-step phase noise sigma (match the checkpoint) for R_global under noise")
    ap.add_argument("--psize", type=int, default=8)
    ap.add_argument("--ksize", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", type=str, default="gn")
    ap.add_argument("--use_omega", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--global_omg", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--init_omg", type=float, default=0.01)
    ap.add_argument("--learn_omg", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--maxpool", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--project", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--use_ro_x", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--no_ro", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--gta", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--model_imsize", type=int, default=None)
    ap.add_argument("--autorescale", type=lambda s: s.lower() == "true", default=False)
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    device = args.device if torch.cuda.is_available() else "cpu"

    # default cluster counts per dataset (tetro: <=3 shapes + bg; clevrtex: up to ~10 objs + bg)
    if args.n_clusters is None:
        args.n_clusters = 6 if args.data == "tetrominoes" else 11

    t0 = time.time()
    imgs, gt, imsize, meta = load_probe_batch(args)
    net = build_net(args, imsize)
    model, load_info = load_checkpoint(net, args.ckpt, device)

    # ---- batched probe (OOM-safe): accumulate decomposition records + metric inputs ----
    from source.layers.kutils import reshape
    all_rec = {l: [] for l in range(model.L)}
    final_R_full_accum = {f"L{l}": [] for l in range(model.L)}
    final_R_jzero_accum = {f"L{l}": [] for l in range(model.L)}
    full_pred_chunks, jzero_pred_chunks = [], []

    n = imgs.shape[0]
    for start in range(0, n, args.bs):
        ib = imgs[start:start + args.bs]
        gtb = gt[start:start + args.bs]

        # (1) instrumented FULL forward -> decomposition + per-layer final R_global
        rec, final_R = run_decomposition(model, ib, device)
        for l in range(model.L):
            all_rec[l].extend(rec[l])
            final_R_full_accum[f"L{l}"].append(final_R[f"L{l}"])

        # (2) FULL readout metrics
        full_b, full_pred = eval_fgari_mbo(model, ib, gtb, args.n_clusters, device)
        full_pred_chunks.append((full_b, len(ib)))

        # (3) J-ZERO readout metrics + final R_global under J-zero (single forward reused for both)
        with JZero(model):
            jz_b, jz_pred = eval_fgari_mbo(model, ib, gtb, args.n_clusters, device)
            with torch.no_grad():
                _c, _x, xs_z, _e = model.feature(ib.to(device))
            for l in range(model.L):
                final_R_jzero_accum[f"L{l}"].append(
                    _order_param(reshape(xs_z[l + 1][-1], model.n))
                )
        jzero_pred_chunks.append((jz_b, len(ib)))

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate metrics (weighted by chunk size) ----
    def agg(chunks, key):
        num = sum(b[key] * w for b, w in chunks)
        den = sum(w for _, w in chunks)
        return num / max(den, 1)

    full_metrics = dict(fgari=agg(full_pred_chunks, "fgari"), mbo=agg(full_pred_chunks, "mbo"))
    jzero_metrics = dict(fgari=agg(jzero_pred_chunks, "fgari"), mbo=agg(jzero_pred_chunks, "mbo"))
    final_R_full = {k: float(np.mean(v)) for k, v in final_R_full_accum.items()}
    final_R_jzero = {k: float(np.mean(v)) for k, v in final_R_jzero_accum.items()}

    decomp_summary = summarize_decomp(all_rec)
    verdict = evaluate_prediction(decomp_summary, full_metrics, jzero_metrics,
                                  final_R_full, final_R_jzero)

    result = dict(
        meta=dict(
            ckpt=args.ckpt, data=args.data, n_images=int(n), bs=args.bs,
            n_clusters=args.n_clusters, imsize=imsize, device=device,
            J=args.J, N=args.N, ch=args.ch, L=args.L, T=args.T, psize=args.psize,
            elapsed_s=round(time.time() - t0, 1),
            load_info=load_info,
        ),
        decomposition=decomp_summary,
        R_global_final_full=final_R_full,
        R_global_final_jzero=final_R_jzero,
        metrics_full=full_metrics,
        metrics_jzero=jzero_metrics,
        prediction=verdict,
        prereg=(
            "PREDICT severance INERT iff: (a) ratio_gJ_gc_mean>=ratio_thresh AND "
            "(b) common_mode_frac_gJ_mean>=cm_thresh AND (c) R_global rises / exceeds J-zero AND "
            "(d) |FG-ARI(full)-FG-ARI(Jzero)|<=fgari_tol pts. Falsifier: FG-ARI collapses under "
            "J-zero while g_J is NOT common-mode => coupling carries binding."
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(dict(
        prediction=verdict["prediction"],
        clauses=verdict["clauses"],
        metrics_full=full_metrics,
        metrics_jzero=jzero_metrics,
        R_global_full=final_R_full,
        R_global_jzero=final_R_jzero,
        out=args.out,
    ), indent=2, default=str))
    print("NATIVE_DECOMPOSE_DONE")


if __name__ == "__main__":
    main()
