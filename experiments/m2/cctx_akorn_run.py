"""M2 HEADLINE DRIVER — true CCC C_ctx of an AKOrN oscillatory workspace, capture-then-freeze.

This is the headline run the M2 pre-registration (experiments/m2/preregistration-M2.md, ratified
2026-06-03) builds toward: measure the TRUE Context Channel Capacity

    C_ctx = I( context c ; GENERATED PARAMETERS theta(c) )      [BITS]   (CCC arXiv 2603.07415, Def 5/Thm 4)

with the AKOrN oscillator PHASE-STATE routed as the context c into a minimal phase-conditioned
theta-generator (the von-Oswald-seed PhaseContextThetaGen, already built + Split-MNIST-VALIDATED:
conditioned 2.09 bits vs agnostic 0.0). We contrast, at MATCHED generator params/dim:

    ON     = R6 full-AKOrN phase-state as context              (synchrony ON)
    OFF(a) = R5:no_proj phase-state as context                 (within-architecture synchrony flip)
    OFF(b) = a matched-dim, NON-oscillatory RATE-coded context (readout-block activation, not phase)

PRIMARY metric = mi_lower_bits (the chance-corrected CV decodability MI lower bound); eff_dim is the
COMPANION only (the Split-MNIST validation showed eff_dim FALSELY inflates on a task-agnostic generator
that jitters theta — so the greenlight rides on mi_lower_bits, NOT eff_dim).

================================  THE DESIGN DECISION (ratified)  ================================
CAPTURE-THEN-FREEZE. Train the AKOrN backbone on Split-CIFAR-100 (naive sequential — its CL accuracy
is NOT the M2 metric; we only need a trained oscillator to read phase-state off of), then FREEZE it and
capture, per task, (i) the oscillator phase-state as the per-sample context, and (ii) the frozen
readout features as the head input. Only THEN train the phase-conditioned theta-generator on those
frozen contexts (CO-training the backbone + generator is M3, explicitly out of scope here). This makes
the context a FIXED signal whose channel to theta we measure cleanly.

================================  THE CHANNEL WE MEASURE  ========================================
For each captured sample i (belonging to task t_i): context c_i (2n-dim pooled phase descriptor) and
frozen features f_i (the backbone's pooled readout, the same vector model.net.out classifies). The
generator g produces a PER-SAMPLE head theta_i = g(c_i); that head scores f_i. We train g by
cross-entropy on the task labels y_i (the 100-way CIFAR class, restricted to each task's 10 classes).
The ONLY context->parameter pathway is g; f is a fixed deterministic readout. So I(c; theta(c)) is the
genuine CCC quantity. C_ctx = compute_C_ctx over {task t -> [theta(c) for c in task t]} -> mi_lower_bits.

================================  CRITICAL CORRECTNESS (review-checked)  =========================
* CONTEXT/LABEL/FEATURE ALIGNMENT. Capture runs with shuffle=False; for each sample we store the
  triple (context, feature, task_label) AS ONE ROW and never reorder one tensor without the others.
  A misalignment silently destroys the measurement, so the row index is the single source of truth
  everywhere (generator training, C_ctx, wrong-context probing).
* NO LEAKAGE. The C_ctx decodability probe decodes TASK-ID from generated theta. Task-id must reach
  theta ONLY through the phase-state context. The rate-coded OFF(b) context is a REAL non-oscillatory
  feature map (the readout-block activation), NOT a relabeled phase one — built by the SAME pooling so
  it is dimension-matched but carries rate, not phase/synchrony, information.
* mi_lower_bits is the headline; eff_dim reported as companion only.
* OOM control: the per-TASK probe is tiny (capped via probe_per_class, ~probe_per_class*n_classes rows),
  so the full per-task (N,C,H,W) capture is small by construction. (build_context_from_phase still pools
  per-sample, but the actual OOM guard HERE is the small probe cap, not per-sample streaming.)

The --demo path exercises this ORCHESTRATION (per-task contexts -> dummy numpy theta-gen -> C_ctx ->
ON-vs-OFF comparison shape) on synthetic data with NO torch/GPU, asserting a separable (ON-style)
context yields higher C_ctx than a rate/noise (OFF-style) context.

References:
  - CCC, arXiv:2603.07415, Def 5 / Thm 4 (C_ctx = I(c; theta(c)); hypernetworks attain C_ctx>>0).
  - von Oswald et al. (2020), Continual learning with hypernetworks, ICLR (the g: c->theta backbone).
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np

# Resolve the M1 helpers RELATIVE to __file__ (a hard-coded absolute path already broke the GPU box).
_HERE = os.path.dirname(os.path.abspath(__file__))
_M1 = os.path.normpath(os.path.join(_HERE, "..", "m1_wk0"))
for _p in (_HERE, _M1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ctx_channel_capacity as ccc                       # compute_C_ctx (mi_lower_bits headline)
import theta_generator as tg                             # PhaseContextThetaGen, context builders, inject_fn
import m2_primitives as m2                               # wrong_context_probe, P5/P5b/P7

# torch + the real model are needed ONLY for run_arm/run_headline (the live GPU path). --demo and the
# orchestration helpers are pure-numpy so this file py_compiles + CPU-tests on the torchless box.
try:
    import torch
except Exception:  # pragma: no cover - torchless analysis box
    torch = None


# =====================================================================================
# 0. ORCHESTRATION PRIMITIVES  (pure-numpy; the --demo exercises exactly these)
# =====================================================================================
def context_by_class_from_rows(contexts, task_ids):
    """Group per-sample contexts into the {task -> [context rows]} dict compute_C_ctx/estimate_c_ctx want.

    contexts : (N, context_dim) per-sample context vectors, row i belongs to sample i.
    task_ids : (N,)          the context-CLASS (task id) of each row — the C_ctx supervision label.
    Returns dict {t: list of context vectors}. Row order within a task is preserved (alignment-safe);
    this only PARTITIONS rows, it never reorders a context away from its label.
    """
    C = np.asarray(contexts, dtype=np.float64)
    t = np.asarray(task_ids).reshape(-1)
    assert C.shape[0] == t.shape[0], f"contexts {C.shape[0]} vs task_ids {t.shape[0]} misaligned"
    out = {}
    for i in range(C.shape[0]):
        out.setdefault(int(t[i]), []).append(C[i])
    return out


def c_ctx_from_theta_rows(theta_rows, task_ids, n_splits=5, seed=0, n_shuffle=10):
    """compute_C_ctx on already-generated per-sample theta rows + their task labels (alignment-safe).

    theta_rows : (N, flat_dim) generated head-parameter vectors, row i is theta(c_i).
    task_ids   : (N,)          task/context-CLASS of each row.
    Returns the compute_C_ctx dict (mi_lower_bits PRIMARY + eff_dim companion + diagnostics).
    """
    theta = np.asarray(theta_rows, dtype=np.float64)
    y = np.asarray(task_ids).reshape(-1)
    assert theta.shape[0] == y.shape[0], f"theta {theta.shape[0]} vs labels {y.shape[0]} misaligned"
    return ccc.compute_C_ctx((theta, y), n_splits=n_splits, seed=seed, n_shuffle=n_shuffle)


def summarize_arm(arm, context_kind, c_ctx, wrong_ctx_deltas, gen_train_acc, n_tasks):
    """Assemble one arm's result record (the per-arm payload of run_headline's JSON)."""
    return {
        "arm": arm,
        "context_kind": context_kind,
        "C_ctx_bits": float(c_ctx["mi_lower_bits"]),          # PRIMARY headline
        "eff_dim_bits": float(c_ctx.get("eff_dim_bits", float("nan"))),  # COMPANION only
        "Hmax_bits": float(c_ctx.get("Hmax_bits", float("nan"))),
        "raw_mi_bits": float(c_ctx.get("raw_mi_bits", float("nan"))),
        "chance_floor_bits": float(c_ctx.get("chance_floor_bits", float("nan"))),
        "cv_accuracy": float(c_ctx.get("cv_accuracy", float("nan"))),
        "n_classes": int(c_ctx.get("n_classes", 0)),
        "n": int(c_ctx.get("n", 0)),
        "gen_train_acc": float(gen_train_acc),
        "wrong_ctx_deltas": {k: float(v) for k, v in (wrong_ctx_deltas or {}).items()},
        "n_tasks": int(n_tasks),
    }


def compare_on_vs_off(arm_results):
    """ON-vs-OFF comparison block from a {arm_name -> arm_record} mapping.

    Expects keys 'ON', 'OFF_a', 'OFF_b' (any subset present is fine). Reports the C_ctx (mi_lower_bits)
    GAP of ON over each OFF control and whether ON exceeds BOTH (the pre-registered greenlight shape;
    the across-seed paired test + nuisance-partialling are applied downstream on the per-seed records).
    Also surfaces the P5 (wrong-task) accuracy delta of the ON arm — must be < 0 for a real channel.
    """
    def cval(k):
        return float(arm_results[k]["C_ctx_bits"]) if k in arm_results else float("nan")

    on = cval("ON")
    off_a = cval("OFF_a")
    off_b = cval("OFF_b")
    gaps = {}
    if "OFF_a" in arm_results:
        gaps["ON_minus_OFF_a"] = on - off_a
    if "OFF_b" in arm_results:
        gaps["ON_minus_OFF_b"] = on - off_b
    on_beats_both = bool(
        ("OFF_a" not in arm_results or on > off_a) and
        ("OFF_b" not in arm_results or on > off_b) and
        np.isfinite(on)
    )
    p5_on = float("nan")
    if "ON" in arm_results:
        p5_on = float(arm_results["ON"].get("wrong_ctx_deltas", {}).get("P5", float("nan")))
    return {
        "C_ctx_ON_bits": on,
        "C_ctx_OFF_a_bits": off_a,
        "C_ctx_OFF_b_bits": off_b,
        "gaps_bits": gaps,
        "ON_beats_both_OFF": on_beats_both,
        "delta_P5_ON": p5_on,                      # must be < 0 for a real channel (falsifier)
    }


# =====================================================================================
# 1. LIVE CAPTURE  (torch; reuse the M1 backbone + probe machinery, do NOT modify M1 files)
# =====================================================================================
def _readout_block(model, layer):
    """The per-layer readout block model.net.layers[l][3] — the NON-oscillatory (rate-coded) feature
    map tap for OFF(b). Same block h3._readout_blocks taps; we register our own hook so no M1 edit."""
    return model.net.layers[layer][3]


def _capture_triples(model, probe_loader, layer, device, eval_inits, base_seed):
    """Capture, row-aligned over the FIXED shuffle=False probe loader, the per-sample triple needed by
    every downstream step:

      phase_osc : (N, C, H, W)  oscillator phase-state xs[layer][-1]   (ON / OFF(a) context source)
      rate_map  : (N, C, H, W)  readout-block activation at `layer`     (OFF(b) rate-coded source)
      feats     : (N, F)        FINAL pooled readout `c` (== model.net.out's input; head feature)
      labels    : (N,)          the CIFAR class label of each sample     (generator-training target)

    All four are averaged over the eval_inits fixed-seed forwards (the LadderClassifier eval scheme),
    so AKOrN oscillator-init noise is integrated out and matched across arms. Row order == probe order
    (shuffle=False) so contexts/features/labels stay paired by index. OOM note: we hold the raw
    (N,C,H,W) phase/rate snapshots only to pool them per-sample immediately after (build_*_context);
    the caller pools and discards.
    """
    from h3 import _seeded as h3_seeded
    block = _readout_block(model, layer)
    rate_buf = {}

    def rate_hook(_m, _i, out):
        rate_buf.setdefault("v", []).append(out.detach().float().cpu())

    was_training = model.training
    model.eval()
    osc_acc = rate_acc = feat_acc = None
    labels = None
    handle = block.register_forward_hook(rate_hook)
    try:
        for j in range(eval_inits):
            rate_buf["v"] = []
            osc_j, feat_j, lab_j = [], [], []
            with h3_seeded(base_seed + j), torch.no_grad():
                for batch in probe_loader:
                    xb = batch[0].to(device)
                    yb = batch[1]
                    c, _x, xs, _es = model.net.feature(xb)        # c: (B,F) pooled readout = head input
                    osc_j.append(xs[layer][-1].detach().float().cpu())   # (B,C,H,W) phase-state
                    feat_j.append(c.detach().float().cpu())              # (B,F) frozen head feature
                    if j == 0:
                        lab_j.append(np.asarray(yb).reshape(-1))
            osc_cat = torch.cat(osc_j, 0)
            feat_cat = torch.cat(feat_j, 0)
            rate_cat = torch.cat(rate_buf["v"], 0)                # (N,C,H,W) readout activation
            osc_acc = osc_cat if osc_acc is None else osc_acc + osc_cat
            rate_acc = rate_cat if rate_acc is None else rate_acc + rate_cat
            feat_acc = feat_cat if feat_acc is None else feat_acc + feat_cat
            if j == 0:
                labels = np.concatenate(lab_j)
        f = float(eval_inits)
        return (
            (osc_acc / f).numpy(),
            (rate_acc / f).numpy(),
            (feat_acc / f).numpy(),
            np.asarray(labels),
        )
    finally:
        handle.remove()
        if was_training:
            model.train()


def _train_backbone(rung, variant, n_tasks, epochs, lr, seed, eval_inits, device):
    """Train a LadderClassifier(rung) on Split-CIFAR-100 (naive sequential) and return (model, bench).

    Same training-loop SHAPE as avalanche_backbone.run_split_cifar100 (Naive strategy, Adam, the same
    keyword-only ctor contract), but we keep ONLY the trained model + benchmark — the CL accuracy is
    not the M2 metric (capture-then-freeze reads phase-state off the trained oscillator afterwards)."""
    import torch.nn as nn
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import InteractiveLogger
    from avalanche_backbone import LadderClassifier

    rung_kw = {"variant": variant} if variant else {}
    bench = SplitCIFAR100(n_experiences=n_tasks, return_task_id=False, seed=seed)
    model = LadderClassifier(rung, num_classes=100, eval_inits=eval_inits,
                             base_seed=seed, **rung_kw).to(device)
    evalp = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[InteractiveLogger()])
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    cl = Naive(model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
               train_mb_size=128, eval_mb_size=100, train_epochs=epochs,
               evaluator=evalp, device=device)
    for exp in bench.train_stream:
        cl.train(exp)
    return model, bench


def _train_generator(contexts, feats, labels, task_ids, num_classes, n_tasks,
                     hidden=64, epochs=200, lr=1e-3, seed=0, device="cpu"):
    """Train PhaseContextThetaGen on the FROZEN (context, feature, label) rows (capture-then-freeze).

    contexts:(N,Dc) feats:(N,F) labels:(N,) task_ids:(N,) — all ROW-ALIGNED (index i is one sample).
    For each sample i, theta_i = g(c_i) parameterizes a linear head applied to its frozen features f_i;
    cross-entropy on the CIFAR class label y_i. We do NOT shuffle the rows out of their (c,f,y) pairing:
    minibatches index the SAME positions in all three arrays. Returns (generator, train_acc).

    The generated head is the full 100-way head (num_classes), but each task only exposes its own 10
    labels; that is fine — the channel we measure is task-context -> theta, and per-task theta still
    differs by the task's class subset. (Restricting to per-task heads would import task identity into
    the loss target; keeping the global head keeps task-id flowing ONLY through the context.)
    """
    import torch.nn as nn
    Dc = int(np.asarray(contexts).shape[1])
    F = int(np.asarray(feats).shape[1])
    gen = tg.PhaseContextThetaGen(context_dim=Dc, feat_dim=F, num_classes=num_classes,
                                  hidden=hidden, seed=seed).to(device)
    C = torch.as_tensor(np.asarray(contexts, dtype=np.float32), device=device)
    Xf = torch.as_tensor(np.asarray(feats, dtype=np.float32), device=device)
    Y = torch.as_tensor(np.asarray(labels).reshape(-1).astype(np.int64), device=device)
    N = C.shape[0]
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g_rng = torch.Generator().manual_seed(seed)
    bs = min(256, N)
    gen.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g_rng)        # permute INDICES; (c,f,y) move together
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            out = gen(C[idx])                            # per-sample heads from per-sample contexts
            theta_flat = out["theta_flat"]
            logits = tg.apply_generated_head(Xf[idx], theta_flat, gen.feat_dim,
                                             gen.num_classes, bias=gen.bias)
            opt.zero_grad()
            loss = lossf(logits, Y[idx])
            loss.backward()
            opt.step()
    # train accuracy (eval, per-sample heads, aligned rows)
    gen.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, N, bs):
            idx = torch.arange(s, min(s + bs, N))
            out = gen(C[idx])
            logits = tg.apply_generated_head(Xf[idx], out["theta_flat"], gen.feat_dim,
                                             gen.num_classes, bias=gen.bias)
            correct += int((logits.argmax(1).cpu().numpy() == Y[idx].cpu().numpy()).sum())
    return gen, float(correct) / float(N)


def _generated_theta_rows(gen, contexts, device="cpu"):
    """Run the trained generator over every (row-aligned) context -> (N, flat_dim) theta matrix."""
    C = torch.as_tensor(np.asarray(contexts, dtype=np.float32), device=device)
    gen.eval()
    rows = []
    with torch.no_grad():
        for s in range(0, C.shape[0], 512):
            out = gen(C[s:s + 512])
            rows.append(out["theta_flat"].detach().cpu().numpy())
    return np.concatenate(rows, 0)


# =====================================================================================
# 2. WRONG-CONTEXT PROBING  (the falsifier — ΔP5/P5b/P7 must be < 0 for a real channel)
# =====================================================================================
def _wrong_context_deltas(gen, contexts, feats, labels, task_ids, device="cpu"):
    """Run P5 (wrong-task), P5b (random), P7 (zero) wrong-context probes (the prereg falsifier).

    CAPTURE-THEN-FREEZE SUBTLETY: there is NO standalone model forward — the CORRECT-context baseline
    is itself an inject_fn call with the real per-sample context. m2_primitives.wrong_context_probe
    computes its baseline as `model(x)` (it assumes the model's OWN correct context), which does not
    fit here, so we compute BOTH accuracies directly with m2._eval_accuracy through the SAME inject_fn
    (correct per-sample context vs each wrong context) and form the delta ourselves. We reuse the M2
    machinery for everything else: make_inject_fn (the context->theta->prediction channel), the P5
    roll (roll_context_by_task), the P5b random draw (random_context_like), and the P7 None=zero
    sentinel that make_inject_fn already interprets.

    Builds a tiny in-memory probe over the FROZEN (feature, label) rows: x is a batch of ROW INDICES
    into the frozen feature matrix, so the inject_fn pulls those rows' features and scores them with
    the head generated from the supplied context. Row order matches contexts/labels (alignment-safe).

    Returns {'P5','P5b','P7'} -> signed accuracy delta (acc_wrong - acc_correct); NEGATIVE = the wrong
    context HURT => a real channel. ~0 => the context is inert / bypassed (the channel is not real).
    """
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    contexts = np.asarray(contexts, dtype=np.float32)
    task_ids = np.asarray(task_ids).reshape(-1)

    # FIX (2026-06-03): (a) DEVICE — the inject_fn path feeds numpy contexts (CPU) but `gen` trained on
    # cuda -> device-mismatch crash; move the tiny generator to CPU for this cheap eval. (b) GRAD — the
    # generator output requires_grad, and _eval_accuracy calls .numpy() on it; put gen in eval() and run
    # the whole probe under torch.no_grad() so nothing tracks grad. Both are eval-only, no science impact.
    _nograd = contextlib.nullcontext()
    try:
        gen = gen.to("cpu")
        gen.eval()
        if torch is not None:
            _nograd = torch.no_grad()
    except Exception:
        pass

    # feature_extractor for make_inject_fn: the "model" is unused; x is ROW INDICES into the frozen
    # feature matrix, so inject_fn scores exactly those rows. No live forward needed.
    def feature_extractor(_model, x):
        idx = np.asarray(x).reshape(-1).astype(int)
        return feats[idx]

    inject_fn = tg.make_inject_fn(gen, feature_extractor)

    # ONE batch over all rows (sizes already capped by probe_per_class). yields (x=indices, y, task).
    N = feats.shape[0]
    all_idx = np.arange(N)
    probe_loader = [(all_idx, labels, task_ids)]

    out = {}
    with _nograd:
        # The CORRECT-context baseline IS an injected forward with the real per-sample context.
        # device="cpu": gen moved to CPU above; the inject path is numpy/CPU (x = row indices).
        acc_correct, _ = m2._eval_accuracy(None, probe_loader, device="cpu",
                                           inject_fn=inject_fn, context=contexts)

        wrong_ctx = {
            "P5": tg.roll_context_by_task(contexts, task_ids),   # bind each sample to the NEXT task's ctx
            "P5b": tg.random_context_like(contexts, seed=0),     # matched-scale random context (no task id)
            "P7": None,                                          # sentinel: make_inject_fn zeros the context
        }
        for name, wc in wrong_ctx.items():
            acc_wrong, _ = m2._eval_accuracy(None, probe_loader, device="cpu",
                                             inject_fn=inject_fn, context=wc)
            out[name] = float(acc_wrong) - float(acc_correct)   # < 0 => degraded (real channel)
    return out


# =====================================================================================
# 3. ONE ARM  (capture-then-freeze, one (rung, variant, context_kind) at a time)
# =====================================================================================
def run_arm(rung, variant, context_kind, n_tasks=5, epochs=400, gen_epochs=200,
            seed=0, device="cuda", layer=1, lr=1e-4, gen_lr=1e-3, hidden=64,
            eval_inits=8, probe_per_class=16, n_shuffle=10, feature_mode="trained"):
    """Run ONE M2 arm end-to-end (the live GPU path).

    feature_mode (the c->theta channel's HEAD-INPUT features):
      'trained'  -> the backbone's task-trained pooled readout (the ORIGINAL design). These already
                    solve the task, so the c->theta pathway is excused from carrying task bits => C_ctx~0
                    is the EXPECTED reading of a state-modifier-equivalent (the 2026-06-04 artifact dx).
      'agnostic' -> a FIXED random-projection of those features (task structure scrambled), so the
                    generated theta becomes the ONLY task-adaptive component -- the validated Split-MNIST
                    condition. Disambiguates artifact (C_ctx jumps toward Hmax) vs true null (stays ~0).

    rung/variant   : backbone selector. ON=('R6',None); OFF(a)=('R5','no_proj'); OFF(b)=('R6',None) too,
                     but with context_kind='rate'.
    context_kind   : 'phase' (ON/OFF(a)) -> build_context_from_phase on xs[layer][-1];
                     'rate'  (OFF(b))     -> build_rate_coded_context on the readout-block activation.
    layer          : the AKOrN layer whose phase-state is the context (prereg: route layer 1-2; default 1).

    Steps (capture-then-freeze):
      1. train the backbone on Split-CIFAR-100 (naive),
      2. build ONE fixed class-balanced probe loader per task,
      3. per task: capture (phase_osc, rate_map, feats, labels), build the per-sample CONTEXT for the
         requested kind, tag every row with task id t,
      4. train the phase-conditioned theta-generator on the frozen (context, feature, label) rows,
      5. C_ctx = compute_C_ctx over {task -> theta(c)} -> mi_lower_bits (PRIMARY) + eff_dim (companion);
         + wrong-context probing (P5/P5b/P7) deltas.

    Returns the summarize_arm record.
    """
    if torch is None:
        raise RuntimeError("run_arm needs torch + the AKOrN model; use --demo on the torchless box.")

    model, bench = _train_backbone(rung, variant, n_tasks, epochs, lr, seed, eval_inits, device)
    n = int(getattr(model.net, "n", 4))

    # ONE fixed class-balanced probe loader per TASK (capped via probe_per_class), shuffle=False.
    per_task_loaders = _build_per_task_probe_loaders(bench, n_tasks, probe_per_class, device)

    all_ctx, all_feat, all_lab, all_task = [], [], [], []
    for t in range(n_tasks):
        loader = per_task_loaders[t]
        osc, rate, feats, labels = _capture_triples(
            model, loader, layer, device, eval_inits=model.eval_inits, base_seed=model.base_seed)
        if context_kind == "phase":
            ctx = tg.build_context_from_phase(osc, layer=0, n=n)        # (n_samples, 2n)
        elif context_kind == "rate":
            ctx = tg.build_rate_coded_context(rate, layer=0, n=n)       # matched-dim NON-oscillatory
        else:
            raise ValueError(f"unknown context_kind {context_kind!r}; expected 'phase'|'rate'")
        m_rows = ctx.shape[0]
        assert feats.shape[0] == m_rows == len(labels), "capture row misalignment (ctx/feat/label)"
        all_ctx.append(ctx)
        all_feat.append(np.asarray(feats, dtype=np.float64))
        all_lab.append(np.asarray(labels).reshape(-1))
        all_task.append(np.full(m_rows, t, dtype=np.int64))

    contexts = np.concatenate(all_ctx, 0)
    feats = np.concatenate(all_feat, 0)
    labels = np.concatenate(all_lab, 0)
    task_ids = np.concatenate(all_task, 0)

    # ARTIFACT-DX (2026-06-04): optionally scramble the head-input features with a FIXED random
    # projection so they no longer solve the task -> the generated theta must carry the task -> a fair
    # CCC test (matches the validated Split-MNIST condition where theta was the only adaptive component).
    if feature_mode == "agnostic":
        F = feats.shape[1]
        rng = np.random.default_rng(seed + 9973)
        # orthonormal-ish random projection F->F (preserves dim, destroys task-aligned axes), then a
        # fixed nonlinearity so it is not a trivially-invertible linear map.
        W = rng.standard_normal((F, F)) / np.sqrt(F)
        feats = np.tanh(feats @ W).astype(np.float64)
    elif feature_mode != "trained":
        raise ValueError(f"unknown feature_mode {feature_mode!r}; expected 'trained'|'agnostic'")

    gen, gen_acc = _train_generator(contexts, feats, labels, task_ids, num_classes=100,
                                    n_tasks=n_tasks, hidden=hidden, epochs=gen_epochs,
                                    lr=gen_lr, seed=seed, device=device)

    theta_rows = _generated_theta_rows(gen, contexts, device=device)
    c_ctx = c_ctx_from_theta_rows(theta_rows, task_ids, seed=seed, n_shuffle=n_shuffle)
    deltas = _wrong_context_deltas(gen, contexts, feats, labels, task_ids, device=device)

    return summarize_arm(f"{rung}:{variant or 'full'}", context_kind, c_ctx, deltas, gen_acc, n_tasks)


def _build_per_task_probe_loaders(bench, n_tasks, per_class, device, probe_seed=12345):
    """One fixed, class-balanced, shuffle=False probe DataLoader PER TASK from the test stream.

    Mirrors avalanche_backbone._build_probe_loader's class-balanced fixed-seed selection but scoped to
    each experience's own test set, so task t's context is captured from task t's classes only (the
    context-CLASS = task id is then unambiguous and carries NO cross-task leakage)."""
    from torch.utils.data import Subset, DataLoader
    loaders = {}
    for t, exp in enumerate(bench.test_stream):
        if t >= n_tasks:
            break
        ds = exp.dataset
        tgt = getattr(ds, "targets", None)
        labels = np.asarray(list(tgt) if tgt is not None
                            else [int(ds[i][1]) for i in range(len(ds))])
        rng = np.random.default_rng(probe_seed + t)
        idx = []
        for c in np.unique(labels):
            pool = np.flatnonzero(labels == c)
            take = min(per_class, len(pool))
            idx.extend(sorted(rng.choice(pool, size=take, replace=False).tolist()))
        loaders[t] = DataLoader(Subset(ds, sorted(idx)), batch_size=100, shuffle=False)
    return loaders


# =====================================================================================
# 4. HEADLINE  (ON vs OFF(a) vs OFF(b) across seeds -> results/cctx_akorn.json)
# =====================================================================================
def run_headline(seeds=(0,), n_tasks=5, epochs=400, gen_epochs=200, device="cuda",
                 layer=1, lr=1e-4, gen_lr=1e-3, hidden=64, eval_inits=8,
                 probe_per_class=16, n_shuffle=10, out_path=None):
    """Run the three arms across seeds and write results/cctx_akorn.json.

    ON     = (R6, full,     context_kind='phase')
    OFF_a  = (R5, no_proj,  context_kind='phase')
    OFF_b  = (R6, full,     context_kind='rate')

    JSON payload: per-arm per-seed records, per-arm mean C_ctx (mi_lower_bits), the ON-vs-OFF comparison
    block (gaps + ON_beats_both + ΔP5), and the run config. The across-seed paired test + nuisance-
    partialling that the decision rule needs are applied downstream on these per-seed records.
    """
    arm_spec = {
        "ON":    ("R6", None,      "phase"),
        "OFF_a": ("R5", "no_proj", "phase"),
        "OFF_b": ("R6", None,      "rate"),
    }
    per_seed = {name: [] for name in arm_spec}
    for seed in seeds:
        for name, (rung, variant, ckind) in arm_spec.items():
            rec = run_arm(rung, variant, ckind, n_tasks=n_tasks, epochs=epochs,
                          gen_epochs=gen_epochs, seed=seed, device=device, layer=layer,
                          lr=lr, gen_lr=gen_lr, hidden=hidden, eval_inits=eval_inits,
                          probe_per_class=probe_per_class, n_shuffle=n_shuffle)
            rec["seed"] = int(seed)
            per_seed[name].append(rec)

    # per-arm mean C_ctx + a representative (last-seed) record for the comparison shape.
    arm_mean = {name: float(np.mean([r["C_ctx_bits"] for r in recs])) if recs else float("nan")
                for name, recs in per_seed.items()}
    repr_arm = {name: recs[-1] for name, recs in per_seed.items() if recs}
    comparison = compare_on_vs_off(repr_arm)
    comparison["C_ctx_ON_bits_mean"] = arm_mean.get("ON", float("nan"))
    comparison["C_ctx_OFF_a_bits_mean"] = arm_mean.get("OFF_a", float("nan"))
    comparison["C_ctx_OFF_b_bits_mean"] = arm_mean.get("OFF_b", float("nan"))

    out = {
        "metric": "C_ctx = I(phase-context; generated-theta) in BITS (PRIMARY=mi_lower_bits)",
        "design": "capture-then-freeze; ON=R6:phase OFF_a=R5:no_proj:phase OFF_b=R6:rate",
        "config": {
            "seeds": list(seeds), "n_tasks": n_tasks, "epochs": epochs, "gen_epochs": gen_epochs,
            "layer": layer, "lr": lr, "gen_lr": gen_lr, "hidden": hidden, "eval_inits": eval_inits,
            "probe_per_class": probe_per_class, "n_shuffle": n_shuffle,
        },
        "per_seed": per_seed,
        "C_ctx_mean_bits": arm_mean,
        "comparison": comparison,
    }
    out_path = out_path or os.path.join(_HERE, "results", "cctx_akorn.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# =====================================================================================
# 5. --demo  (numpy-only orchestration check: NO torch / NO GPU)
# =====================================================================================
class _DummyNumpyThetaGen:
    """A torchless stand-in for the generator: a FIXED random linear map context -> theta_flat.

    forward(context) -> {'theta_flat': (B, flat_dim)} matching PhaseContextThetaGen's output contract
    (so estimate_c_ctx / the orchestration helpers consume it unchanged). Because the map is linear and
    deterministic, a context that SEPARATES tasks yields task-separable theta (high C_ctx) and a context
    that is task-agnostic noise yields task-agnostic theta (~0 C_ctx) — exactly the comparison shape the
    headline produces, without any training. This validates the DRIVER PLUMBING, not a real generator.
    """

    def __init__(self, context_dim, flat_dim=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((context_dim, flat_dim)) / np.sqrt(context_dim)
        self.b = rng.standard_normal(flat_dim) * 0.01
        self.context_dim = int(context_dim)
        self.feat_dim = 1
        self.num_classes = flat_dim
        self.bias = False

    def __call__(self, context):
        C = np.asarray(context, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return {"theta_flat": C @ self.W + self.b}


def _synth_contexts(n_tasks, per_task, ctx_dim, kind, seed=0):
    """Synthetic per-sample contexts tagged by task. kind='separable' (ON-style: task-specific mean,
    small jitter) vs kind='noise' (OFF/rate-style: shared mean, pure noise, no task identity)."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_tasks, ctx_dim)) * 3.0
    rows, tasks = [], []
    for t in range(n_tasks):
        for _ in range(per_task):
            if kind == "separable":
                rows.append(means[t] + rng.standard_normal(ctx_dim) * 0.3)
            else:                                   # task-agnostic noise (rate/OFF stand-in)
                rows.append(rng.standard_normal(ctx_dim) * 0.3)
            tasks.append(t)
    return np.asarray(rows), np.asarray(tasks)


def _demo():
    """Exercise the orchestration on synthetic contexts + a dummy numpy theta-gen; assert ON-style
    (separable) context gives higher C_ctx than a rate/noise context — the headline comparison shape."""
    n_tasks, per_task, ctx_dim = 5, 16, 8                # ctx_dim = 2n for n=4
    flat_dim = 6 * 2                                     # tiny generated head width

    ctx_on, task_on = _synth_contexts(n_tasks, per_task, ctx_dim, "separable", seed=1)
    ctx_off, task_off = _synth_contexts(n_tasks, per_task, ctx_dim, "noise", seed=2)

    # 1. context_by_class_from_rows partitions row-aligned contexts by task (alignment-safe).
    cbc_on = context_by_class_from_rows(ctx_on, task_on)
    assert set(cbc_on.keys()) == set(range(n_tasks))
    assert sum(len(v) for v in cbc_on.values()) == n_tasks * per_task

    gen = _DummyNumpyThetaGen(context_dim=ctx_dim, flat_dim=flat_dim, seed=0)

    # 2. generate theta(c) for every row (the live path uses _generated_theta_rows; same shape).
    theta_on = np.asarray([gen(c)["theta_flat"].ravel() for c in ctx_on])
    theta_off = np.asarray([gen(c)["theta_flat"].ravel() for c in ctx_off])
    assert theta_on.shape == (n_tasks * per_task, flat_dim)

    # 3. C_ctx via the row-aligned helper (PRIMARY = mi_lower_bits).
    C_on = c_ctx_from_theta_rows(theta_on, task_on, n_shuffle=6)
    C_off = c_ctx_from_theta_rows(theta_off, task_off, n_shuffle=6)

    # cross-check: estimate_c_ctx (the {class->context} API) agrees with the row-aligned path on ON.
    C_on_api = ccc.estimate_c_ctx(gen, cbc_on, n_shuffle=6)

    # 4. assemble the per-arm records + the ON-vs-OFF comparison block (the headline output shape).
    rec_on = summarize_arm("R6:full", "phase", C_on,
                           {"P5": -0.4, "P5b": -0.35, "P7": -0.5}, gen_train_acc=0.9, n_tasks=n_tasks)
    rec_off = summarize_arm("R6:full", "rate", C_off,
                            {"P5": 0.0, "P5b": 0.0, "P7": 0.0}, gen_train_acc=0.5, n_tasks=n_tasks)
    comparison = compare_on_vs_off({"ON": rec_on, "OFF_b": rec_off})

    print(f"[demo] C_ctx ON(separable) = {C_on['mi_lower_bits']:.3f} bits  "
          f"OFF(noise) = {C_off['mi_lower_bits']:.3f} bits  (Hmax={C_on['Hmax_bits']:.3f})")
    print(f"[demo] estimate_c_ctx API agrees on ON: {C_on_api['mi_lower_bits']:.3f} bits")
    print(f"[demo] comparison block: gap={comparison['gaps_bits']}  "
          f"ON_beats_both={comparison['ON_beats_both_OFF']}  ΔP5_ON={comparison['delta_P5_ON']}")

    # ASSERTIONS (the plumbing produces the right comparison shape):
    assert C_on["mi_lower_bits"] > C_off["mi_lower_bits"] + 0.5, \
        f"separable ON should exceed noise OFF by >0.5 bits ({C_on['mi_lower_bits']} vs {C_off['mi_lower_bits']})"
    assert C_off["mi_lower_bits"] < 0.3, f"noise OFF should be ~0 ({C_off['mi_lower_bits']})"
    assert comparison["ON_beats_both_OFF"] is True
    assert comparison["delta_P5_ON"] < 0, "ON arm P5 falsifier must be < 0 (channel degrades)"
    assert C_on_api["mi_lower_bits"] > C_off["mi_lower_bits"] + 0.5, "estimate_c_ctx API path must agree"
    print("=== CCTX-AKORN DEMO OK (orchestration produces the ON>OFF comparison shape) ===")


def main():
    ap = argparse.ArgumentParser(description="M2 headline driver: true CCC C_ctx of an AKOrN workspace.")
    ap.add_argument("--demo", action="store_true", help="numpy-only orchestration check (no torch/GPU).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400, help="backbone CL epochs/experience.")
    ap.add_argument("--gen-epochs", type=int, default=200, help="theta-generator training epochs.")
    ap.add_argument("--layer", type=int, default=1, help="AKOrN layer whose phase-state is the context.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--probe-per-class", type=int, default=16,
                    help="rows per task-class for the C_ctx decoder; higher relieves the p>>n regime (theta-dim "
                         "~25k decoded from ~probe_per_class*n_classes*n_tasks rows). 16 -> ~800 rows.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.demo:
        _demo()
        return

    out = run_headline(seeds=tuple(args.seeds), n_tasks=args.n_tasks, epochs=args.epochs,
                       gen_epochs=args.gen_epochs, device=args.device, layer=args.layer,
                       probe_per_class=args.probe_per_class, out_path=args.out)
    print(json.dumps(out["C_ctx_mean_bits"], indent=2, default=str))
    print(json.dumps(out["comparison"], indent=2, default=str))


if __name__ == "__main__":
    main()
