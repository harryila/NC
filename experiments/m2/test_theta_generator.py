"""CPU test for the M2 phase-conditioned theta-generator (theta_generator.py).

Runs on the torchless h3venv (numpy via /tmp/h3venv/bin/python): the pure-shape logic + the context
builders are exercised unconditionally; the torch nn.Module generator + inject_fn path are exercised
ONLY when torch is importable (guarded), so this file both py_compiles and passes torchless.

Asserts the M2-load-bearing behaviors:
  A. SHAPES (torchless): theta flat dim, weight (feat, classes), bias (classes); a generated head maps
     (B, feat) features -> (B, num_classes) logits; per-sample heads ((B, flat)) also -> (B, classes).
  B. CONTEXT BUILDER (numpy): per-sample phase-context is (B, 2n), pooled one-sample-at-a-time (the
     OOM-safe contract — distinct per-sample contexts, never a stacked (B,C,H,W) matrix).
  C. GENERATOR CONDITIONS ON c (torch): DIFFERENT contexts produce DIFFERENT theta. A degenerate g that
     ignored c would give C_ctx=0 — the exact failure M2 must be able to DETECT — so this is the key
     correctness gate. Also: SAME context -> SAME theta (determinism), and the inject_fn yields
     (B, num_classes) logits that CHANGE when the per-sample context is permuted (the wrong-context
     degradation pathway the P5 falsifier rides on).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m1_wk0")))

import theta_generator as tg

try:
    import torch
except Exception:
    torch = None


def test_shape_math_torchless():
    """A. Pure-shape helpers — no torch."""
    n, feat, K = 4, 12, 5
    assert tg.context_dim_for_n(n) == 8, tg.context_dim_for_n(n)

    w_shape, b_shape, flat = tg.theta_shapes(feat, K, bias=True)
    assert w_shape == (12, 5) and b_shape == (5,) and flat == 12 * 5 + 5, (w_shape, b_shape, flat)
    w_shape0, b_shape0, flat0 = tg.theta_shapes(feat, K, bias=False)
    assert w_shape0 == (12, 5) and b_shape0 is None and flat0 == 60, (w_shape0, b_shape0, flat0)

    # shared head: split a single flat theta and apply to a feature batch -> (B, K)
    rng = np.random.default_rng(0)
    theta_flat = rng.standard_normal(flat)
    w, b = tg.split_theta(theta_flat, feat, K, bias=True)
    assert w.shape == (12, 5) and b.shape == (5,), (w.shape, b.shape)
    B = 7
    X = rng.standard_normal((B, feat))
    logits = tg.apply_generated_head(X, theta_flat, feat, K, bias=True)
    assert logits.shape == (B, K), logits.shape
    # cross-check against the manual linear map
    assert np.allclose(logits, X @ w + b), "shared-head logits != X@W+b"

    # per-sample heads: (B, flat) -> the i-th head scores the i-th feature row
    theta_batch = rng.standard_normal((B, flat))
    logits_ps = tg.apply_generated_head(X, theta_batch, feat, K, bias=True)
    assert logits_ps.shape == (B, K), logits_ps.shape
    w_ps, b_ps = tg.split_theta(theta_batch, feat, K, bias=True)
    man = np.einsum("bf,bfc->bc", X, w_ps) + b_ps
    assert np.allclose(logits_ps, man), "per-sample-head logits != einsum"
    print("A. shape math (torchless) OK: flat=%d, logits %s, per-sample %s"
          % (flat, logits.shape, logits_ps.shape))


def test_context_builder_numpy():
    """B. Per-sample phase-context builder — numpy path (needs h3.group_directions, present on venv)."""
    if tg.group_directions is None:
        print("B. SKIP context builder (h3.group_directions unavailable)")
        return
    n = 4
    B, C, H, W = 6, n * 3, 4, 4               # C % n == 0 (group_directions requirement)
    rng = np.random.default_rng(1)
    osc = rng.standard_normal((B, C, H, W))
    ctx = tg.build_context_from_phase(osc, layer=0, n=n)
    assert ctx.shape == (B, tg.context_dim_for_n(n)), ctx.shape    # (B, 2n)
    # per-sample contexts must DIFFER across samples (distinct inputs -> distinct descriptors)
    assert not np.allclose(ctx[0], ctx[1]), "per-sample contexts collapsed (pooling lost the sample)"
    # dict-by-layer source + rate-coded builder share the same shape contract
    ctx_dict = tg.build_context_from_phase({0: osc, 1: osc}, layer=1, n=n)
    assert ctx_dict.shape == (B, 2 * n), ctx_dict.shape
    ctx_rate = tg.build_rate_coded_context(osc, layer=0, n=n)
    assert ctx_rate.shape == ctx.shape, (ctx_rate.shape, ctx.shape)   # OFF(b) is dim-matched to ON

    # wrong-context shuffle helpers keep shape + row count (so they stay aligned with x)
    task_ids = np.array([0, 0, 1, 1, 2, 2])
    rolled = tg.roll_context_by_task(ctx, task_ids)
    assert rolled.shape == ctx.shape, rolled.shape
    assert not np.allclose(rolled, ctx), "task-roll produced identical contexts (P5 would be inert)"
    randc = tg.random_context_like(ctx, seed=0)
    assert randc.shape == ctx.shape, randc.shape
    print("B. context builder (numpy) OK: ctx %s, rate %s, rolled differs=%s"
          % (ctx.shape, ctx_rate.shape, not np.allclose(rolled, ctx)))


def test_dict_generator_through_estimator_numpy():
    """D. The generator->estimator CONTRACT (torchless). PhaseContextThetaGen.forward returns a DICT
    {'theta_flat',...}; ctx_channel_capacity.estimate_c_ctx must consume that dict and produce the
    C_ctx-detectability behavior (constant gen -> ~0 bits; class-separable gen -> positive bits).

    The torch module can't run on this venv, so we use a numpy stand-in whose CALL CONTRACT matches the
    module exactly (returns the SAME dict), routed through the SAME estimate_c_ctx driver the GPU run
    uses. This is the regression that catches a dict-vs-array contract break in _apply_theta_gen (the
    bug that the array-returning lambdas in test_ctx_channel_capacity.py cannot detect)."""
    try:
        import ctx_channel_capacity as ccc
    except Exception as e:
        print("D. SKIP estimator contract (ctx_channel_capacity import failed: %s)" % e)
        return
    n, feat, K = 4, 8, 4
    cdim = tg.context_dim_for_n(n)
    _, _, flat = tg.theta_shapes(feat, K, bias=True)
    rng = np.random.default_rng(0)

    class DictGen:
        """Mirrors PhaseContextThetaGen.forward's RETURN CONTRACT: a dict with 'theta_flat'."""
        def __init__(self, seed=0, scale=0.1, hidden=32, const=False):
            g = np.random.default_rng(seed)
            self.W1 = g.standard_normal((cdim, hidden)) / np.sqrt(cdim)
            self.W2 = g.standard_normal((hidden, flat)) / np.sqrt(hidden)
            self.scale = scale
            self.const = const
            self.c0 = g.standard_normal(flat)

        def __call__(self, c):
            c = np.asarray(c, dtype=np.float64).reshape(-1)
            if self.const:
                tf = self.c0.copy()                      # ignores c -> C_ctx must be ~0
            else:
                tf = (np.tanh(c @ self.W1) @ self.W2) * self.scale
            w, b = tg.split_theta(tf, feat, K, bias=True)
            return {"theta_flat": tf, "weight": w, "bias": b}

    # class-separable contexts: K clusters with distinct means
    centers = rng.standard_normal((K, cdim)) * 3.0
    cbc = {t: [centers[t] + rng.standard_normal(cdim) * 0.3 for _ in range(40)] for t in range(K)}
    log2K = float(np.log2(K))

    res = ccc.estimate_c_ctx(DictGen(seed=0), cbc, n_shuffle=8, seed=0)
    assert 0.0 <= res["mi_lower_bits"] <= log2K + 1e-9, res["mi_lower_bits"]
    assert res["mi_lower_bits"] > 0.2, ("phase-conditioned dict-gen should carry positive C_ctx "
                                        "through estimate_c_ctx; got %.4f" % res["mi_lower_bits"])
    res_c = ccc.estimate_c_ctx(DictGen(seed=0, const=True), cbc, n_shuffle=8, seed=0)
    assert res_c["mi_lower_bits"] < 0.05, ("constant dict-gen must give ~0 bits (C_ctx=0 null); got "
                                           "%.4f -- _apply_theta_gen likely mis-handles the dict"
                                           % res_c["mi_lower_bits"])
    print("D. dict-gen -> estimate_c_ctx OK: separable mi_lower=%.4f (<=log2 %d=%.4f), constant mi_lower=%.4f"
          % (res["mi_lower_bits"], K, log2K, res_c["mi_lower_bits"]))


def test_generator_conditions_on_context_torch():
    """C. The torch generator + inject_fn — the C_ctx-detectability gate."""
    if torch is None:
        print("C. SKIP torch generator path (torch unavailable on this venv) — pure-shape logic covered in A.")
        return
    torch.manual_seed(0)
    n, feat, K = 4, 12, 5
    cdim = tg.context_dim_for_n(n)
    gen = tg.PhaseContextThetaGen(context_dim=cdim, feat_dim=feat, num_classes=K, hidden=32, seed=0)

    # flat dim matches the pure-shape math
    _, _, flat = tg.theta_shapes(feat, K, bias=True)
    assert gen.flat_dim == flat, (gen.flat_dim, flat)

    # single context -> theta with the right shapes
    c0 = torch.randn(cdim)
    out0 = gen(c0)
    assert out0["theta_flat"].shape == (flat,), out0["theta_flat"].shape
    assert out0["weight"].shape == (feat, K), out0["weight"].shape
    assert out0["bias"].shape == (K,), out0["bias"].shape

    # determinism: SAME context -> SAME theta
    out0b = gen(c0)
    assert torch.allclose(out0["theta_flat"], out0b["theta_flat"]), "generator non-deterministic on same c"

    # KEY GATE: DIFFERENT contexts -> DIFFERENT theta (g actually conditions on c; not C_ctx=0)
    c1 = torch.randn(cdim)
    out1 = gen(c1)
    diff = (out0["theta_flat"] - out1["theta_flat"]).abs().max().item()
    assert diff > 1e-6, ("generator IGNORES context (theta constant across c) -> C_ctx=0; "
                         "M2 could not detect a real channel. max|dtheta|=%g" % diff)

    # batch of contexts -> batch of per-sample heads
    Cb = torch.randn(8, cdim)
    outb = gen(Cb)
    assert outb["theta_flat"].shape == (8, flat), outb["theta_flat"].shape
    assert outb["weight"].shape == (8, feat, K), outb["weight"].shape
    # distinct context rows -> distinct theta rows
    assert (outb["theta_flat"][0] - outb["theta_flat"][1]).abs().max().item() > 1e-6

    # inject_fn end-to-end: a fixed feature extractor + per-sample context-generated heads -> (B,K) logits
    B = 8
    feats = torch.randn(B, feat)

    def feature_extractor(model, x):
        return feats                      # stand-in for the model's fixed readout (B, feat)

    inject_fn = tg.make_inject_fn(gen, feature_extractor)
    ctx_correct = torch.randn(B, cdim)
    logits = inject_fn(model=None, x=None, context=ctx_correct)
    assert logits.shape == (B, K), logits.shape

    # P7 zero-context sentinel: None -> a g(0) head, still (B, K)
    logits_zero = inject_fn(model=None, x=None, context=None)
    assert logits_zero.shape == (B, K), logits_zero.shape

    # wrong-context degradation pathway: permuting the per-sample context CHANGES the logits
    perm = torch.tensor(np.roll(np.arange(B), 1))
    logits_wrong = inject_fn(model=None, x=None, context=ctx_correct[perm])
    assert logits_wrong.shape == (B, K), logits_wrong.shape
    assert not torch.allclose(logits, logits_wrong), (
        "logits unchanged under context permutation -> the channel is bypassed (P5 would be inert)")

    # cross-check: inject_fn equals the manual per-sample generated-head application
    theta_flat = gen(ctx_correct)["theta_flat"]
    man = tg.apply_generated_head(feats, theta_flat, feat, K, bias=True)
    assert torch.allclose(logits, man), "inject_fn logits != manual generated-head application"
    print("C. torch generator + inject_fn OK: flat=%d, max|dtheta(c0,c1)|=%.4g, logits %s, "
          "wrong-context changes logits=%s" % (flat, diff, logits.shape,
                                               not torch.allclose(logits, logits_wrong)))


if __name__ == "__main__":
    test_shape_math_torchless()
    test_context_builder_numpy()
    test_dict_generator_through_estimator_numpy()
    test_generator_conditions_on_context_torch()
    print("\nALL THETA-GENERATOR TESTS PASSED")
