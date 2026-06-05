"""M2 — the MINIMAL phase-conditioned theta-generator (the SEED of M3's von Oswald hypernetwork).

CCC FRAMING (arXiv 2603.07415, Def 5 / Thm 4). M2 measures the TRUE Context Channel Capacity
    C_ctx = I(context c ; GENERATED PARAMETERS theta(c))     [bits]
which is STRICTLY POSITIVE only for conditional-regeneration (hypernetwork) architectures: a plain
classifier that MODIFIES A STATE has C_ctx = 0 by definition (no context->parameter pathway). This
module supplies that missing pathway in the smallest faithful form:

    g : c  ->  theta              (a tiny von Oswald-style hypernetwork)

where c is the AKOrN oscillator PHASE-STATE pooled into a fixed-dim context vector and theta are the
WEIGHTS of a TINY linear prediction head theta : feat -> num_classes. g is DELIBERATELY the seed of
M3's full von Oswald hypernetwork, so M2 -> M3 is one continuous build.

WHY a generator and not a gate. C_ctx is governed by how much theta VARIES with c. A degenerate g
that ignores c (constant theta) gives C_ctx = 0 — exactly the null the M2 estimator must be able to
DETECT — so the unit test here asserts that DIFFERENT contexts produce DIFFERENT theta. The capacity
estimation itself lives in ctx_channel_capacity.py; this file is only the generator + the plumbing
(context builders + an inject_fn) that the m2_primitives wrong-context probes drive.

WHAT THIS PROVIDES
  (1) PhaseContextThetaGen(nn.Module): forward(context_vec) -> theta. theta is returned BOTH as a flat
      param vector and reshaped to the head weight matrix (feat_dim, num_classes) (+ optional bias).
  (2) make_inject_fn(theta_gen, feature_extractor): -> inject_fn(model, x, context) that GENERATES the
      head from `context` and applies it to the model's features for x, returning (B, num_classes)
      logits. Drop-in for m2_primitives.wrong_context_probe / _eval_accuracy. Correct per-sample
      phase-context -> correct-context accuracy; wrong/random/zero context (P5/P5b/P6/P7) -> degradation.
  (3) build_context_from_phase(osc_state_or_loader, layer): per-sample phase-context vectors, pooled
      ONE SAMPLE AT A TIME via h3.group_directions + m2_primitives._pool_phase_state (OOM-safe: never
      stacks a full (B,C,H,W) batch; reduces each sample to a small 2n-dim descriptor first).
  (4) build_rate_coded_context: the matched-dim NON-oscillatory OFF(b) baseline — same pooling applied
      to a non-oscillator (pre-projection / readout) feature map, so OFF(b) is dimension-matched to ON.

DESIGN INVARIANTS (mirror h3.py / m2_primitives.py)
  * numpy-first; torch is OPTIONAL and only ever imported behind a guard so this file py_compiles and
    the pure-shape logic CPU-tests on the torchless h3venv. The torch nn.Module is only DEFINED when
    torch is importable (a torchless box still gets the context builders + the pooling/shape helpers).
  * Determinism: the generator's init takes a fixed seed; pooling reuses the fixed-order primitives.
  * OOM-safety: the context builder pools per-sample, exactly the _capture_osc / group_directions
    contract, and NEVER materializes the (B,C,H,W) tensor into a stacked context matrix.

Reuses (does NOT modify the M1 files):
  - m2_primitives._pool_phase_state   (fixed 2n-dim descriptor of one sample's group directions)
  - h3.group_directions               ((B,C,H,W) -> (n_sites, n) unit group-vectors; C % n == 0)
References:
  - von Oswald et al. (2020) "Continual learning with hypernetworks", ICLR. (g: c -> theta backbone.)
  - CCC, arXiv 2603.07415, Def 5 / Thm 4 (C_ctx = I(c; theta(c)); hypernetworks alone get C_ctx>>0).
"""
import os
import os
import sys

import numpy as np

# Make the M1 helpers importable WITHOUT copying them (we import, never touch, the experiment files).
# m1_wk0 is a sibling of this m2/ dir; resolve RELATIVE to __file__ so it works on any box/CWD
# (the hard-coded local path broke the import on the GPU box at /root/NC).
_M1_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m1_wk0"))
if _M1_DIR not in sys.path:
    sys.path.insert(0, _M1_DIR)

from m2_primitives import _pool_phase_state  # fixed 2n-dim per-sample descriptor (reused verbatim)

try:
    from h3 import group_directions            # (B,C,H,W) -> (n_sites, n) unit group-vectors
except Exception:                              # pragma: no cover - only needed for the real phase path
    group_directions = None

# torch is OPTIONAL: the nn.Module + inject_fn need it, but the context builders / pooling / the
# pure-shape helpers run on numpy alone so this file CPU-tests on the torchless analysis venv.
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torchless analysis box
    torch = None
    nn = None


# =====================================================================================
# 0. SHAPE MATH  (pure-numpy; testable without torch)
# =====================================================================================
def context_dim_for_n(n):
    """Dimension of the pooled phase-context for `n` group-axes.

    h3.group_directions yields (n_sites, n) unit vectors; m2_primitives._pool_phase_state pools the
    per-axis mean and per-axis second-moment over sites -> a fixed 2n-dim descriptor (independent of
    the site count, hence batch/H/W-invariant). This is the context_dim the generator consumes.
    """
    return int(2 * n)


def theta_shapes(feat_dim, num_classes, bias=True):
    """Shapes of the generated head parameters.

    The generated head is a LINEAR map feat -> num_classes: weight (feat_dim, num_classes) and an
    optional bias (num_classes,). Returns (weight_shape, bias_shape_or_None, flat_dim) where flat_dim
    is the total number of generated scalars (the hypernetwork's output width).
    """
    w_shape = (int(feat_dim), int(num_classes))
    b_shape = (int(num_classes),) if bias else None
    flat = w_shape[0] * w_shape[1] + (b_shape[0] if b_shape else 0)
    return w_shape, b_shape, flat


def split_theta(theta_flat, feat_dim, num_classes, bias=True):
    """Split a flat theta vector (..., flat_dim) into (weight (..., feat, classes), bias (..., classes)).

    Works for numpy arrays OR torch tensors (uses the array's own reshape), so the pure-shape test can
    exercise it without torch. The leading dims (if any) are preserved (per-sample batch of heads).
    """
    w_shape, b_shape, flat = theta_shapes(feat_dim, num_classes, bias=bias)
    lead = tuple(theta_flat.shape[:-1])
    assert int(theta_flat.shape[-1]) == flat, (
        f"theta_flat last dim {theta_flat.shape[-1]} != expected {flat} "
        f"(feat_dim={feat_dim}, num_classes={num_classes}, bias={bias})")
    n_w = w_shape[0] * w_shape[1]
    w = theta_flat[..., :n_w].reshape(*lead, w_shape[0], w_shape[1])
    b = theta_flat[..., n_w:].reshape(*lead, b_shape[0]) if bias else None
    return w, b


def apply_generated_head(features, theta_flat, feat_dim, num_classes, bias=True):
    """Apply a (possibly per-sample) generated linear head to features -> logits.

    features    : (B, feat_dim)
    theta_flat  : (flat_dim,)  -> ONE shared head applied to all B rows, OR
                  (B, flat_dim) -> a PER-SAMPLE head, the i-th head applied to the i-th feature row.
    Returns logits (B, num_classes). numpy-or-torch (uses matmul/einsum on the input's own type), so
    the shape test runs torchless.
    """
    w, b = split_theta(theta_flat, feat_dim, num_classes, bias=bias)
    if w.ndim == 2:                                  # shared head: (feat, classes)
        logits = features @ w
        if bias:
            logits = logits + b
    else:                                            # per-sample head: (B, feat, classes)
        # einsum the i-th feature row through the i-th weight matrix.
        if torch is not None and hasattr(features, "matmul") and not isinstance(features, np.ndarray):
            logits = torch.einsum("bf,bfc->bc", features, w)
        else:
            logits = np.einsum("bf,bfc->bc", np.asarray(features), np.asarray(w))
        if bias:
            logits = logits + b
    return logits


# =====================================================================================
# 1. THE GENERATOR  g : c -> theta   (torch nn.Module; the von Oswald hypernetwork SEED)
# =====================================================================================
if nn is not None:

    class PhaseContextThetaGen(nn.Module):
        """Minimal phase-conditioned theta-generator (von Oswald-style hypernetwork seed).

        g(c) = theta, a 2-layer MLP context_dim -> hidden -> flat_dim, where flat_dim parameterizes a
        TINY linear head theta : feat_dim -> num_classes (+ optional bias). The generated theta is the
        ONLY context->parameter pathway in the whole M2 pipeline; everything downstream (the head
        applied to features) is a fixed deterministic readout. That is precisely the CCC setup where
        C_ctx = I(c; theta(c)) can be > 0.

        forward(context_vec):
            context_vec : (context_dim,) or (B, context_dim)
            returns a dict {theta_flat, weight, bias}:
              theta_flat : (..., flat_dim)            — the hypernetwork output (use for C_ctx)
              weight     : (..., feat_dim, num_classes)
              bias       : (..., num_classes) or None
            The leading batch dim is preserved, so a batch of contexts yields a batch of (per-sample)
            heads — the inject_fn uses that to give every sample ITS OWN context-generated head.

        NOTE this is deliberately small: M2 only needs g to be EXPRESSIVE enough that different
        contexts can yield different theta (so C_ctx can be measured), not to win accuracy. M3 swaps in
        the full von Oswald hypernetwork without changing this interface.
        """

        def __init__(self, context_dim, feat_dim, num_classes, hidden=64, bias=True,
                     seed=0, theta_scale=0.1):
            super().__init__()
            self.context_dim = int(context_dim)
            self.feat_dim = int(feat_dim)
            self.num_classes = int(num_classes)
            self.bias = bool(bias)
            _, _, self.flat_dim = theta_shapes(feat_dim, num_classes, bias=bias)
            self.theta_scale = float(theta_scale)
            # Deterministic init (fixed seed) so the generator is reproducible across arms/seeds.
            g = torch.Generator().manual_seed(int(seed))
            self.fc1 = nn.Linear(self.context_dim, hidden)
            self.fc2 = nn.Linear(hidden, self.flat_dim)
            self.act = nn.Tanh()
            with torch.no_grad():
                for lin in (self.fc1, self.fc2):
                    lin.weight.copy_(torch.empty_like(lin.weight).normal_(0.0, 1.0, generator=g)
                                     * (1.0 / np.sqrt(lin.in_features)))
                    lin.bias.zero_()

        def forward(self, context_vec):
            if not torch.is_tensor(context_vec):
                context_vec = torch.as_tensor(np.asarray(context_vec), dtype=self.fc1.weight.dtype)
            context_vec = context_vec.to(self.fc1.weight.dtype)
            squeeze = (context_vec.dim() == 1)
            if squeeze:
                context_vec = context_vec.unsqueeze(0)            # (1, context_dim)
            h = self.act(self.fc1(context_vec))
            theta_flat = self.fc2(h) * self.theta_scale           # (B, flat_dim)
            w, b = split_theta(theta_flat, self.feat_dim, self.num_classes, bias=self.bias)
            if squeeze:
                theta_flat = theta_flat.squeeze(0)
                w = w.squeeze(0)
                b = b.squeeze(0) if b is not None else None
            return {"theta_flat": theta_flat, "weight": w, "bias": b}

        def generate_theta_flat(self, context_vec):
            """Convenience: just the flat theta vector (what the C_ctx estimator consumes).

            NOTE forward() returns a DICT {'theta_flat','weight','bias'}; the C_ctx estimator
            (ctx_channel_capacity.estimate_c_ctx) extracts theta_flat from that dict via _theta_vec, so
            either the module itself OR this bound method may be passed as the generator to the estimator.
            """
            return self.forward(context_vec)["theta_flat"]

else:  # pragma: no cover - torchless box: expose a clear error if someone instantiates the module.
    class PhaseContextThetaGen:  # type: ignore
        def __init__(self, *a, **k):
            raise RuntimeError(
                "PhaseContextThetaGen needs torch (absent on this analysis venv). The context builders "
                "and the pure-shape helpers (theta_shapes/split_theta/apply_generated_head) run torchless.")


# =====================================================================================
# 2. INJECT_FN  (drop-in for m2_primitives.wrong_context_probe / _eval_accuracy)
# =====================================================================================
def make_inject_fn(theta_gen, feature_extractor):
    """Build inject_fn(model, x, context) for the m2_primitives wrong-context probes.

    feature_extractor(model, x) -> features (B, feat_dim): the FIXED readout the generated head sits on
    top of (e.g. the model's pooled penultimate features). It is the deterministic part; the ONLY
    context->parameter pathway is theta_gen(context).

    inject_fn(model, x, context) -> logits (B, num_classes):
      * context is a (B, context_dim) tensor/array of PER-SAMPLE phase-contexts (the row order matches
        x). theta_gen turns each row into its own head; that head scores that sample's features ->
        every sample is classified through ITS OWN context-generated parameters. This is the genuine
        context->theta->prediction channel.
      * context is a single (context_dim,) vector -> one shared head for the batch.
      * context is None (P7 ZERO sentinel) -> a zero context vector -> the head g(0) (the no-context
        readout); the strongest ablation.
    A WRONG/RANDOM context (P5/P5b/P6) simply arrives as a row-permuted / random context matrix from
    the corresponding context_source, so the SAME code path yields the degraded accuracy. Pure eval:
    _eval_accuracy already wraps this in eval()/no_grad with RNG save-restore.
    """
    feat_dim = theta_gen.feat_dim
    num_classes = theta_gen.num_classes
    bias = theta_gen.bias
    context_dim = theta_gen.context_dim

    def inject_fn(model, x, context):
        features = feature_extractor(model, x)                    # (B, feat_dim)
        if torch is not None and not torch.is_tensor(features):
            features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        B = int(features.shape[0])

        # Resolve the context into a (B, context_dim) batch of per-sample contexts.
        if context is None:                                       # P7: zero context
            if torch is not None:
                ctx = torch.zeros(B, context_dim, dtype=features.dtype, device=features.device)
            else:
                ctx = np.zeros((B, context_dim), dtype=np.float64)
        else:
            if torch is not None and not torch.is_tensor(context):
                context = torch.as_tensor(np.asarray(context), dtype=features.dtype)
            ctx = context
            if ctx.ndim == 1:                                     # single shared context
                ctx = ctx.reshape(1, -1).repeat(B, 1) if torch is not None else \
                    np.repeat(ctx.reshape(1, -1), B, axis=0)
            assert int(ctx.shape[0]) == B, (
                f"context rows {ctx.shape[0]} != batch {B}; per-sample contexts must align with x. "
                "(wrong-context sources permute ROWS, keeping the count.)")
            assert int(ctx.shape[-1]) == context_dim, (
                f"context dim {ctx.shape[-1]} != generator context_dim {context_dim}")
            if torch is not None:
                ctx = ctx.to(features.dtype).to(features.device)

        out = theta_gen(ctx)                                      # per-sample heads: (B, flat_dim)
        theta_flat = out["theta_flat"]
        logits = apply_generated_head(features, theta_flat, feat_dim, num_classes, bias=bias)
        return logits

    return inject_fn


# =====================================================================================
# 3. CONTEXT BUILDERS  (OOM-safe, pool-per-sample, never stack full (B,C,H,W))
# =====================================================================================
def _pool_one_sample_osc(osc_sample, n):
    """Pool ONE sample's oscillator state (C, H, W) into its fixed 2n-dim phase-context descriptor.

    Wraps to (1, C, H, W), runs h3.group_directions (-> (n_sites, n) unit group-vectors for this one
    sample) and m2_primitives._pool_phase_state (-> [mean per axis, meansq per axis] = 2n dims). We
    NEVER stack the (B,C,H,W) batch; we slice one sample, reduce it, and discard it.
    """
    if group_directions is None:
        raise RuntimeError("build_context_from_phase needs h3.group_directions (h3 import failed).")
    a = np.asarray(osc_sample, dtype=np.float64)
    if a.ndim == 3:                       # (C, H, W) -> add the singleton batch dim group_directions wants
        a = a[None, ...]
    assert a.ndim == 4, f"expected one sample (C,H,W) or (1,C,H,W); got shape {a.shape}"
    U = group_directions(a, n=n)          # (n_sites, n) unit vectors for THIS sample
    return _pool_phase_state(U)           # (2n,) fixed descriptor


def _iter_osc_samples(osc_state_or_loader, layer, capture_fn=None, **capture_kw):
    """Yield per-sample oscillator states (C, H, W) from one of the accepted sources, WITHOUT ever
    holding more than one batch of (B,C,H,W) in memory at a time.

    Accepted `osc_state_or_loader`:
      * np.ndarray (B, C, H, W)                       -> one captured snapshot; iterate its rows.
      * dict {layer: (B,C,H,W)}                       -> index by `layer`, iterate rows.
      * a torch DataLoader + a `capture_fn`           -> capture_fn(batch_x) must return the layer's
        (b, C, H, W) phase-state for that batch (e.g. wrapping model.net.feature); iterate batch rows.
        This is the live path; the offline path (already-captured array) needs no torch.
    """
    if capture_fn is not None:
        for batch in osc_state_or_loader:
            xb = batch[0] if isinstance(batch, (tuple, list)) else batch
            osc = capture_fn(xb)                       # (b, C, H, W) for this layer, one batch only
            osc = np.asarray(osc.detach().cpu().numpy() if (torch is not None and torch.is_tensor(osc))
                             else osc, dtype=np.float64)
            for b in range(osc.shape[0]):
                yield osc[b]                           # (C, H, W)
        return
    src = osc_state_or_loader
    if isinstance(src, dict):
        src = src[layer]
    arr = np.asarray(src, dtype=np.float64)
    assert arr.ndim == 4, f"offline osc snapshot must be (B,C,H,W); got {arr.shape}"
    for b in range(arr.shape[0]):
        yield arr[b]                                   # (C, H, W)


def build_context_from_phase(osc_state_or_loader, layer=0, n=4, capture_fn=None):
    """Per-sample phase-context vectors c from the AKOrN oscillator phase-state (the ON / OFF(a) input).

    Returns a numpy array (n_samples, 2n): one fixed-dim phase-context per sample, pooled ONE SAMPLE AT
    A TIME (OOM-safe — never stacks a full (B,C,H,W); each sample is reduced to a 2n-dim descriptor
    first). `n` MUST match the model's group-axis count (BACKBONE['n']=4); read it from the model via
    int(getattr(model.net, 'n', 4)) at the call site.

    Sources (see _iter_osc_samples): an already-captured (B,C,H,W) array (offline path, the
    _capture_osc output for one layer), a {layer: (B,C,H,W)} dict, or a (loader, capture_fn) pair for
    the live path. ON = R6 phase-state; OFF(a) = R5:no_proj phase-state (same builder, different model).
    """
    ctx = [_pool_one_sample_osc(s, n) for s in
           _iter_osc_samples(osc_state_or_loader, layer, capture_fn=capture_fn)]
    return np.asarray(ctx, dtype=np.float64)


def build_rate_coded_context(feat_state_or_loader, layer=0, n=4, capture_fn=None):
    """OFF(b): the matched-dim, NON-oscillatory rate-coded context baseline.

    Same pooling pipeline, but fed a NON-oscillator feature map (e.g. the pre-projection / readout
    activations, NOT the sphere-normalized phase-state) so the context carries firing-RATE information
    rather than phase/synchrony information. The pooling reduces each sample's (C,H,W) to the SAME
    2n-dim descriptor as the phase context, so OFF(b) is DIMENSION-MATCHED to ON by construction (the
    ±2% capacity match is enforced separately on the generator params, which are identical here).

    Implementation note: group_directions normalizes each n-group to the unit sphere, so even a
    rate-coded map is reduced to a directional descriptor of the SAME shape — what differs between ON
    and OFF(b) is the CONTENT (oscillatory phase vs rate map), not the dimension. The driver supplies a
    capture_fn (or an offline array) that taps a non-oscillator activation for the OFF(b) arm.
    """
    return build_context_from_phase(feat_state_or_loader, layer=layer, n=n, capture_fn=capture_fn)


# =====================================================================================
# 4. WRONG-CONTEXT SHUFFLE HELPERS  (for m2_primitives.P5)  — bottleneck-specific, live here
# =====================================================================================
def roll_context_by_task(per_sample_context, task_ids):
    """P5 shuffle_fn: bind each sample to a DIFFERENT task's context.

    Given per-sample contexts (B, context_dim) and their task_ids (B,), roll the per-task context
    assignment by one task: every sample receives the context of the NEXT task (cyclically) at a
    matched position, so each sample is bound to a real-but-WRONG task's context (the P5 falsifier).
    Returns a (B, context_dim) array in the SAME row order as the input (so it stays aligned with x).
    """
    C = np.asarray(per_sample_context, dtype=np.float64)
    t = np.asarray(task_ids).reshape(-1)
    tasks = sorted(set(t.tolist()))
    if len(tasks) < 2:                       # only one task -> roll rows within the batch instead
        return np.roll(C, shift=1, axis=0)
    nxt = {tasks[i]: tasks[(i + 1) % len(tasks)] for i in range(len(tasks))}
    out = C.copy()
    # For each sample, pull a context drawn from the NEXT task's pool (first matching row, deterministic).
    by_task = {tk: np.flatnonzero(t == tk) for tk in tasks}
    for i in range(len(t)):
        donor_rows = by_task[nxt[t[i]]]
        out[i] = C[donor_rows[i % len(donor_rows)]]
    return out


def random_context_like(per_sample_context, seed=0):
    """P5b source: a fixed-seed random context matched in SHAPE and per-dim scale (mean/std) to the
    real contexts, carrying no task identity. Returns (B, context_dim)."""
    C = np.asarray(per_sample_context, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mu = C.mean(0, keepdims=True)
    sd = C.std(0, keepdims=True) + 1e-12
    return rng.standard_normal(C.shape) * sd + mu


if __name__ == "__main__":
    # Tiny self-demo: shapes only (no torch required for the pooling math).
    n, feat, K = 4, 12, 5
    print("context_dim_for_n(4) =", context_dim_for_n(n), "(expect 8)")
    print("theta_shapes(feat=12, K=5) =", theta_shapes(feat, K))
    osc = np.random.default_rng(0).standard_normal((3, n * 2, 4, 4))  # (B, C=8, H, W), C%n==0
    if group_directions is not None:
        ctx = build_context_from_phase(osc, layer=0, n=n)
        print("per-sample context shape =", ctx.shape, "(expect (3, 8))")
