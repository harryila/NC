"""M2 measurement PRIMITIVES — the workspace-bottleneck observables (M1-independent).

This module is the analysis scaffolding for M2 (the Oscillatory Workspace bottleneck). It is
deliberately decoupled from the live M1 experiment: it imports a couple of pure helpers from h3
(group_directions, effective_rank) but does NOT touch avalanche_backbone / ladder / the driver, and
every numeric routine here is CPU-testable on numpy alone (torch + the real model are only needed for
the forward passes inside the wrong-context probes, which take an injected, mockable forward).

Three primitives, in the order M2 needs them:

  1. ctx_capacity_bits(F)        -- the log2 analogue of h3.effective_rank. How many BITS of context
                                    capacity a workspace bottleneck representation carries (spectral
                                    entropy in bits + the 2**entropy effective dimensionality).
  2. wrong_context_probe / P5/P5b/P6/P7
                                  -- pure-eval ablations that REPLACE the context a sample would
                                    normally bind to (wrong-task / random / random-theta / zero) and
                                    report the accuracy DELTA vs the correct-context forward. The
                                    context-injection SITE depends on the not-yet-built bottleneck, so
                                    it is a documented hook (`inject_fn`); the accuracy + delta math is
                                    real and tested on a dummy model.
  3. linear_task_decodability(...)
                                  -- a frozen linear probe (sklearn LogisticRegression if present, else
                                    a numpy ridge one-vs-rest least-squares) that decodes task-id from
                                    the phase state. The cheap M2 pre-check for the S_N "zero task bits"
                                    trap: if a linear probe CANNOT read task-id off the phases, there is
                                    no task information for the workspace to bind, and the whole
                                    bottleneck story collapses before we spend GPU on it.

DESIGN INVARIANTS (mirror h3.py):
  * numpy-first, float64 for the spectral math; torch + sklearn are OPTIONAL and only ever imported
    behind a guard so this file py_compiles and CPU-tests on the torchless/sklearnless h3venv.
  * Determinism: every estimator that resamples (CV folds) takes a fixed seed.
  * The wrong-context probes are PURE EVAL: no grad, model put in eval(), RNG save/restored so the
    injected context does not perturb the rest of the driver's stream.

References (verified against primary sources):
  - Roy & Vetterli (2007) "The effective rank: a measure of effective dimensionality", EUSIPCO.
    erank = exp(H_spec); the bits form here is 2**H_2 with H_2 the base-2 spectral (Shannon) entropy.
  - Shannon (1948). Entropy in bits; 2**H is the "perplexity"/effective-alphabet-size reading.
  - Kornblith et al. (2019) feature centering convention (we mean-center F over examples, as h3 does).
"""
import math

import numpy as np

# torch is needed ONLY for the real forward inside the wrong-context probes; the dummy-model CPU test
# injects a numpy forward, so this file imports/CPU-tests with no torch (mirrors h3.py's guard).
try:
    import torch
except Exception:  # pragma: no cover - torchless analysis box
    torch = None


# =====================================================================================
# 1. CONTEXT CAPACITY IN BITS  (log2 analogue of h3.effective_rank)
# =====================================================================================
def ctx_capacity_bits(F):
    """Context capacity of a representation matrix F:(n, p), measured in BITS.

    This is the base-2 twin of h3.effective_rank (which returns exp(H) with H the natural-log
    spectral entropy). We mean-center F over examples (the Kornblith convention h3 uses), take the
    squared singular values lambda_i of the centered F (== the PCA covariance eigenvalues up to the
    constant 1/(n-1) that cancels under normalization), normalize to a probability spectrum
    p_i = lambda_i / sum(lambda), and form the base-2 Shannon entropy

        H2 = -sum_i p_i * log2(p_i)      [BITS]

    RETURNS a tuple (effective_dim_bits, entropy_bits):
      * effective_dim_bits = 2 ** H2  -- the "effective dimensionality expressed as a COUNT" (a.k.a.
        perplexity / effective number of equally-weighted dimensions). This is a DIMENSION COUNT, not
        a bit count: for k equal eigenvalues it equals exactly k. It is the base-2 sibling of
        h3.effective_rank (whose exp(H_nat) gives the same number; the base cancels in 2**H2 == e**Hnat).
      * entropy_bits = H2            -- the raw spectral entropy IN BITS. For k equal eigenvalues this
        is log2(k). This is the quantity to report when you want "how many BITS of context capacity".

    Read the two apart: entropy_bits answers "how many bits" (log2 k); effective_dim_bits answers "how
    many effective dimensions" (k). They are related by effective_dim_bits == 2 ** entropy_bits.

    Degenerate cases: an all-constant / rank-0 centered F (no spectral mass) returns (1.0, 0.0)
    — one effective dimension, zero bits — matching the entropy=0 / 2**0=1 limit.
    """
    F = np.asarray(F, dtype=np.float64)
    F = F - F.mean(0, keepdims=True)               # center features over examples (rows), as h3 does
    s = np.linalg.svd(F, compute_uv=False)
    ev = s ** 2
    ev = ev[ev > 1e-12]
    if ev.size == 0:                               # no spectral mass: 1 effective dim, 0 bits
        return 1.0, 0.0
    p = ev / ev.sum()
    entropy_bits = float(-np.sum(p * np.log2(p)))
    effective_dim_bits = float(2.0 ** entropy_bits)
    return effective_dim_bits, entropy_bits


# =====================================================================================
# 2. WRONG-CONTEXT PROBING  (P5 / P5b / P6 / P7)  — pure-eval ablations
# =====================================================================================
def _eval_accuracy(model, probe_loader, device="cpu", inject_fn=None, context=None):
    """Mean top-1 accuracy of `model` over `probe_loader`, OPTIONALLY injecting a context at the
    workspace bottleneck via `inject_fn`.

    Contract for the not-yet-built bottleneck (documented HOOK):
      inject_fn(model, x, context) -> logits  (B, num_classes)
        Runs ONE forward of `model` on inputs `x`, but with the workspace context REPLACED by
        `context` at the bottleneck site. When inject_fn is None we call the model normally
        (inject_fn defaults to "no injection" == the correct-context forward), so the SAME code path
        computes both the reference accuracy and the ablated accuracy. The probe is the live model's
        OWN context plumbing — this primitive does not know where the bottleneck is, the injected
        inject_fn does. The accuracy + delta math below is real and testable regardless.

    probe_loader yields (x, y, task_id) batches (the Avalanche / _build_probe_loader convention);
    a 2-tuple (x, y) is also accepted. Returns (accuracy, n_examples). Pure eval: no grad, model.eval(),
    RNG save/restored if torch is present so the injected context cannot perturb the rest of the stream.
    """
    correct = 0
    total = 0

    def _argmax(logits):
        a = np.asarray(logits)
        return a.argmax(axis=1)

    # torch path: eval(), no_grad, RNG save/restore so injection is side-effect-free on the stream.
    if torch is not None and isinstance(model, torch.nn.Module):
        was_training = model.training
        model.eval()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                for batch in probe_loader:
                    x, y = batch[0], batch[1]
                    x = x.to(device)
                    logits = inject_fn(model, x, context) if inject_fn is not None else model(x)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    yv = np.asarray(y.cpu() if hasattr(y, "cpu") else y).reshape(-1)
                    correct += int((pred == yv).sum())
                    total += len(yv)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if was_training:
                model.train()
    else:
        # numpy/dummy path: model is a plain callable; inject_fn / model return array-like logits.
        for batch in probe_loader:
            x, y = batch[0], batch[1]
            logits = inject_fn(model, x, context) if inject_fn is not None else model(x)
            pred = _argmax(logits)
            yv = np.asarray(y).reshape(-1)
            correct += int((pred == yv).sum())
            total += len(yv)

    acc = float(correct) / float(total) if total else float("nan")
    return acc, total


def wrong_context_probe(model, probe_loader, context_source, inject_fn=None, device="cpu"):
    """Generic wrong-context ablation. Returns the accuracy DELTA = acc_correct - acc_wrong.

    Steps (all PURE EVAL):
      1. acc_correct = accuracy with NO injection (the model's own, correct context) — `inject_fn` is
         bypassed for this reference forward so P5/P5b/P6/P7 share one byte-identical baseline.
      2. context = context_source(model, probe_loader)  — a CALLABLE the M2 driver supplies that
         produces whatever object the bottleneck binds (wrong-task code, random vector, zero, ...).
         Making it a callable keeps THIS primitive ignorant of the bottleneck's dtype/shape.
      3. acc_wrong = accuracy with that context injected at the bottleneck via inject_fn.
      4. delta = acc_correct - acc_wrong  (POSITIVE => the wrong context HURT, i.e. the model was
         genuinely relying on the correct context; ~0 => context was inert / "zero task bits").

    Returns dict {accuracy_correct, accuracy_wrong, accuracy_delta, n}. The accuracy + delta are real;
    only the injection SITE is abstracted behind inject_fn (the documented bottleneck hook).
    """
    acc_correct, n = _eval_accuracy(model, probe_loader, device=device, inject_fn=None)
    context = context_source(model, probe_loader)
    acc_wrong, _ = _eval_accuracy(model, probe_loader, device=device,
                                  inject_fn=inject_fn, context=context)
    return {
        "accuracy_correct": acc_correct,
        "accuracy_wrong": acc_wrong,
        "accuracy_delta": acc_correct - acc_wrong,
        "n": n,
    }


# --- the four pre-registered context_source callables -------------------------------------------
# Each has signature context_source(model, probe_loader) -> context object. They are thin, named
# wrappers so the four ablations read self-documenting at the call site; the actual context payload
# is opaque to wrong_context_probe and is only interpreted by the (bottleneck-specific) inject_fn.

def _ctx_wrong_task(reference_context_source):
    """P5 builder: wrong-TASK context. Wraps a reference context_source that, given (model, loader),
    yields the correct per-batch context; P5 deterministically PERMUTES the task assignment (rolls the
    per-sample context by one task) so every sample is bound to another task's context. We cannot
    construct the shuffle without knowing the context object, so P5 is parameterized by the driver's
    real context_source plus a `shuffle_fn` that rolls it; see P5() below."""
    raise NotImplementedError  # placeholder; P5 is assembled in P5() from driver-supplied callables


def P5(reference_context_source, shuffle_fn):
    """P5 = WRONG-TASK context. context_source = shuffle_fn(reference_context_source(model, loader)).
    `reference_context_source(model, loader)` returns the correct context; `shuffle_fn(ctx)` returns a
    task-permuted version (e.g. roll the per-sample task code by +1 task) so each sample is bound to a
    DIFFERENT real task's context. Both callables are driver-supplied (bottleneck-specific). Returns a
    context_source callable for wrong_context_probe."""
    def context_source(model, probe_loader):
        return shuffle_fn(reference_context_source(model, probe_loader))
    return context_source


def P5b(random_context_source):
    """P5b = RANDOM context. context_source = random_context_source(model, loader), a fixed-seed random
    draw in the SAME space as a real context (matched norm/shape) but carrying no task identity. The
    driver supplies random_context_source so this primitive stays bottleneck-agnostic. Returns a
    context_source callable."""
    def context_source(model, probe_loader):
        return random_context_source(model, probe_loader)
    return context_source


def P6(random_theta_source):
    """P6 = RANDOM theta_base. The AKOrN oscillator's base natural-frequency / phase offset theta_base
    is the workspace's *control* input; P6 replaces it with a fixed-seed random theta_base while
    leaving the bound context intact, isolating the oscillator-control pathway from the content
    pathway. random_theta_source(model, loader) -> a theta_base object the inject_fn knows how to
    splice. Returns a context_source callable."""
    def context_source(model, probe_loader):
        return random_theta_source(model, probe_loader)
    return context_source


def P7(zero_context_source=None):
    """P7 = ZERO context. context_source returns a sentinel telling inject_fn to ZERO the bottleneck
    context (the strongest ablation: no context at all). If the driver needs a typed zero (e.g. a zero
    tensor of the right shape) it passes zero_context_source(model, loader); otherwise the sentinel
    None is returned and inject_fn must interpret None as "zero the context". Returns a context_source
    callable."""
    def context_source(model, probe_loader):
        if zero_context_source is not None:
            return zero_context_source(model, probe_loader)
        return None  # sentinel: inject_fn zeros the bottleneck context
    return context_source


# =====================================================================================
# 3. LINEAR TASK DECODABILITY  (the S_N "zero task bits" pre-check)
# =====================================================================================
def _pool_phase_state(state):
    """Flatten/pool ONE sample's phase state into a 1-D feature vector.

    Accepts:
      * a group_directions-style array (n_sites, n) -> pooled to a 1-D descriptor by concatenating
        the per-axis mean and the per-axis second moment over sites (a permutation-invariant,
        fixed-length summary of the spherical phase cloud, robust to the site count).
      * an already-1-D vector -> returned as-is (float).
      * any other ndarray -> ravelled.
    """
    a = np.asarray(state, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        # (n_sites, n) spherical directions: pool over sites -> [mean per axis, meansq per axis].
        mean = a.mean(axis=0)
        meansq = (a ** 2).mean(axis=0)
        return np.concatenate([mean, meansq])
    return a.ravel()


def _onehot(y, classes):
    Y = np.zeros((len(y), len(classes)), dtype=np.float64)
    idx = {c: i for i, c in enumerate(classes)}
    for r, v in enumerate(y):
        Y[r, idx[v]] = 1.0
    return Y


def _ridge_ovr_fit_predict(Xtr, ytr, Xte, classes, ridge=1e-3):
    """Frozen linear probe with NO sklearn: ridge-regularized one-vs-rest least squares on a one-hot
    target, predict by argmax of the linear scores. Bias via an appended ones column. Deterministic.
    Returns predicted labels for Xte."""
    Xtr = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xte = np.column_stack([Xte, np.ones(len(Xte))])
    Y = _onehot(ytr, classes)
    p = Xtr.shape[1]
    R = ridge * np.eye(p)
    R[-1, -1] = 0.0                                # do not penalize the bias term
    # W = (X^T X + ridge I)^-1 X^T Y   (normal equations; stable for the small probe-feature dims)
    W = np.linalg.solve(Xtr.T @ Xtr + R, Xtr.T @ Y)
    scores = Xte @ W
    pred_idx = scores.argmax(axis=1)
    return np.asarray([classes[i] for i in pred_idx])


def linear_task_decodability(phase_state_by_task, labels=None, n_splits=5, seed=0, ridge=1e-3):
    """Can a FROZEN linear probe read task-id off the phase state? The cheap M2 pre-check for the
    S_N "zero task bits" trap.

    phase_state_by_task:
      * dict {task t: list/array of per-sample phase states}  (states are group_directions output
        (n_sites, n), already-pooled vectors, or any ndarray; pooled by _pool_phase_state), OR
      * an array/list of per-sample phase states, in which case `labels` MUST give the task-id of each.

    Fits a linear probe (sklearn LogisticRegression if importable, else the numpy ridge one-vs-rest)
    under stratified K-fold CV and returns (cv_accuracy, chance) where chance = 1 / n_tasks (uniform)
    refined to the majority-class frequency so an imbalanced design cannot masquerade as signal.

    Interpretation: cv_accuracy >> chance => task identity is linearly present in the phases (good,
    there are task bits for the workspace to bind). cv_accuracy ~ chance => "zero task bits": the
    phases carry no decodable task identity and the bottleneck story needs rethinking BEFORE GPU.
    """
    # ---- assemble (X, y) -------------------------------------------------------------------------
    if isinstance(phase_state_by_task, dict):
        X, y = [], []
        for t, states in sorted(phase_state_by_task.items(), key=lambda kv: str(kv[0])):
            for s in states:
                X.append(_pool_phase_state(s))
                y.append(t)
    else:
        if labels is None:
            raise ValueError("labels required when phase_state_by_task is not a {task: states} dict")
        X = [_pool_phase_state(s) for s in phase_state_by_task]
        y = list(labels)
    # pad/truncate to a common length defensively (pooled descriptors should already match)
    dims = {len(v) for v in X}
    if len(dims) != 1:
        raise ValueError(f"pooled phase descriptors have ragged dims {sorted(dims)}; "
                         "ensure all samples share the same phase-state shape")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()), key=str)
    n = len(y)

    # chance = max(uniform, majority-class) so imbalance can't look like decodability
    counts = np.asarray([int((y == c).sum()) for c in classes], dtype=np.float64)
    chance = float(max(1.0 / len(classes), counts.max() / n))

    if len(classes) < 2:
        return float("nan"), chance  # only one task present: decodability is undefined

    # ---- stratified K-fold split (fixed seed) ----------------------------------------------------
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(min(n_splits, int(counts.min())) or 1)]
    K = len(folds)
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % K].append(int(ix))

    # ---- frozen linear probe: sklearn if present, else numpy ridge OvR ---------------------------
    use_sklearn = False
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        use_sklearn = True
    except Exception:
        use_sklearn = False

    accs = []
    for k in range(K):
        te = np.asarray(folds[k], dtype=int)
        tr = np.asarray([ix for j in range(K) if j != k for ix in folds[j]], dtype=int)
        if len(te) == 0 or len(tr) == 0:
            continue
        # guard: a fold's train split must see >=2 classes for any probe to fit
        if len(set(y[tr].tolist())) < 2:
            continue
        if use_sklearn:
            from sklearn.linear_model import LogisticRegression
            # NOTE: the `multi_class` kwarg was deprecated in sklearn 1.5 and REMOVED in 1.7+
            # (multinomial is the default for >2 classes). Omit it for forward-compat.
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
        else:
            pred = _ridge_ovr_fit_predict(X[tr], y[tr], X[te], classes, ridge=ridge)
        accs.append(float((pred == y[te]).mean()))

    cv_accuracy = float(np.mean(accs)) if accs else float("nan")
    return cv_accuracy, chance
