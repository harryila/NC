"""M2 HEADLINE METRIC — the TRUE CCC context channel capacity C_ctx, in BITS.

This module estimates the quantity the M2 pre-registration ratified (experiments/m2/preregistration-M2.md,
2026-06-03): the CCC Context Channel Capacity

    C_ctx = I(context c ; GENERATED PARAMETERS theta(c))         [CCC arXiv 2603.07415, Def 5 / Thm 4]

NOT the representational I(phase; task) the M1/M2 pre-check measured. The distinction is the whole point
of CCC: a plain classifier that MODIFIES A STATE (AKOrN / GASPnet as a classifier) has C_ctx = 0 by
definition, no matter how much task info lives in its phase-state, because there is no context->parameter
pathway. Only a CONDITIONAL-REGENERATION architecture (a hypernetwork g: c -> theta) has C_ctx >> 0. So M2
routes the AKOrN oscillator phase-state (layers 1-2) as the CONTEXT c into a MINIMAL phase-conditioned
theta-GENERATOR g: c -> theta, where theta parameterizes a small prediction head, and estimates I(c; theta(c)).

theta(c) is a DETERMINISTIC function of c, so I(c; theta) is governed entirely by how much theta VARIES with
c. The two limits the estimators MUST honor:
  * a CONSTANT generator (theta independent of c)                 -> C_ctx = 0 bits;
  * a generator whose theta perfectly SEPARATES the K contexts    -> up to log2(K) bits.

TWO complementary, pre-registered estimators (report BOTH; primary = (1)):

  (1) DECODABILITY LOWER BOUND  (Fano-style, the CCC P5 spirit, PRIMARY).
      Can a frozen, CROSS-VALIDATED linear probe recover the CONTEXT-CLASS T (which task/context the
      phase-context came from) from the GENERATED theta vector? Build the held-out confusion matrix
      N[t, t_hat] over CV folds and convert it to a mutual-information LOWER BOUND in bits:

          I(c; theta) >= I(T; That) = H(T) - H(T | That)      [BITS]

      where T is the true context-class, That = probe(theta(c)) the decoded class, and both entropies are
      taken from the empirical confusion-matrix joint. This is a VALID lower bound on I(c; theta): T -> c ->
      theta -> That is a Markov chain (the probe sees ONLY theta), so by the Data Processing Inequality
      I(T; That) <= I(T; theta) <= I(c; theta). We additionally CHANCE-CORRECT: a probe that ignores theta
      and guesses produces a confusion matrix whose I(T; That) -> 0 in expectation, and we clamp the report
      at >= 0 and cap at H(T) = log2(K) (the channel cannot carry more than the context's own entropy).

  (2) EFFECTIVE-DIMENSION reading  (the representational-dimensionality twin).
      m2_primitives.ctx_capacity_bits on the matrix of generated theta vectors (one row per sampled
      context) -> effective_dim_bits / entropy_bits of the theta CLOUD: how many bits of variation the
      context induces in the generated params. A constant generator -> rank-0 theta cloud -> (1.0, 0.0).

WHY (1) IS THE HEADLINE AND (2) IS THE COMPANION: (1) is a genuine information LOWER BOUND tied to the
context-class T (it answers "how many task bits does the generator's theta actually carry"); (2) is an
unsupervised spectral reading of theta's variability (it answers "how many effective dimensions of theta
the context moves") and is NOT a bound on I(c; theta) by itself — a generator can spread theta over many
dimensions that are uncorrelated with the context-class. Report both; greenlight on (1).

DESIGN INVARIANTS (mirror m2_primitives.py / h3.py):
  * numpy-first, float64 for the information math; torch is OPTIONAL and only needed to RUN a real
    theta-generator's forward — every estimator here consumes already-materialized numpy theta vectors,
    so the file py_compiles and CPU-tests on the torchless/sklearnless h3venv.
  * Determinism: the CV probe takes a fixed seed (delegated to linear_task_decodability's seed).
  * Reuse, don't reinvent: the CV linear probe is m2_primitives.linear_task_decodability's machinery;
    the effective-dim reading is m2_primitives.ctx_capacity_bits. This module adds ONLY (a) the
    confusion-matrix -> MI-in-bits conversion, and (b) the theta-generator plumbing that turns a
    {context-class -> phase-context} mapping into {context-class -> theta(c)} pairs.

References (verified against primary sources):
  - CCC, arXiv:2603.07415. C_ctx = max_{P(c)} I(c; theta(c)) (Def 5 / Thm 4); hypernetworks attain
    C_ctx >> 0 (0.95-0.98 on Split-MNIST), state-modifying learners sit at C_ctx ~ 0.
  - Cover & Thomas, "Elements of Information Theory", 2e: I(X;Y)=H(X)-H(X|Y); Data Processing Inequality
    (Thm 2.8.1): X->Y->Z => I(X;Z) <= I(X;Y). Fano's inequality (Thm 2.10.1) relates error rate to H(X|Y).
  - von Oswald et al. (2020) "Continual learning with hypernetworks", ICLR 2020, arXiv:1906.00695 — the
    g: c -> theta backbone this minimal generator is the seed of (M2 -> M3 one build).
  - Roy & Vetterli (2007) effective rank; Shannon (1948) — via m2_primitives.ctx_capacity_bits.
"""
import math
import os
import sys

import numpy as np

# Reuse the M1/M2 primitives verbatim (do NOT touch the M1 experiment files; import from them). The
# m1_wk0 dir is a sibling of this m2/ dir; add it to sys.path so the import works regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_M1 = os.path.normpath(os.path.join(_HERE, "..", "m1_wk0"))
if _M1 not in sys.path:
    sys.path.insert(0, _M1)

import m2_primitives as m2  # noqa: E402  (ctx_capacity_bits, linear_task_decodability, _ridge_ovr_fit_predict)

# torch is needed ONLY to RUN a real theta-generator's forward; the CPU test passes plain numpy
# generators, so this file imports/CPU-tests with no torch (mirrors m2_primitives.py's guard).
try:
    import torch
except Exception:  # pragma: no cover - torchless analysis box
    torch = None


# =====================================================================================
# CONFUSION-MATRIX  ->  MUTUAL-INFORMATION LOWER BOUND  (in BITS)
# =====================================================================================
def confusion_to_mi_bits(confusion):
    """Mutual information I(T; That) in BITS from a confusion matrix N[t, t_hat].

    N[t, t_hat] = (held-out) count of samples whose TRUE context-class is t and whose probe-PREDICTED
    class is t_hat. We form the empirical joint p(t, t_hat) = N / N.sum() and compute

        I(T; That) = sum_{t, t_hat} p(t,t_hat) * log2[ p(t,t_hat) / (p(t) p(t_hat)) ]   [BITS]

    equivalently H(T) - H(T | That). This is a VALID LOWER BOUND on I(c; theta) by the Data Processing
    Inequality: the probe sees only theta(c), so T -> c -> theta -> That is a Markov chain and
    I(T; That) <= I(T; theta) <= I(c; theta).

    Properties this guarantees (and the tests assert):
      * a probe that perfectly recovers T (diagonal confusion) -> I = H(T) = log2(K) for balanced T;
      * a probe whose prediction is INDEPENDENT of T (rows proportional to the column marginal, the
        chance / constant-generator case) -> I = 0 exactly.
    The raw plug-in MI is biased UP on finite samples; chance-correction is applied by the caller
    (compute_C_ctx subtracts the matched-shuffle floor). Here we only clamp at >= 0 for numerical safety.
    """
    N = np.asarray(confusion, dtype=np.float64)
    total = N.sum()
    if total <= 0:
        return 0.0
    p = N / total                                  # joint p(t, t_hat)
    pt = p.sum(axis=1, keepdims=True)              # p(t)      (row marginal)
    pth = p.sum(axis=0, keepdims=True)             # p(t_hat)  (col marginal)
    # mask the support; 0*log0 := 0. Only cells with mass AND nonzero marginals contribute.
    denom = pt @ pth                               # p(t) p(t_hat), outer product
    mask = (p > 0) & (denom > 0)
    mi = float(np.sum(p[mask] * np.log2(p[mask] / denom[mask])))
    return max(0.0, mi)


def entropy_bits(labels):
    """Shannon entropy H(T) in BITS of a label array (the channel ceiling for the MI lower bound)."""
    y = np.asarray(labels)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


# =====================================================================================
# THE PRIMARY ESTIMATOR (1): DECODABILITY MI LOWER BOUND  (Fano-style, cross-validated)
# =====================================================================================
def _cv_confusion(X, y, classes, n_splits=5, seed=0, ridge=1e-3):
    """Held-out confusion matrix N[t, t_hat] from a FROZEN linear probe under stratified K-fold CV.

    Mirrors m2_primitives.linear_task_decodability's split + probe machinery EXACTLY (same stratified
    folds, same fixed seed, same sklearn-or-numpy-ridge probe) but accumulates the CONFUSION rather than
    just the accuracy, because the MI lower bound needs the full joint. X:(n, p) is the GENERATED THETA
    matrix (one row per context sample), y the context-class of each row. Returns (N, used_classes)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    N = np.zeros((K, K), dtype=np.float64)

    counts = np.asarray([int((y == c).sum()) for c in classes], dtype=np.int64)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_folds = min(n_splits, int(counts.min())) or 1
    folds = [[] for _ in range(n_folds)]
    F = len(folds)
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % F].append(int(ix))

    use_sklearn = False
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        use_sklearn = True
    except Exception:
        use_sklearn = False

    for k in range(F):
        te = np.asarray(folds[k], dtype=int)
        tr = np.asarray([ix for j in range(F) if j != k for ix in folds[j]], dtype=int)
        if len(te) == 0 or len(tr) == 0:
            continue
        if len(set(y[tr].tolist())) < 2:           # train fold must see >=2 classes for any probe
            continue
        if use_sklearn:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
        else:
            pred = m2._ridge_ovr_fit_predict(X[tr], y[tr], X[te], classes, ridge=ridge)
        for true_lbl, pred_lbl in zip(y[te], pred):
            N[cidx[true_lbl], cidx[pred_lbl]] += 1.0
    return N, classes


def decodability_mi_lower_bits(theta, labels, n_splits=5, seed=0, ridge=1e-3, n_shuffle=10):
    """PRIMARY estimator: a chance-corrected, cross-validated MI LOWER BOUND on I(context; theta), in BITS.

    theta:(n, p)  = the GENERATED head-parameter vectors, one row per sampled context.
    labels:(n,)   = the context-CLASS of each sample (which task/context c was drawn from).

    Procedure:
      1. CV confusion N[t, t_hat] from a frozen linear probe on {theta -> context-class}  (_cv_confusion).
      2. raw_mi = confusion_to_mi_bits(N)        -- I(T; That), a DPI lower bound on I(c; theta).
      3. CHANCE FLOOR: the plug-in MI is biased UP on finite samples (a probe with no real signal still
         produces a confusion matrix with small positive MI). We estimate that bias by re-running the
         WHOLE CV pipeline on `n_shuffle` LABEL-PERMUTED copies (theta fixed, labels shuffled => no real
         theta->class signal) and take the mean shuffled MI as the chance floor.
      4. mi_lower = clamp( raw_mi - chance_floor, 0, H(T) )  -- clamp at >= 0 (a lower bound is never
         negative) and cap at H(T) = log2(K) (the channel cannot exceed the context's own entropy).

    Returns dict {mi_lower_bits, raw_mi_bits, chance_floor_bits, cv_accuracy, Hmax_bits, n_classes, n}.
    A CONSTANT generator (all theta rows identical) yields raw_mi ~ chance_floor -> mi_lower ~ 0.
    A generator whose theta perfectly separates the K classes yields raw_mi ~ log2(K) -> mi_lower ~ log2(K).
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    y = np.asarray(labels)
    classes = sorted(set(y.tolist()), key=str)
    K = len(classes)
    n = len(y)
    Hmax = entropy_bits(y)

    if K < 2 or n < 2:
        return {"mi_lower_bits": 0.0, "raw_mi_bits": 0.0, "chance_floor_bits": 0.0,
                "cv_accuracy": float("nan"), "Hmax_bits": Hmax, "n_classes": K, "n": n}

    # 1-2: real confusion + raw MI
    N, _ = _cv_confusion(theta, y, classes, n_splits=n_splits, seed=seed, ridge=ridge)
    raw_mi = confusion_to_mi_bits(N)
    cv_acc = float(np.trace(N) / N.sum()) if N.sum() > 0 else float("nan")

    # 3: chance floor via label permutation (theta fixed; same CV machinery, same seed family)
    floors = []
    perm_rng = np.random.default_rng(seed + 9973)
    for s in range(max(0, n_shuffle)):
        yp = y.copy()
        perm_rng.shuffle(yp)
        Ns, _ = _cv_confusion(theta, yp, classes, n_splits=n_splits, seed=seed + 1 + s, ridge=ridge)
        floors.append(confusion_to_mi_bits(Ns))
    chance_floor = float(np.mean(floors)) if floors else 0.0

    # 4: chance-correct, clamp at >= 0, cap at H(T)
    mi_lower = max(0.0, raw_mi - chance_floor)
    mi_lower = min(mi_lower, Hmax)
    return {
        "mi_lower_bits": float(mi_lower),
        "raw_mi_bits": float(raw_mi),
        "chance_floor_bits": float(chance_floor),
        "cv_accuracy": cv_acc,
        "Hmax_bits": float(Hmax),
        "n_classes": K,
        "n": n,
    }


# =====================================================================================
# THE COMPANION ESTIMATOR (2): EFFECTIVE-DIMENSION OF THE THETA CLOUD  (in BITS)
# =====================================================================================
def theta_effective_dim_bits(theta):
    """COMPANION estimator: effective dimensionality of the generated-theta cloud, in BITS.

    theta:(n, p) = generated head-params, one row per sampled context. Delegates to
    m2_primitives.ctx_capacity_bits (mean-centers theta over samples, base-2 spectral entropy) and
    returns (effective_dim_bits, entropy_bits). A constant generator -> rank-0 centered theta ->
    (1.0, 0.0): one effective dimension, zero bits of context-induced variation.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    return m2.ctx_capacity_bits(theta)


# =====================================================================================
# compute_C_ctx  — both estimators on pre-materialized {context-class -> theta(c)} pairs
# =====================================================================================
def compute_C_ctx(theta_by_context_class, n_splits=5, seed=0, ridge=1e-3, n_shuffle=10):
    """Compute C_ctx from already-generated theta vectors grouped BY CONTEXT-CLASS.

    theta_by_context_class:
      * dict {context-class t: array/list of theta(c) vectors for samples whose context came from t}, OR
      * a tuple (theta_matrix:(n,p), labels:(n,)) already flattened.

    Runs BOTH pre-registered estimators and returns a single dict:
      {
        mi_lower_bits   : PRIMARY — chance-corrected CV decodability MI lower bound (bits), in [0, log2 K]
        eff_dim_bits    : COMPANION — effective_dim of the theta cloud (a DIMENSION COUNT, == 2**ent_dim_bits)
        ent_dim_bits    : COMPANION — spectral entropy of the theta cloud in BITS (the "how many bits" read)
        n_classes       : K
        Hmax_bits       : log2(K) for balanced classes (= H(T) actually observed), the MI ceiling
        raw_mi_bits, chance_floor_bits, cv_accuracy, n : diagnostics from the primary estimator
      }
    """
    # ---- assemble (theta_matrix, labels) ----------------------------------------------------------
    if isinstance(theta_by_context_class, dict):
        rows, labels = [], []
        for t, thetas in sorted(theta_by_context_class.items(), key=lambda kv: str(kv[0])):
            for th in thetas:
                rows.append(np.asarray(th, dtype=np.float64).ravel())
                labels.append(t)
        dims = {len(r) for r in rows}
        if len(dims) != 1:
            raise ValueError(f"generated theta vectors have ragged dims {sorted(dims)}; "
                             "every generated head-parameter vector must share one length")
        theta = np.asarray(rows, dtype=np.float64)
        labels = np.asarray(labels)
    else:
        theta, labels = theta_by_context_class
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        labels = np.asarray(labels)

    primary = decodability_mi_lower_bits(theta, labels, n_splits=n_splits, seed=seed,
                                         ridge=ridge, n_shuffle=n_shuffle)
    eff_dim_bits, ent_dim_bits = theta_effective_dim_bits(theta)

    return {
        "mi_lower_bits": primary["mi_lower_bits"],          # PRIMARY headline (bits)
        "eff_dim_bits": float(eff_dim_bits),                # COMPANION: effective dimension count (2**ent)
        "ent_dim_bits": float(ent_dim_bits),                # COMPANION: spectral entropy in bits
        "n_classes": primary["n_classes"],
        "Hmax_bits": primary["Hmax_bits"],                  # log2(K), the MI ceiling
        "raw_mi_bits": primary["raw_mi_bits"],
        "chance_floor_bits": primary["chance_floor_bits"],
        "cv_accuracy": primary["cv_accuracy"],
        "n": primary["n"],
    }


# =====================================================================================
# estimate_c_ctx  — ONE-CALL driver: run a theta-generator over contexts, then compute_C_ctx
# =====================================================================================
def _theta_vec(out):
    """Coerce a theta-generator output into a 1-D numpy theta vector.

    The minimal phase-conditioned theta-generator (theta_generator.PhaseContextThetaGen) returns a
    DICT {'theta_flat', 'weight', 'bias'} from its forward (theta_flat is the hypernetwork output the
    C_ctx estimator consumes). A plain numpy/torch callable may instead return the bare theta tensor/
    array. Accept BOTH: pull 'theta_flat' from a mapping, else use the value as-is; detach torch tensors.
    """
    if isinstance(out, dict):
        out = out.get("theta_flat", out.get("theta", None))
        if out is None:
            raise ValueError("theta-generator returned a dict without a 'theta_flat'/'theta' key")
    if torch is not None and isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64).ravel()


def _apply_theta_gen(theta_gen, context):
    """Run the theta-generator g: c -> theta on ONE phase-context, returning a 1-D numpy theta vector.

    Accepts:
      * a plain python/numpy callable theta_gen(context) -> array-like theta (or {'theta_flat': ...});
      * a torch.nn.Module (run under no_grad/eval; context coerced to a tensor; output detached to numpy).
        The real PhaseContextThetaGen.forward returns a DICT, so we extract theta_flat via _theta_vec.
    Kept tiny + duck-typed so the CPU test can pass a numpy lambda and the GPU driver can pass the real
    von-Oswald-seed module without this module importing torch at module scope.
    """
    if torch is not None and isinstance(theta_gen, torch.nn.Module):
        was_training = theta_gen.training
        theta_gen.eval()
        try:
            with torch.no_grad():
                c = context
                if not isinstance(c, torch.Tensor):
                    c = torch.as_tensor(np.asarray(context, dtype=np.float32))
                return _theta_vec(theta_gen(c))
        finally:
            if was_training:
                theta_gen.train()
    return _theta_vec(theta_gen(context))


def estimate_c_ctx(theta_gen, context_by_class, n_splits=5, seed=0, ridge=1e-3, n_shuffle=10):
    """ONE-CALL DRIVER. Generate theta(c) for every phase-context, grouped by context-class, then C_ctx.

    theta_gen:  g: c -> theta. A callable (numpy) or a torch.nn.Module (the minimal phase-conditioned
                theta-generator that is the seed of M3's von Oswald hypernetwork). theta parameterizes a
                small prediction head; we treat its OUTPUT VECTOR as theta(c).
    context_by_class: dict {context-class t: iterable of per-sample phase-CONTEXTS c}. Each c is whatever
                the generator consumes (e.g. a pooled phase-state descriptor from
                m2_primitives._pool_phase_state / avalanche_backbone._capture_osc). The context-CLASS t is
                the supervision label for the decodability lower bound (which task the context came from).

    Returns the compute_C_ctx dict (primary mi_lower_bits + companion eff_dim_bits + diagnostics).
    Because theta(c) is DETERMINISTIC, identical contexts give identical theta; the MI is driven purely by
    how the generator's theta varies WITH the context-class.
    """
    theta_by_class = {}
    for t, contexts in context_by_class.items():
        thetas = [_apply_theta_gen(theta_gen, c) for c in contexts]
        theta_by_class[t] = thetas
    return compute_C_ctx(theta_by_class, n_splits=n_splits, seed=seed, ridge=ridge, n_shuffle=n_shuffle)
