"""CPU tests for the TRUE CCC C_ctx estimator (numpy-only; no torch, no sklearn, no GPU).

Run:  /tmp/h3venv/bin/python test_ctx_channel_capacity.py

Asserts the three pre-registered correctness limits for C_ctx = I(context; generated-theta) in bits,
plus the confusion->MI conversion's two extreme points:

  [0] confusion_to_mi_bits: a perfect (diagonal) balanced confusion -> log2(K) bits; an independent
      confusion (every row == the column marginal) -> exactly 0 bits.
  [a] CONSTANT theta-generator (theta does not depend on context) -> C_ctx ~ 0 bits (BOTH estimators).
  [b] PERFECT generator (theta = one-hot(context-class)) -> mi_lower ~ log2(K) and eff_dim_bits ~ K.
  [c] NOISY-but-separable generator -> 0 < mi_lower < log2(K) (strictly between the two limits).
  [d] the estimate_c_ctx one-call driver runs a numpy theta_gen end-to-end and reproduces (a)/(b).
"""
import math

import numpy as np

import ctx_channel_capacity as ccc


# =====================================================================================
# [0] confusion_to_mi_bits — the two extreme points the MI conversion MUST hit exactly
# =====================================================================================
def test_confusion_to_mi_extremes():
    K = 4
    per = 50
    # perfect diagonal, balanced classes -> I(T;That) = H(T) = log2(K)
    N_perfect = np.eye(K) * per
    mi_perfect = ccc.confusion_to_mi_bits(N_perfect)
    assert abs(mi_perfect - math.log2(K)) < 1e-9, mi_perfect

    # independent predictor: every row proportional to the SAME column marginal -> I = 0 exactly.
    col_marginal = np.array([0.1, 0.2, 0.3, 0.4])
    N_indep = np.outer(np.full(K, per), col_marginal)   # row t: per * col_marginal -> p(t,that)=p(t)p(that)
    mi_indep = ccc.confusion_to_mi_bits(N_indep)
    assert abs(mi_indep - 0.0) < 1e-9, mi_indep

    # empty confusion -> 0 (numerical safety)
    assert ccc.confusion_to_mi_bits(np.zeros((K, K))) == 0.0
    print(f"[0] confusion->MI: diagonal={mi_perfect:.4f}=log2({K}); independent={mi_indep:.2e}~0 OK")


# =====================================================================================
# [a] CONSTANT generator -> C_ctx ~ 0 bits
# =====================================================================================
def test_constant_generator_is_zero():
    """A generator whose theta is INDEPENDENT of the context. Two sub-cases:

      (i) TRULY constant theta (no variation at all) -> BOTH estimators bottom out: mi_lower ~ 0 AND
          eff_dim ~ 1 / ent_dim ~ 0 (rank-0 centered cloud).
      (ii) constant MEAN + class-INDEPENDENT isotropic noise -> the PRIMARY mi_lower ~ 0 (the headline:
          no CONTEXT information in theta), while the COMPANION eff_dim counts the noise dimensions
          (>1) — by design: estimator (2) is an unsupervised reading of ALL theta variation, not a bound
          on I(c;theta), so isotropic noise legitimately raises eff_dim without raising the channel. This
          is exactly why (1) is the headline and (2) is a companion."""
    K, per, p = 5, 60, 8
    # (i) truly constant: every theta row identical -> rank-0 centered cloud
    const_theta = np.random.default_rng(0).normal(size=p)
    theta_const = {t: [const_theta.copy() for _ in range(per)] for t in range(K)}
    out_i = ccc.compute_C_ctx(theta_const, seed=0)
    assert out_i["n_classes"] == K
    assert out_i["mi_lower_bits"] < 0.10, out_i             # ~0 bits (primary)
    assert out_i["eff_dim_bits"] < 1.5, out_i               # ~1 effective dim (companion bottoms out too)
    assert out_i["ent_dim_bits"] < 0.6, out_i               # ~0 bits of spectral variation

    # (ii) constant mean + class-INDEPENDENT noise: primary still ~0 (no context info in theta)
    rng = np.random.default_rng(1)
    cmean = rng.normal(size=p)
    theta_noisy = {t: [cmean + 1e-2 * rng.normal(size=p) for _ in range(per)] for t in range(K)}
    out_ii = ccc.compute_C_ctx(theta_noisy, seed=0)
    assert out_ii["mi_lower_bits"] < 0.10, out_ii           # PRIMARY headline: ~0 context bits
    print(f"[a] constant gen: (i) truly-const mi_lower={out_i['mi_lower_bits']:.4f}, "
          f"eff_dim={out_i['eff_dim_bits']:.3f}; (ii) noisy-const mi_lower={out_ii['mi_lower_bits']:.4f} "
          f"(both ~0; companion eff_dim(ii)={out_ii['eff_dim_bits']:.2f} counts the noise) OK")


# =====================================================================================
# [b] PERFECT generator (theta = one-hot(class)) -> mi_lower ~ log2(K)
# =====================================================================================
def test_perfect_generator_is_logK():
    """theta(c) = one-hot(context-class) exactly. The context perfectly DETERMINES theta-class, so the
    linear probe recovers T with no error -> mi_lower ~ log2(K). The theta cloud spans K orthogonal
    one-hot directions equally -> eff_dim_bits ~ K, ent_dim_bits ~ log2(K)."""
    K, per = 4, 60
    theta_by_class = {}
    for t in range(K):
        onehot = np.zeros(K)
        onehot[t] = 1.0
        theta_by_class[t] = [onehot.copy() for _ in range(per)]  # DETERMINISTIC: identical within class
    out = ccc.compute_C_ctx(theta_by_class, seed=0)
    Hmax = math.log2(K)
    assert abs(out["Hmax_bits"] - Hmax) < 1e-9, out
    assert out["mi_lower_bits"] > Hmax - 0.10, out          # ~log2(K)
    assert out["mi_lower_bits"] <= Hmax + 1e-9, out         # capped at the ceiling
    assert abs(out["cv_accuracy"] - 1.0) < 1e-9, out        # probe is perfect
    # K balanced one-hot points lie on a simplex: mean-centering removes 1 dof, leaving K-1 EQUAL
    # nonzero eigenvalues -> eff_dim_bits == K-1 exactly, ent_dim_bits == log2(K-1).
    assert abs(out["eff_dim_bits"] - (K - 1)) < 1e-6, out
    assert abs(out["ent_dim_bits"] - math.log2(K - 1)) < 1e-6, out
    print(f"[b] perfect gen: mi_lower={out['mi_lower_bits']:.4f} bits (~log2({K})={Hmax:.4f}), "
          f"cv_acc={out['cv_accuracy']:.3f}, eff_dim={out['eff_dim_bits']:.3f} OK")


# =====================================================================================
# [c] NOISY-but-separable -> 0 < mi_lower < log2(K)
# =====================================================================================
def test_noisy_separable_is_between():
    """theta(c) = class-mean (a per-class axis bias) + noise large enough to cause SOME probe errors but
    not destroy separability. mi_lower must land STRICTLY between 0 and log2(K)."""
    rng = np.random.default_rng(2)
    K, per, p = 4, 80, 6
    theta_by_class = {}
    for t in range(K):
        center = np.zeros(p)
        center[t] = 1.6                              # modest per-class axis bias
        theta_by_class[t] = [center + 1.4 * rng.normal(size=p) for _ in range(per)]  # heavy noise
    out = ccc.compute_C_ctx(theta_by_class, seed=0)
    Hmax = math.log2(K)
    assert 0.05 < out["mi_lower_bits"] < Hmax - 0.05, out   # strictly between the limits
    assert 0.0 < out["cv_accuracy"] < 1.0, out              # imperfect but above chance
    assert out["mi_lower_bits"] <= out["Hmax_bits"] + 1e-9, out
    print(f"[c] noisy-separable: mi_lower={out['mi_lower_bits']:.4f} bits in (0,{Hmax:.3f}), "
          f"cv_acc={out['cv_accuracy']:.3f}, eff_dim={out['eff_dim_bits']:.3f} OK")


# =====================================================================================
# [d] estimate_c_ctx one-call driver with a numpy theta_gen
# =====================================================================================
def test_estimate_c_ctx_driver():
    """The one-call driver: a numpy theta_gen: c -> theta run over {class -> [contexts]}.
      * PERFECT gen reads the class off the context (context carries the class in its first coord) and
        emits a one-hot -> mi_lower ~ log2(K).
      * CONSTANT gen ignores the context -> mi_lower ~ 0.
    Confirms the generator plumbing (the M2->M3 bridge) ties into compute_C_ctx correctly."""
    rng = np.random.default_rng(5)
    K, per, cdim = 3, 50, 7
    # contexts: each carries its class in coord 0 (so a generator CAN read it), plus noise elsewhere.
    context_by_class = {}
    for t in range(K):
        ctxs = []
        for _ in range(per):
            c = rng.normal(size=cdim)
            c[0] = float(t)                          # class signal in coord 0
            ctxs.append(c)
        context_by_class[t] = ctxs

    def perfect_gen(c):
        """g: c -> theta. Reads the class off coord 0 and emits the exact one-hot -> theta determines class."""
        cls = int(round(float(np.asarray(c).ravel()[0])))
        cls = max(0, min(K - 1, cls))
        oh = np.zeros(K)
        oh[cls] = 1.0
        return oh

    def constant_gen(c):
        """g: c -> theta independent of c (a degenerate generator) -> C_ctx = 0."""
        return np.array([0.5, -0.5, 0.5])

    out_perfect = ccc.estimate_c_ctx(perfect_gen, context_by_class, seed=0)
    out_const = ccc.estimate_c_ctx(constant_gen, context_by_class, seed=0)
    Hmax = math.log2(K)
    assert out_perfect["mi_lower_bits"] > Hmax - 0.10, out_perfect
    assert out_const["mi_lower_bits"] < 0.10, out_const
    print(f"[d] driver: perfect_gen mi_lower={out_perfect['mi_lower_bits']:.4f} (~log2({K})={Hmax:.4f}); "
          f"constant_gen mi_lower={out_const['mi_lower_bits']:.4f} (~0) OK")


# =====================================================================================
# [e] tuple input path + Hmax cap sanity
# =====================================================================================
def test_tuple_input_and_cap():
    """compute_C_ctx also accepts a pre-flattened (theta, labels) tuple, and the mi_lower never exceeds
    Hmax even on a perfect-but-imbalanced design."""
    K = 3
    rows, labels = [], []
    for t in range(K):
        oh = np.zeros(K); oh[t] = 1.0
        reps = 30 + 10 * t                           # imbalanced classes
        for _ in range(reps):
            rows.append(oh.copy()); labels.append(t)
    out = ccc.compute_C_ctx((np.asarray(rows), np.asarray(labels)), seed=0)
    assert out["mi_lower_bits"] <= out["Hmax_bits"] + 1e-9, out
    assert out["mi_lower_bits"] > out["Hmax_bits"] - 0.10, out   # still ~perfect
    print(f"[e] tuple input + cap: mi_lower={out['mi_lower_bits']:.4f} <= Hmax={out['Hmax_bits']:.4f} OK")


if __name__ == "__main__":
    test_confusion_to_mi_extremes()
    test_constant_generator_is_zero()
    test_perfect_generator_is_logK()
    test_noisy_separable_is_between()
    test_estimate_c_ctx_driver()
    test_tuple_input_and_cap()
    print("\nALL C_ctx CPU TESTS PASSED")
