"""CPU tests for the M2 measurement primitives (numpy-only; no torch, no sklearn, no GPU).

Run:  /tmp/h3venv/bin/python test_m2_primitives.py

Asserts:
  [1] ctx_capacity_bits on k-equal-eigenvalue feature matrices -> entropy = log2(k) bits,
      effective_dim = k exactly; plus the degenerate (constant F) -> (1.0, 0.0) limit, and the
      2**entropy == effective_dim identity / agreement with h3.effective_rank.
  [2] linear_task_decodability ~ 1.0 on synthetic LINEARLY-SEPARABLE phase states (per-task mean
      shift), and ~ chance on RANDOM (task-independent) phase states.
  [3] one wrong-context probe (P7-style zero injection) on a dummy numpy model+loader, confirming
      the accuracy + delta computation actually runs and is signed correctly.
"""
import math

import numpy as np

import m2_primitives as m2
import h3


# =====================================================================================
# [1] ctx_capacity_bits
# =====================================================================================
def _k_equal_eigenvalue_matrix(k, n=200, seed=0):
    """Build F:(n, k) whose mean-centered covariance has exactly k EQUAL nonzero eigenvalues:
    orthonormal columns scaled equally -> equal singular values -> p_i = 1/k. We orthonormalize a
    random (n,k) matrix (QR) and column-center; centering only removes a rank<=1 mean component, so
    we instead build columns that are already mean-zero and mutually orthogonal."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, k))
    # mean-center columns first (ctx_capacity_bits will center again, idempotently)
    A = A - A.mean(0, keepdims=True)
    # orthonormalize the centered columns -> equal singular values (all 1) after QR's R is dropped
    Q, _ = np.linalg.qr(A)
    Q = Q[:, :k]
    return Q  # columns orthonormal & (numerically) mean-zero -> k equal eigenvalues


def test_ctx_capacity_bits_k_equal():
    for k in (1, 2, 4, 8, 16):
        F = _k_equal_eigenvalue_matrix(k)
        eff_dim, ent_bits = m2.ctx_capacity_bits(F)
        # entropy of k equal eigenvalues = log2(k) bits; effective dim = k.
        assert abs(ent_bits - math.log2(k)) < 1e-6, (k, ent_bits, math.log2(k))
        assert abs(eff_dim - k) < 1e-6, (k, eff_dim)
        # the 2**entropy == effective_dim identity
        assert abs(eff_dim - 2.0 ** ent_bits) < 1e-9, (eff_dim, ent_bits)
        # base-2 twin agrees with h3.effective_rank (exp of natural-log entropy) to numerical noise
        assert abs(eff_dim - h3.effective_rank(F)) < 1e-6, (eff_dim, h3.effective_rank(F))
    print("[1a] ctx_capacity_bits: k equal eigenvalues -> entropy=log2(k), eff_dim=k (k=1,2,4,8,16) OK")


def test_ctx_capacity_bits_degenerate_and_spike():
    # constant matrix -> mean-centering kills all spectral mass -> (1 effective dim, 0 bits)
    F_const = np.full((50, 10), 3.14)
    eff, ent = m2.ctx_capacity_bits(F_const)
    assert eff == 1.0 and ent == 0.0, (eff, ent)
    # single dominant direction (rank-1 signal + tiny noise) -> ~1 effective dim, ~0 bits
    rng = np.random.default_rng(1)
    v = rng.normal(size=(1, 8))
    F_spike = rng.normal(size=(100, 1)) @ v + 1e-6 * rng.normal(size=(100, 8))
    eff_s, ent_s = m2.ctx_capacity_bits(F_spike)
    assert 1.0 <= eff_s < 1.2, eff_s
    assert ent_s < 0.3, ent_s
    print(f"[1b] ctx_capacity_bits: constant->(1.0,0.0); rank-1 spike->(~{eff_s:.3f},{ent_s:.3f} bits) OK")


# =====================================================================================
# [2] linear_task_decodability
# =====================================================================================
def test_decodability_separable_is_high():
    """Per-task MEAN-SHIFTED group-direction clouds: task identity is linearly present -> cv ~ 1.0."""
    rng = np.random.default_rng(0)
    n_tasks, per_task, n_sites, n = 4, 40, 30, 4
    phase_state_by_task = {}
    for t in range(n_tasks):
        center = np.zeros(n)
        center[t % n] = 3.0                        # task-specific axis bias -> separable after pooling
        states = []
        for _ in range(per_task):
            U = center[None, :] + 0.3 * rng.normal(size=(n_sites, n))
            U = U / np.clip(np.linalg.norm(U, axis=1, keepdims=True), 1e-12, None)
            states.append(U)
        phase_state_by_task[t] = states
    cv, chance = m2.linear_task_decodability(phase_state_by_task, seed=0)
    assert cv > 0.9, (cv, chance)
    assert abs(chance - 0.25) < 1e-9, chance
    print(f"[2a] decodability(separable) cv={cv:.3f} >> chance={chance:.3f} OK")


def test_decodability_random_is_chance():
    """Task-INDEPENDENT random phase clouds: no task bits -> cv ~ chance."""
    rng = np.random.default_rng(7)
    n_tasks, per_task, n_sites, n = 4, 40, 30, 4
    phase_state_by_task = {}
    for t in range(n_tasks):
        states = []
        for _ in range(per_task):
            U = rng.normal(size=(n_sites, n))      # identical distribution for EVERY task
            U = U / np.clip(np.linalg.norm(U, axis=1, keepdims=True), 1e-12, None)
            states.append(U)
        phase_state_by_task[t] = states
    cv, chance = m2.linear_task_decodability(phase_state_by_task, seed=0)
    # near chance: allow generous slack for the small synthetic CV, but it must NOT decode the tasks
    assert abs(cv - chance) < 0.15, (cv, chance)
    assert cv < 0.45, cv
    print(f"[2b] decodability(random) cv={cv:.3f} ~ chance={chance:.3f} OK")


def test_decodability_labels_array_form():
    """The array+labels signature path (not the dict path) also works and pools 1-D states. Each task
    sits on its OWN axis (one-vs-rest friendly, the realistic per-task group-direction layout) so the
    least-squares OvR fallback can solve it — a single shared ordinal axis is genuinely hard for any
    linear-regression OvR probe (the middle class gets swamped at the argmax), which is a property of
    least-squares OvR, not of the data, so we use the separable per-axis geometry instead."""
    rng = np.random.default_rng(3)
    n_tasks = 3
    X, y = [], []
    for t in range(n_tasks):
        for _ in range(30):
            v = 0.4 * rng.normal(size=n_tasks)
            v[t] += 5.0                            # task t lives on its own axis -> OvR-separable
            X.append(v)
            y.append(t)
    cv, chance = m2.linear_task_decodability(X, labels=y, seed=1)
    assert cv > 0.9, (cv, chance)
    print(f"[2c] decodability(array+labels) cv={cv:.3f} OK")


# =====================================================================================
# [3] wrong-context probe on a dummy model
# =====================================================================================
class _DummyLoader:
    """Yields (x, y, task_id) numpy batches like _build_probe_loader (shuffle=False)."""
    def __init__(self, X, y, batch=8):
        self.X, self.y, self.batch = X, y, batch

    def __iter__(self):
        for i in range(0, len(self.y), self.batch):
            xb = self.X[i:i + self.batch]
            yb = self.y[i:i + self.batch]
            tid = np.zeros(len(yb), dtype=int)
            yield xb, yb, tid


def test_wrong_context_probe_delta_runs():
    """Dummy linear model whose logits depend on an injected context. With the CORRECT context the
    model is ~perfect; injecting ZERO context (P7) collapses it to one class -> positive delta. This
    exercises the REAL accuracy + delta math; only the injection site is the abstracted hook."""
    rng = np.random.default_rng(0)
    n, d, C = 80, 5, 4
    y = rng.integers(0, C, size=n)
    # class-separated inputs so a context-aware model can classify them
    centers = rng.normal(size=(C, d)) * 4.0
    X = np.stack([centers[c] + rng.normal(size=d) for c in y])

    W = rng.normal(size=(d, C))  # the "content" projection

    def model(x):
        """Correct-context forward: logits = x @ W + the model's OWN context (here a perfect per-row
        one-hot of the true class, standing in for a workspace that has bound the right task)."""
        x = np.asarray(x, float)
        # the model's internal correct context: a strong bias toward the true class.
        # (we fake "knowing" via nearest-center; a real model would read this from the bottleneck.)
        sims = -((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (B, C)
        return x @ W + 10.0 * sims

    def inject_fn(model, x, context):
        """Bottleneck hook: rerun the content path but REPLACE the context with `context`.
        context is a (C,) bias vector broadcast over the batch, or None (P7 sentinel) -> ZERO."""
        x = np.asarray(x, float)
        if context is None:
            bias = np.zeros(C)
        else:
            bias = np.asarray(context, float).reshape(C)
        return x @ W + bias[None, :]

    loader = _DummyLoader(X, y)

    # P7 = zero context. Build the context_source via the factory; inject_fn zeros on None sentinel.
    ctx_src = m2.P7()
    res = m2.wrong_context_probe(model, loader, ctx_src, inject_fn=inject_fn)
    assert res["n"] == n, res["n"]
    assert 0.0 <= res["accuracy_correct"] <= 1.0 and 0.0 <= res["accuracy_wrong"] <= 1.0, res
    # correct context should classify well; zeroed context loses the task signal -> positive delta.
    assert res["accuracy_correct"] > 0.8, res
    assert res["accuracy_delta"] > 0.3, res
    assert abs(res["accuracy_delta"] - (res["accuracy_correct"] - res["accuracy_wrong"])) < 1e-12

    # also exercise P5b (random context) to confirm a different context_source path runs end-to-end.
    def random_ctx(model, loader):
        g = np.random.default_rng(123)
        return g.normal(size=C)  # (C,) random bias broadcast over the batch, carries no task identity
    res_b = m2.wrong_context_probe(model, loader, m2.P5b(random_ctx), inject_fn=inject_fn)
    assert 0.0 <= res_b["accuracy_wrong"] <= 1.0
    print(f"[3] wrong-context probe: acc_correct={res['accuracy_correct']:.3f}, "
          f"acc_wrong(zero)={res['accuracy_wrong']:.3f}, delta={res['accuracy_delta']:.3f}; "
          f"P5b random delta={res_b['accuracy_delta']:.3f} OK")


if __name__ == "__main__":
    test_ctx_capacity_bits_k_equal()
    test_ctx_capacity_bits_degenerate_and_spike()
    test_decodability_separable_is_high()
    test_decodability_random_is_chance()
    test_decodability_labels_array_form()
    test_wrong_context_probe_delta_runs()
    print("\nALL M2 PRIMITIVE CPU TESTS PASSED")
