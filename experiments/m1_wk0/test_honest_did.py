"""CPU test for the HONEST-DiD: O_intra gets real augmentation-baseline content.

Pure-numpy, no torch/GPU. Injects a toy linear-CKA-shaped scoring function so we exercise the
overlap_summaries / intra_task_cka / paired_did control flow exactly as the driver does, and assert:

  * fallback (no augmented features): O_intra == 1.0, inner == O_inter - 1, flag False  (back-compat).
  * honest path (augmented features given): O_intra < 1 (real self-CKA), inner != O_inter - 1,
    flag True, and per_layer_intra is populated.
  * paired_did over per-seed summaries flips honest_did True only when BOTH arms supply the baseline.

Run:  python test_honest_did.py     (expects numpy; no torch needed)
"""
import numpy as np

import h3


def _toy_cka(X, Y):
    """A bounded [0,1] similarity that does NOT need torch: cosine^2 between the row-flattened,
    mean-centered feature matrices. CKA(X,X)==1; CKA of perturbed copies < 1; matches the contract
    overlap_summaries relies on (diag handled separately by inter_task_cka_matrix)."""
    X = np.asarray(X, float).ravel()
    Y = np.asarray(Y, float).ravel()
    X = X - X.mean()
    Y = Y - Y.mean()
    denom = (np.linalg.norm(X) * np.linalg.norm(Y)) + 1e-12
    return float((X @ Y) ** 2 / (denom ** 2))


def _feat_dicts(n_tasks=4, layers=(0, 1, 2), p=12, aug_scale=0.0, seed=0):
    """Build {task: {layer: feature matrix}} clean dicts and an augmented variant. aug_scale>0
    perturbs each snapshot's features to emulate the augmented probe pass (so self-CKA < 1)."""
    rng = np.random.default_rng(seed)
    n = 6
    clean, aug = {}, {}
    for t in range(n_tasks):
        clean[t], aug[t] = {}, {}
        for l in layers:
            F = rng.normal(size=(n, p))
            clean[t][l] = F
            aug[t][l] = F + aug_scale * rng.normal(size=(n, p))
    return clean, aug


def test_fallback_is_legacy_overlap_contrast():
    clean, _ = _feat_dicts()
    s = h3.overlap_summaries(clean, layers=[0, 1, 2], cka_fn=_toy_cka)
    assert s["O_intra"] == 1.0, s["O_intra"]
    assert s["intra_is_augmentation_baseline"] is False
    assert s["per_layer_intra"] is None
    assert abs(s["inner"] - (s["O_inter"] - 1.0)) < 1e-12, s["inner"]
    print("[1] fallback OK: O_intra=1.0, inner == O_inter - 1, flag False")


def test_honest_did_has_real_intra():
    clean, aug = _feat_dicts(aug_scale=0.6, seed=1)
    s = h3.overlap_summaries(clean, layers=[0, 1, 2], cka_fn=_toy_cka, aug_features_by_task=aug)
    assert s["intra_is_augmentation_baseline"] is True
    assert s["per_layer_intra"] is not None and set(s["per_layer_intra"]) == {0, 1, 2}
    # real augmentation self-CKA is strictly between cross-task overlap floor and the trivial 1.0
    assert 0.0 < s["O_intra"] < 1.0, s["O_intra"]
    # the load-bearing assertion: inner is NOT the legacy O_inter - 1 once O_intra has real content
    assert abs(s["inner"] - (s["O_inter"] - 1.0)) > 1e-6, (s["inner"], s["O_inter"])
    assert abs(s["inner"] - (s["O_inter"] - s["O_intra"])) < 1e-12
    print(f"[2] honest OK: O_intra={s['O_intra']:.4f} (<1), "
          f"inner={s['inner']:.4f} != O_inter-1={s['O_inter']-1:.4f}")


def test_paired_did_flags_honest_only_when_both_arms_baselined():
    layers = [0, 1, 2]
    # R6 keeps MORE self-similarity under augment (aug_scale small) than R5 (aug_scale large);
    # R6 also has lower cross-task overlap (different feature scale via seed).
    sum_R6, sum_R5 = {}, {}
    for seed in range(5):
        c6, a6 = _feat_dicts(aug_scale=0.2, seed=100 + seed)
        c5, a5 = _feat_dicts(aug_scale=0.8, seed=200 + seed)
        sum_R6[seed] = h3.overlap_summaries(c6, layers=layers, cka_fn=_toy_cka, aug_features_by_task=a6)
        sum_R5[seed] = h3.overlap_summaries(c5, layers=layers, cka_fn=_toy_cka, aug_features_by_task=a5)
    did = h3.paired_did(sum_R6, sum_R5)
    assert did["honest_did"] is True, did["honest_did"]
    assert "difference-in-differences" in did["estimand_note"]
    # delta (inner_R5 - inner_R6) differs from the legacy Obar-only delta because O_intra now matters
    assert abs(did["mean_delta_R5_minus_R6"] - did["mean_delta_obar_R5_minus_R6"]) > 1e-9
    print(f"[3a] paired_did honest_did=True; delta={did['mean_delta_R5_minus_R6']:.4f} "
          f"!= obar-delta={did['mean_delta_obar_R5_minus_R6']:.4f}")

    # If one arm falls back to O_intra=1, the honest flag must drop (mixed inputs are not an honest DiD).
    sum_R5_legacy = {}
    for seed in range(5):
        c5, _ = _feat_dicts(aug_scale=0.8, seed=200 + seed)
        sum_R5_legacy[seed] = h3.overlap_summaries(c5, layers=layers, cka_fn=_toy_cka)
    did_mixed = h3.paired_did(sum_R6, sum_R5_legacy)
    assert did_mixed["honest_did"] is False
    assert "NOT a true difference-in-differences" in did_mixed["estimand_note"]
    print("[3b] mixed (one arm fallback) -> honest_did=False, note downgraded")


if __name__ == "__main__":
    test_fallback_is_legacy_overlap_contrast()
    test_honest_did_has_real_intra()
    test_paired_did_flags_honest_only_when_both_arms_baselined()
    print("\nALL HONEST-DiD CPU TESTS PASSED")
