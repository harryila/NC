"""
Stage-3 analysis (HARDENED) — turn results/*.json into the decisive R6-R5 synchrony increment
and the pre-registered GREENLIGHT / PIVOT gate call.

This extends the original analyze.py (its exact paired sign-flip permutation test, TOST equivalence,
paired Cohen's d, --demo, and the GREENLIGHT/PIVOT decide() are kept verbatim) and adds the three
hardening pieces required by preregistration.md:

  (1) PLASTICITY GUARD  -- a forgetting win is only real if R6 LEARNED the tasks as well as R5.
      Using each run's per-seed LEARNING accuracy (top-level 'learning_acc' written by the patched
      run_split_cifar100; reconstructed from get_last_metrics() if that field is absent), TOST-test
      that R6's mean learning-acc is EQUIVALENT to R5 within +-Delta_p, AND that R6 is not LOWER
      than R5 by more than Delta_p (one-sided non-inferiority). An H2 GREENLIGHT is INVALIDATED
      if R6 underfit (a forgetting win bought by learning less).

  (2) MULTIPLICITY      -- the confirmatory family is {decisive contrast} x {primary metric} x
      {primary scenario}, instantiated here as the R6 - R5 permutation test across the three locked
      R5 brackets (depthwise[primary], no_proj, frozen_J). Holm controls FWER WITHIN that family.
      For the GREENLIGHT CONJUNCTION (R6<R5 AND replication AND plasticity-ok AND H3) we use an
      INTERSECTION-UNION TEST (IUT): every component must individually pass at alpha; the conjunction
      p is the MAX of the component p-values (no alpha-splitting needed for an IUT).

  (3) H3 FOLD-IN        -- read the 'h3' block from the results json (per-seed overlap summaries
      and/or phase-cluster stability). When per-seed summaries are present we recompute the
      seed-paired difference-in-differences (h3.paired_did) and fold its SIGN + one-sided p into
      the GREENLIGHT IUT; otherwise we degrade gracefully to "H3 unavailable" and surface it in
      the verdict without silently passing the component.

The final gate.call() combines: permutation p (Holm-adjusted within family), effect size,
plasticity guard, replication sign, and (when present) the H3 paired-contrast sign -- exactly the
pre-registered GREENLIGHT conjunction.

Everything is CPU-runnable: `python analyze_hardened.py --demo` exercises the original stats AND the
new guards (plasticity TOST, Holm, IUT, H3 fold-in) on synthetic data, with both a should-GREENLIGHT
and a should-INVALIDATE (R6 underfit) scenario.
"""
import argparse
import glob
import json
import os
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Pre-registered constants (preregistration.md "Numbers" / "Decision rule"). [locked]
ALPHA = 0.05
DELTA_G = 3.0          # SESOI / GREENLIGHT effect on forgetting (abs pts) OR |d|>=0.8
DELTA_E = 1.5          # equivalence margin for PIVOT nulls (TOST), pts
R5_BRACKETS = ["R5:depthwise", "R5:no_proj", "R5:frozen_J"]   # depthwise is the locked PRIMARY
PRIMARY_R5 = "R5:no_proj"   # param-identical apply_proj flip = cleanest causal contrast (depthwise/frozen_J = -39%-param robustness brackets)
FORGETTING_KEY = "StreamForgetting/eval_phase/test_stream"    # confirmed class-IL stream key


# =====================================================================================
# statistics (UNCHANGED from analyze.py -- the runnable, pre-registered core)
# =====================================================================================
def paired_permutation_p(diffs, n_perm=200_000, seed=0):
    """Two-sided exact sign-flip test if n<=22, else Monte-Carlo. diffs = per-seed (B - A)."""
    d = np.asarray(diffs, float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 22:
        cnt = tot = 0
        for signs in product((1, -1), repeat=n):
            tot += 1
            if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
                cnt += 1
        return cnt / tot
    rng = np.random.default_rng(seed)
    signs = rng.choice((1, -1), size=(n_perm, n))
    return float((np.abs((signs * d).mean(1)) >= obs).mean())


def tost(diffs, margin):
    """Two one-sided t-tests. Returns p; equivalent within +-margin if p < alpha."""
    from scipy import stats
    d = np.asarray(diffs, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((m - (-margin)) / se, n - 1)   # H0: mean <= -margin
    p_upper = stats.t.cdf((m - margin) / se, n - 1)     # H0: mean >=  margin
    return float(max(p_lower, p_upper))


def paired_cohens_d(diffs):
    d = np.asarray(diffs, float)
    return float(d.mean() / d.std(ddof=1))


def decide(diffs, delta_g, delta_e, alpha=0.05):
    """diffs = per-seed (R6 - R5) on the primary endpoint (lower forgetting is better,
    so a BENEFICIAL synchrony effect is diffs < 0; pass diffs already signed that way).

    UNCHANGED single-contrast call (kept for back-compat with analyze.py and --demo). The
    multiplicity-corrected, plasticity-guarded GREENLIGHT lives in gate()."""
    p_perm = paired_permutation_p(diffs)
    d = paired_cohens_d(diffs)
    p_eq = tost(diffs, delta_e)
    mean = float(np.mean(diffs))
    # SESOI: a big absolute effect OR a reliable standardized effect.
    # FIX (2026-05-31, prereg Amendment 1): the |d|>=0.8 branch is now gated behind a raw-effect
    # floor of Delta_e, so it can no longer rubber-stamp a sub-equivalence-margin effect under tiny
    # sd (the front-load degeneracy: at sd~1.08, |d|=0.8 corresponds to only 0.86 pt -- a "benefit"
    # that simultaneously sits inside the +-Delta_e "~null" band). A GREENLIGHT now requires the
    # effect to CLEAR the equivalence band, not merely be variance-normalized-large.
    big_magnitude = mean <= -delta_g
    reliable_and_material = (abs(d) >= 0.8) and (mean <= -delta_e)
    if (big_magnitude or reliable_and_material) and p_perm < alpha:
        call = "GREENLIGHT (synchrony reduces forgetting)"
    elif p_eq < alpha:
        call = "PIVOT-A (synchrony ~= geometry -- equivalence)"
    else:
        call = "INCONCLUSIVE (add seeds)"
    return {"mean_R6_minus_R5": mean, "cohens_d": d, "perm_p": p_perm,
            "tost_p": p_eq, "call": call,
            "sesoi_path": ("magnitude" if big_magnitude else
                           ("d_clause_material" if reliable_and_material else "none"))}


# =====================================================================================
# multiplicity helpers (NEW)
# =====================================================================================
def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down FWER control over a family of p-values.

    pvals : dict {label -> p} (the confirmatory family: one R6-R5 permutation p per R5 bracket).
    Returns dict {label -> {'p','p_holm','reject','rank','thresh'}} where p_holm is the
    Holm-ADJUSTED p (monotone, comparable to a single alpha) and reject is the step-down decision.
    A label rejects iff its sorted p clears alpha/(m-rank) AND all earlier (smaller) p's did too.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    prior_ok = True
    for rank, (label, p) in enumerate(items):           # rank = 0..m-1
        thresh = alpha / (m - rank)
        p_adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, p_adj)            # enforce monotone non-decreasing adjusted p
        reject = prior_ok and (p <= thresh)
        prior_ok = prior_ok and reject                   # step-down: stop rejecting after first failure
        out[label] = {"p": float(p), "p_holm": float(running_max),
                      "reject": bool(reject), "rank": rank, "thresh": float(thresh)}
    return out


def intersection_union(components, alpha=0.05):
    """Intersection-union test for a CONJUNCTION (all sub-nulls must be rejected).

    components : list of dicts, each {'name', 'p', 'directional_ok'(bool), 'available'(bool)}.
      - 'p'              : one-sided p for that component's null (smaller => stronger evidence).
      - 'directional_ok' : the pre-registered direction/sign constraint (e.g. R6 < R5; DiD > 0).
      - 'available'      : False => component data missing -> conjunction CANNOT pass (no silent pass).
    IUT rule: reject the global null (=> conjunction PASSES) iff EVERY component rejects at the FULL
    alpha (no alpha-splitting) AND every directional constraint holds AND every component is available.
    The conjunction p is the MAX of the component p's (valid for an IUT)."""
    avail = all(c.get("available", True) for c in components)
    dir_ok = all(c.get("directional_ok", True) for c in components)
    ps = [c["p"] for c in components if c.get("p") is not None]
    p_conj = max(ps) if ps else 1.0
    passed = bool(avail and dir_ok and all((c["p"] is not None and c["p"] <= alpha) for c in components))
    return {"pass": passed, "p_conjunction": float(p_conj),
            "all_available": bool(avail), "all_directional_ok": bool(dir_ok),
            "components": components}


# =====================================================================================
# plasticity guard (NEW) -- preregistration.md decision rule + Guards
# =====================================================================================
def plasticity_guard(learn_R6, learn_R5, margin=DELTA_E, alpha=ALPHA):
    """Pre-registered plasticity guard: a forgetting win must NOT be bought by underfitting.

    learn_R6 / learn_R5 : dict {seed -> learning-accuracy in POINTS} (mean_k A[k,k]).
    diffs := per-seed (R6 - R5) learning-acc. We require BOTH:
      (a) EQUIVALENCE within +-margin   : symmetric TOST p < alpha  (R6 ~= R5 plasticity), AND
      (b) NON-INFERIORITY               : R6 not lower than R5 by more than `margin`
                                          (one-sided lower TOST: H0 mean <= -margin rejected).
    The guard 'holds' iff non-inferiority passes (the load-bearing direction: R6 must not learn
    WORSE). Equivalence is reported too; a GREENLIGHT additionally wants full equivalence so that
    R6 is not merely non-inferior but indistinguishable in plasticity.
    Returns the diffs, both p-values, the mean gap, and the two booleans.
    """
    seeds = sorted(set(learn_R6) & set(learn_R5))
    if len(seeds) < 2:
        return {"seeds": seeds, "available": False, "holds": False, "equivalent": False,
                "inferior": False, "reason": "need >=2 paired seeds with learning_acc",
                "mean_R6_minus_R5": float("nan"), "tost_p": None, "noninf_p": None, "inferior_p": None}
    from scipy import stats
    diffs = np.array([learn_R6[s] - learn_R5[s] for s in seeds], float)
    n = len(diffs)
    m = float(diffs.mean())
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    if se == 0:                                          # degenerate (identical or n too small)
        p_lower = 0.0 if m > -margin else 1.0
        p_upper = 0.0 if m < margin else 1.0
        inferior_p = 0.0 if m < -margin else 1.0
    else:
        p_lower = float(stats.t.sf((m - (-margin)) / se, n - 1))   # H0: mean <= -margin -> reject => NON-INFERIOR
        p_upper = float(stats.t.cdf((m - margin) / se, n - 1))     # H0: mean >=  margin -> reject => R6 better
        inferior_p = float(stats.t.cdf((m - (-margin)) / se, n - 1))  # H0: mean >= -margin -> reject => AFFIRMATIVELY inferior
    tost_p = max(p_lower, p_upper)                       # equivalence p (symmetric TOST)
    noninf_p = p_lower                                   # non-inferiority: reject "R6 worse by >=margin"
    holds = bool(noninf_p < alpha)                       # guard "holds" = NON-INFERIORITY established
    equivalent = bool(tost_p < alpha)
    # FIX (2026-05-31): INVALIDATION must require AFFIRMATIVE inferiority, not merely "holds=False".
    # holds=False is failure-to-ESTABLISH non-inferiority (absence of evidence); it is NOT the same as
    # established inferiority. `inferior` rejects H0: mean >= -margin (R6 demonstrably worse by >margin).
    inferior = bool(inferior_p < alpha)
    return {"seeds": seeds, "available": True,
            "mean_R6_minus_R5": m, "diffs": diffs.tolist(),
            "tost_p": float(tost_p), "noninf_p": float(noninf_p), "inferior_p": float(inferior_p),
            "margin": float(margin), "holds": holds, "equivalent": equivalent, "inferior": inferior,
            "interpretation": ("holds=non-inferiority established (R6 not worse by >margin); "
                               "inferior=affirmative inferiority established (R6 worse by >margin -> INVALIDATES); "
                               "neither holds nor inferior => CONFOUNDED-INCONCLUSIVE (underpowered/ambiguous). "
                               "equivalent=symmetric TOST. GREENLIGHT requires holds.")}


# =====================================================================================
# results loading
# =====================================================================================
def _forgetting(metrics):
    """Stream forgetting in POINTS (x100). Class-IL key 'StreamForgetting/eval_phase/test_stream';
    falls back to any stream-level then any forgetting scalar. (UNCHANGED from analyze.py.)"""
    if isinstance(metrics, dict):
        if isinstance(metrics.get(FORGETTING_KEY), (int, float)):
            return 100.0 * float(metrics[FORGETTING_KEY])
        for k, v in metrics.items():
            if "streamforgetting" in k.lower() and isinstance(v, (int, float)):
                return 100.0 * float(v)
        for k, v in metrics.items():
            if "forget" in k.lower() and isinstance(v, (int, float)):
                return 100.0 * float(v)
    return None


def _to_points(x):
    """Normalize an accuracy to POINTS. Avalanche reports fractions in [0,1]; if a value is <=1.5
    we treat it as a fraction and x100, else assume it is already in points."""
    if x is None:
        return None
    x = float(x)
    return 100.0 * x if x <= 1.5 else x


def _learning_acc(run):
    """Per-run LEARNING accuracy in POINTS = mean_k A[k,k] (accuracy on task k right after training k).

    Prefer the explicit top-level 'learning_acc' field written by the patched run_split_cifar100
    (a fraction or points scalar, OR a per-task list whose mean we take). If absent, RECONSTRUCT it
    from get_last_metrics() using the avalanche identity  A[k,k] = A[T-1,k] + ExperienceForgetting[k]
    (forgetting = initial - last => initial = last + forgetting); the last-trained task T-1 carries no
    forgetting entry, so its learning-acc == its final accuracy A[T-1,T-1]. This makes the guard
    computable even on the current results that only logged get_last_metrics()."""
    # 1) explicit field (preferred; the patched backbone returns this alongside metrics)
    la = run.get("learning_acc")
    if la is not None:
        if isinstance(la, (list, tuple)) and len(la):
            return _to_points(float(np.mean([float(v) for v in la])))
        if isinstance(la, dict) and la:
            return _to_points(float(np.mean([float(v) for v in la.values()])))
        if isinstance(la, (int, float)):
            return _to_points(la)
    # 1b) sometimes nested inside metrics
    m = run.get("metrics", {})
    if isinstance(m, dict) and isinstance(m.get("learning_acc"), (int, float)):
        return _to_points(m["learning_acc"])
    # 2) reconstruct from the final-row accuracies + per-experience forgetting
    if not isinstance(m, dict):
        return None
    tj = 0 if run.get("scenario", "class") != "task" else None    # class-IL: all task labels 0
    final_acc, forget = {}, {}
    for k, v in m.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith("Top1_Acc_Exp/") and "/Exp" in k:
            j = int(k.rsplit("Exp", 1)[1])
            final_acc[j] = float(v)
        elif k.startswith("ExperienceForgetting/") and "/Exp" in k:
            j = int(k.rsplit("Exp", 1)[1])
            forget[j] = float(v)
    if not final_acc:
        return None
    diag = []
    for j, a_final in final_acc.items():
        # A[j,j] = final_acc[j] + forgetting[j]; last-trained task has no forgetting entry (== final).
        diag.append(a_final + forget.get(j, 0.0))
    return _to_points(float(np.mean(diag)))


def load(primary_scenario="class", primary_nexp=10, results_dir=RESULTS):
    """Load runs into nested dicts keyed by rung then seed.

    Returns (forget, learn, h3) where each is {rung: {seed: value}}:
      forget[rung][seed] -> stream forgetting in POINTS,
      learn[rung][seed]  -> learning accuracy in POINTS (explicit or reconstructed),
      h3[rung][seed]     -> the raw 'h3' block from the json (overlap summary / phase stability) or None.
    Replication (20x5) is loaded separately by passing primary_nexp=20.
    """
    forget, learn, h3 = {}, {}, {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        if os.path.basename(f) == "sparsity_target.json":
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("scenario") == primary_scenario and r.get("nexp") == primary_nexp:
            rung, seed = r.get("rung"), r.get("seed")
            forget.setdefault(rung, {})[seed] = _forgetting(r.get("metrics", {}))
            learn.setdefault(rung, {})[seed] = _learning_acc(r)
            h3.setdefault(rung, {})[seed] = r.get("h3")
    return forget, learn, h3


# =====================================================================================
# H3 fold-in (NEW)
# =====================================================================================
def h3_contrast(h3_R6, h3_R5, alpha=ALPHA):
    """Fold the H3 mechanism evidence into the gate.

    h3_R6 / h3_R5 : {seed -> h3-block}. We accept either of two pre-registered shapes:
      (a) per-seed overlap_summaries() dicts (with an 'inner'/'Obar' field) -> recompute the
          seed-paired difference-in-differences via h3.paired_did (one-sided delta>0: R6 has LOWER
          cross-task overlap). The DiD SIGN + p are folded into the GREENLIGHT IUT.
      (b) only a precomputed scalar DiD summary (e.g. {'mean_delta_R5_minus_R6','p_one_sided_delta_gt_0'})
          -> use it directly.
    Returns {'available','mean_delta_R5_minus_R6','p_one_sided','directional_ok',...}. directional_ok
    is the pre-registered prediction: delta>0 (overlap reduction present in R6, the DiD positive)."""
    # FIX (2026-05-31): the driver writes each seed's block as
    #   {"overlap_summary": {...O_inter, inner, Obar...}, "phase_stability": ...}
    # but _is_summary/paired_did expect the BARE overlap-summary (inner/Obar at top level).
    # Unwrap the nested driver shape; pass through bare summaries or precomputed-DiD blocks.
    def _unwrap_overlap(blocks):
        if not isinstance(blocks, dict):
            return blocks
        out = {}
        for _s, _b in blocks.items():
            if isinstance(_b, dict) and isinstance(_b.get("overlap_summary"), dict):
                out[_s] = _b["overlap_summary"]
            else:
                out[_s] = _b
        return out
    h3_R6 = _unwrap_overlap(h3_R6)
    h3_R5 = _unwrap_overlap(h3_R5)
    seeds = sorted(set(k for k in (h3_R6 or {}) if h3_R6.get(k) is not None)
                   & set(k for k in (h3_R5 or {}) if h3_R5.get(k) is not None))
    if not seeds:
        return {"available": False, "reason": "no paired H3 blocks", "directional_ok": False,
                "p_one_sided": None, "mean_delta_R5_minus_R6": float("nan")}
    # shape (a): per-seed overlap summaries -> recompute the paired DiD with the real h3 module.
    def _is_summary(b):
        return isinstance(b, dict) and ("inner" in b or "Obar" in b)
    if all(_is_summary(h3_R6[s]) for s in seeds) and all(_is_summary(h3_R5[s]) for s in seeds):
        try:
            import h3 as h3mod
            sR6 = {s: h3_R6[s] for s in seeds}
            sR5 = {s: h3_R5[s] for s in seeds}
            did = h3mod.paired_did(sR6, sR5)
            mean_delta = did["mean_delta_R5_minus_R6"]
            p = did["p_one_sided_delta_gt_0"]
        except Exception as e:                            # h3 not importable / degenerate -> fall back
            inner_R6 = np.array([float(h3_R6[s].get("inner", h3_R6[s].get("Obar", np.nan))) for s in seeds])
            inner_R5 = np.array([float(h3_R5[s].get("inner", h3_R5[s].get("Obar", np.nan))) for s in seeds])
            delta = inner_R5 - inner_R6
            mean_delta = float(np.nanmean(delta))
            sd = np.nanstd(delta, ddof=1) if len(delta) > 1 else 0.0
            p = float(paired_permutation_p(delta)) if sd > 0 else (0.0 if mean_delta > 0 else 1.0)
            did = {"note": f"h3.paired_did unavailable ({e}); used numpy fallback"}
        return {"available": True, "seeds": seeds, "mean_delta_R5_minus_R6": float(mean_delta),
                "p_one_sided": float(p), "directional_ok": bool(mean_delta > 0 and p < alpha),
                "detail": did,
                "interpretation": "delta>0 & p<alpha => R6 reduces inter-task overlap more than R5 (H3 holds)"}
    # shape (b): precomputed scalar DiD summaries -> average across seeds, take min p.
    deltas, ps = [], []
    for s in seeds:
        b = h3_R6[s] if isinstance(h3_R6[s], dict) and "mean_delta_R5_minus_R6" in h3_R6[s] else h3_R5[s]
        if isinstance(b, dict):
            if "mean_delta_R5_minus_R6" in b:
                deltas.append(float(b["mean_delta_R5_minus_R6"]))
            if "p_one_sided_delta_gt_0" in b:
                ps.append(float(b["p_one_sided_delta_gt_0"]))
    if not deltas:
        return {"available": False, "reason": "H3 blocks present but unrecognized shape",
                "directional_ok": False, "p_one_sided": None, "mean_delta_R5_minus_R6": float("nan")}
    mean_delta = float(np.mean(deltas))
    p = float(min(ps)) if ps else (0.0 if mean_delta > 0 else 1.0)
    return {"available": True, "seeds": seeds, "mean_delta_R5_minus_R6": mean_delta,
            "p_one_sided": p, "directional_ok": bool(mean_delta > 0 and p < alpha),
            "interpretation": "precomputed DiD summary folded in (delta>0 & p<alpha => H3 holds)"}


# =====================================================================================
# the HARDENED gate (NEW) -- combines all components into the pre-registered verdict
# =====================================================================================
def gate(forget, learn, h3blocks, forget_rep=None,
         r5_brackets=R5_BRACKETS, primary_r5=PRIMARY_R5, h3_control=None,
         a5_rung="A5:derpp", a5_margin=3.0, positive_control_pass=None,
         delta_g=DELTA_G, delta_e=DELTA_E, alpha=ALPHA):
    """Full pre-registered gate.

    forget/learn/h3blocks : {rung:{seed:val}} for the PRIMARY scenario (class-IL 10x10).
    forget_rep            : {rung:{seed:val}} for the REPLICATION stream (20x5); sign-replication only.

    Steps:
      1. Per-bracket R6-R5 forgetting effect (decide()) -> permutation p per bracket.
      2. MULTIPLICITY: Holm over the {3 brackets} confirmatory family on those permutation p's.
      3. PLASTICITY GUARD on the PRIMARY bracket (must hold; full equivalence wanted for GREENLIGHT).
      4. REPLICATION: sign of mean(R6-R5) on the 20x5 stream must match (negative == benefit).
      5. H3: seed-paired DiD sign (delta>0) folded in when available.
      6. GREENLIGHT IUT over {forgetting effect, replication sign, plasticity-ok, H3 DiD} on the
         PRIMARY bracket; each component at full alpha (intersection-union).
    Returns a verdict dict.
    """
    if "R6" not in forget:
        return {"error": "no R6 runs found", "have": sorted(forget)}
    if h3_control is None:
        h3_control = primary_r5   # H3 DiD needs the param-identical control (== primary when primary is no_proj)

    # ---- 1. per-bracket forgetting effect + permutation p ----
    per_bracket = {}
    family_p = {}
    for b in r5_brackets:
        if b not in forget:
            continue
        seeds = sorted(s for s in (set(forget["R6"]) & set(forget[b]))
                       if forget["R6"].get(s) is not None and forget[b].get(s) is not None)
        if len(seeds) < 2:
            per_bracket[b] = {"seeds": seeds, "skipped": "need >=2 paired seeds"}
            continue
        diffs = [forget["R6"][s] - forget[b][s] for s in seeds]    # benefit => negative
        dec = decide(diffs, delta_g, delta_e, alpha)
        dec["seeds"] = seeds
        dec["n"] = len(seeds)
        per_bracket[b] = dec
        family_p[b] = dec["perm_p"]

    # ---- 2. multiplicity: Holm within the confirmatory family ----
    holm_res = holm(family_p, alpha) if family_p else {}

    # ---- 3. plasticity guard on the PRIMARY bracket ----
    plast = {"available": False, "holds": False, "inferior": False}
    if primary_r5 in learn and "R6" in learn:
        lr6 = {s: v for s, v in learn["R6"].items() if v is not None}
        lr5 = {s: v for s, v in learn[primary_r5].items() if v is not None}
        plast = plasticity_guard(lr6, lr5, margin=delta_e, alpha=alpha)

    # ---- 3b. SATURATION-CONFOUND diagnostic (front-load finding): under a head-saturated metric,
    # forgetting becomes a learning proxy. If the per-seed forgetting-diff and learning-diff are
    # highly correlated, any forgetting "benefit" is confounded by a plasticity difference (r=0.996
    # on the front-load). Reported so a CONFOUNDED-INCONCLUSIVE call is auditable, not asserted.
    confound_r = None
    if primary_r5 in forget and "R6" in forget and primary_r5 in learn and "R6" in learn:
        cs = sorted(s for s in (set(forget["R6"]) & set(forget[primary_r5])
                                & set(learn["R6"]) & set(learn[primary_r5]))
                    if None not in (forget["R6"].get(s), forget[primary_r5].get(s),
                                    learn["R6"].get(s), learn[primary_r5].get(s)))
        if len(cs) >= 3:
            fd = np.array([forget["R6"][s] - forget[primary_r5][s] for s in cs], float)
            ld = np.array([learn["R6"][s] - learn[primary_r5][s] for s in cs], float)
            if fd.std() > 0 and ld.std() > 0:
                confound_r = float(np.corrcoef(fd, ld)[0, 1])

    # ---- 4. replication (sign only) on the longer stream ----
    rep = {"available": False, "sign_replicates": False}
    if forget_rep and "R6" in forget_rep and primary_r5 in forget_rep:
        rseeds = sorted(s for s in (set(forget_rep["R6"]) & set(forget_rep[primary_r5]))
                        if forget_rep["R6"].get(s) is not None and forget_rep[primary_r5].get(s) is not None)
        if len(rseeds) >= 2:
            rdiffs = [forget_rep["R6"][s] - forget_rep[primary_r5][s] for s in rseeds]
            rmean = float(np.mean(rdiffs))
            primary_mean = per_bracket.get(primary_r5, {}).get("mean_R6_minus_R5", float("nan"))
            same_sign = bool(np.sign(rmean) == np.sign(primary_mean) and rmean < 0)
            rep = {"available": True, "seeds": rseeds, "mean_R6_minus_R5": rmean,
                   "sign_replicates": same_sign,
                   "perm_p": paired_permutation_p(rdiffs)}

    # ---- 5. H3 fold-in (primary bracket vs R6) ----
    h3c = {"available": False, "directional_ok": False, "p_one_sided": None}
    if h3blocks and "R6" in h3blocks and h3_control in h3blocks:
        h3c = h3_contrast(h3blocks["R6"], h3blocks[h3_control], alpha)

    # ---- 5b. A5-competitiveness: R6 not WORSE than DER++ by more than a5_margin pts (one-sided NI) ----
    a5 = {"available": False, "directional_ok": False, "p": None}
    if a5_rung in forget and "R6" in forget:
        aseeds = sorted(s for s in (set(forget["R6"]) & set(forget[a5_rung]))
                        if forget["R6"].get(s) is not None and forget[a5_rung].get(s) is not None)
        if len(aseeds) >= 2:
            adiffs = np.array([forget["R6"][s] - forget[a5_rung][s] for s in aseeds], float)   # R6 - A5
            m = float(adiffs.mean()); se = adiffs.std(ddof=1) / np.sqrt(len(adiffs))
            try:
                from scipy import stats
                p_a5 = float(stats.t.cdf((m - a5_margin) / se, len(adiffs) - 1)) if se > 0 else (0.0 if m < a5_margin else 1.0)
            except Exception:
                p_a5 = 0.0 if m < a5_margin else 1.0
            a5 = {"available": True, "seeds": aseeds, "mean_R6_minus_A5": m,
                  "p": p_a5, "directional_ok": bool(p_a5 < alpha)}

    # ---- 6. GREENLIGHT intersection-union test on the primary bracket ----
    primary = per_bracket.get(primary_r5, {})
    # forgetting effect component: pre-registered direction (R6<R5) + SESOI; p = Holm-adjusted perm p.
    p_forget = holm_res.get(primary_r5, {}).get("p_holm", primary.get("perm_p"))
    sesoi_ok = bool((primary.get("mean_R6_minus_R5", 0) <= -delta_g)
                    or (abs(primary.get("cohens_d", 0)) >= 0.8))
    forget_ok_dir = bool(primary.get("mean_R6_minus_R5", 0) < 0)   # benefit direction

    components = [
        {"name": "forgetting_effect(R6<R5, Holm)", "p": p_forget,
         "directional_ok": forget_ok_dir and sesoi_ok, "available": bool(primary and "perm_p" in primary)},
        {"name": "replication_sign(20x5)", "p": rep.get("perm_p") if rep["available"] else None,
         "directional_ok": rep["sign_replicates"], "available": rep["available"]},
        {"name": "plasticity_guard(non-inferior)", "p": plast.get("noninf_p"),
         "directional_ok": plast.get("holds", False), "available": plast.get("available", False)},
        {"name": "H3_DiD(delta>0)", "p": h3c.get("p_one_sided"),
         "directional_ok": h3c.get("directional_ok", False), "available": h3c.get("available", False)},
        {"name": f"A5_competitive(R6<=A5+{a5_margin}pts)", "p": a5.get("p"),
         "directional_ok": a5.get("directional_ok", False), "available": a5.get("available", False)},
    ]
    iut = intersection_union(components, alpha)

    # ---- final call ----
    # R6 ~= R1 (no CL benefit) equivalence -> PIVOT-B candidate
    r6_eq_r1 = None
    if "R1" in forget and "R6" in forget:
        p1 = sorted(s for s in (set(forget["R6"]) & set(forget["R1"]))
                    if forget["R6"].get(s) is not None and forget["R1"].get(s) is not None)
        if len(p1) >= 2:
            r6_eq_r1 = tost([forget["R6"][s] - forget["R1"][s] for s in p1], delta_e) < alpha
    pc_ok = bool(positive_control_pass)   # prereg Guard: positive control MUST pass before any null is declared

    forget_beneficial = bool(primary.get("mean_R6_minus_R5", 0) < 0)   # R6 forgets less (benefit direction)
    if iut["pass"]:
        call = "GREENLIGHT M2 (synchrony reduces forgetting; conjunction holds)"
    elif plast.get("available") and plast.get("inferior"):
        # AFFIRMATIVE inferiority (established, not just NI-not-shown): the forgetting "win" is bought
        # by R6 learning worse by >margin. (FIX: was firing on `not holds`, which over-claimed underfit.)
        call = "INVALIDATED (R6 affirmatively underfit: learns worse than R5 by >margin -- forgetting 'win' bought by reduced plasticity)"
    elif forget_beneficial and plast.get("available") and not plast.get("holds"):
        # The apparent benefit cannot be separated from a plasticity difference the guard could neither
        # clear (non-inferiority not established) nor condemn (inferiority not established). Under
        # saturation this is the expected state (forgetting ~ learning); confound_r quantifies it.
        call = ("CONFOUNDED-INCONCLUSIVE (apparent forgetting benefit not separable from a plasticity "
                "difference -- guard neither clears nor condemns R6"
                + (f"; forgetting~learning collinearity r={confound_r:.3f}" if confound_r is not None else "")
                + ")")
    elif pc_ok and r6_eq_r1:
        call = "PIVOT-B (no CL benefit: R6 ~= R1; positive control passed)"
    elif pc_ok and primary.get("tost_p") is not None and primary["tost_p"] < alpha:
        call = "PIVOT-A (synchrony ~= geometry; positive control passed)"
    elif primary.get("tost_p") is not None and primary["tost_p"] < alpha:
        call = "PIVOT-A-PENDING (equivalence holds, but positive control not yet passed -> not declarable)"
    else:
        call = "INCONCLUSIVE (conjunction incomplete; add seeds / collect H3 / replication / positive control)"

    return {"call": call,
            "primary_bracket": primary_r5, "h3_control": h3_control,
            "a5_competitive": a5, "positive_control_pass": pc_ok,
            "confound_r": confound_r,
            "per_bracket": per_bracket,
            "holm_family": holm_res,
            "plasticity_guard": plast,
            "replication": rep,
            "h3": h3c,
            "greenlight_iut": iut}


# =====================================================================================
# DEMO  (CPU, synthetic -- exercises the ORIGINAL stats AND the new guards)
# =====================================================================================
def _synth_h3_summary(rng, off_mean, off_sd, n_tasks=5, layers=(0, 1, 2)):
    """A minimal overlap_summaries()-shaped dict (only the fields h3_contrast reads)."""
    from itertools import combinations
    per_layer = {}
    for l in layers:
        off = [float(np.clip(rng.normal(off_mean, off_sd), 0, 1))
               for _ in combinations(range(n_tasks), 2)]
        per_layer[l] = float(np.mean(off))
    o_inter = float(np.mean(list(per_layer.values())))
    return {"per_layer_inter": per_layer, "O_inter": o_inter, "O_intra": 1.0,
            "Obar": o_inter, "inner": o_inter - 1.0}


def _demo():
    rng = np.random.default_rng(0)

    print("=== analyze_hardened DEMO (synthetic; CPU) ===\n")

    # --- (0) original single-contrast decide() (unchanged behavior) ---
    diffs = -3.5 + rng.normal(0, 2.0, size=12)        # R6 ~3.5 pts lower forgetting, n=12
    print("[0] original decide() (R6-R5 forgetting diffs):")
    print("   ", np.round(diffs, 2))
    print(json.dumps(decide(diffs, DELTA_G, DELTA_E), indent=2))

    # --- (1) Holm over the confirmatory family ---
    print("\n[1] Holm over the 3-bracket confirmatory family:")
    fam = {"R5:depthwise": 0.004, "R5:no_proj": 0.02, "R5:frozen_J": 0.30}
    print(json.dumps(holm(fam, ALPHA), indent=2))

    # --- (2) plasticity guard: PASS scenario (R6 learns as well as R5) ---
    n = 12
    learn_R5 = {s: 72.0 + rng.normal(0, 1.5) for s in range(n)}
    learn_R6_ok = {s: learn_R5[s] + rng.normal(0, 1.2) for s in range(n)}      # ~equal plasticity
    g_ok = plasticity_guard(learn_R6_ok, learn_R5, margin=DELTA_E)
    print("\n[2a] plasticity guard -- R6 learns as well as R5 (should HOLD):")
    print(json.dumps({k: g_ok[k] for k in ["mean_R6_minus_R5", "tost_p", "noninf_p", "holds", "equivalent"]}, indent=2))

    # plasticity guard: FAIL scenario (R6 underfit by ~4 pts -> guard must NOT hold)
    learn_R6_bad = {s: learn_R5[s] - 4.0 + rng.normal(0, 1.0) for s in range(n)}
    g_bad = plasticity_guard(learn_R6_bad, learn_R5, margin=DELTA_E)
    print("\n[2b] plasticity guard -- R6 UNDERFIT by ~4 pts (should FAIL: holds=False):")
    print(json.dumps({k: g_bad[k] for k in ["mean_R6_minus_R5", "tost_p", "noninf_p", "holds", "equivalent"]}, indent=2))

    # --- (3) build full {rung:{seed:val}} stores and run the gate end-to-end ---
    seeds = list(range(n))
    # PRIMARY (class10) forgetting: R6 ~3 pts below each R5 bracket; A5 (DER++) within ~1 pt of R6
    forget = {"R6": {}, "R5:depthwise": {}, "R5:no_proj": {}, "R5:frozen_J": {}, "A5:derpp": {}, "R1": {}}
    for s in seeds:
        base = 19.0 + rng.normal(0, 1.0)
        forget["R6"][s] = base
        forget["R5:depthwise"][s] = base + 3.2 + rng.normal(0, 1.2)
        forget["R5:no_proj"][s] = base + 2.6 + rng.normal(0, 1.2)     # param-identical PRIMARY control
        forget["R5:frozen_J"][s] = base + 0.4 + rng.normal(0, 1.2)    # weak bracket (Holm should drop it)
        forget["A5:derpp"][s] = base + 0.8 + rng.normal(0, 1.0)       # DER++ competitive (within 3 pts)
        forget["R1"][s] = base + 4.0 + rng.normal(0, 1.2)            # dense floor (for PIVOT-B contrast)
    # learning-acc: PASS plasticity for the primary bracket
    learn = {"R6": learn_R6_ok, "R5:depthwise": learn_R5,
             "R5:no_proj": dict(learn_R5), "R5:frozen_J": dict(learn_R5)}
    # H3 per-seed overlap summaries on the PRIMARY (no_proj) control: R6 LOWER inter-task overlap (DiD positive)
    h3blocks = {"R6": {}, "R5:no_proj": {}}
    for s in seeds:
        h3blocks["R6"][s] = _synth_h3_summary(rng, 0.55, 0.05)
        h3blocks["R5:no_proj"][s] = _synth_h3_summary(rng, 0.72, 0.05)
    # REPLICATION (class20): same sign (R6 below the no_proj primary)
    forget_rep = {"R6": {}, "R5:no_proj": {}}
    for s in seeds:
        b = 24.0 + rng.normal(0, 1.0)
        forget_rep["R6"][s] = b
        forget_rep["R5:no_proj"][s] = b + 2.8 + rng.normal(0, 1.3)

    print("\n[3] FULL GATE -- everything aligned (should GREENLIGHT):")
    v = gate(forget, learn, h3blocks, forget_rep)
    print("    call:", v["call"])
    print("    greenlight IUT pass:", v["greenlight_iut"]["pass"],
          "| components:",
          {c["name"]: (c["available"], c["directional_ok"], None if c["p"] is None else round(c["p"], 4))
           for c in v["greenlight_iut"]["components"]})
    print("    Holm primary p_holm:", round(v["holm_family"]["R5:depthwise"]["p_holm"], 4),
          "reject:", v["holm_family"]["R5:depthwise"]["reject"])

    print("\n[4] FULL GATE -- R6 UNDERFIT (forgetting win, but plasticity fails -> INVALIDATED):")
    learn_bad = dict(learn); learn_bad["R6"] = learn_R6_bad
    v2 = gate(forget, learn_bad, h3blocks, forget_rep)
    print("    call:", v2["call"])
    print("    plasticity holds:", v2["plasticity_guard"]["holds"],
          "| IUT pass:", v2["greenlight_iut"]["pass"])

    print("\n[5] FULL GATE -- H3 missing (conjunction cannot pass -> not GREENLIGHT, no silent pass):")
    v3 = gate(forget, learn, h3blocks=None, forget_rep=forget_rep)
    print("    call:", v3["call"])
    print("    H3 available:", v3["h3"]["available"], "| IUT pass:", v3["greenlight_iut"]["pass"])

    # --- (6) the FRONT-LOAD pattern: R6 forgets less BUT also learns ~1pt less (within margin) ---
    # plasticity guard neither holds (NI not established) nor condemns (inferiority not established),
    # and forgetting~learning are collinear -> CONFOUNDED-INCONCLUSIVE (not INVALIDATED, not GREENLIGHT).
    print("\n[6] FULL GATE -- forgets-less-but-learns-less, gap inside margin (-> CONFOUNDED-INCONCLUSIVE):")
    # deterministic = the ACTUAL front-load learning diffs (mean -1.109, sd 1.070 -> NI p~0.139 not-holds,
    # inferiority p~0.861 not-inferior); reproduces the real CONFOUNDED-INCONCLUSIVE state, not flaky noise.
    real_ldiff = [-1.02, -3.28, -0.65, -1.65, -0.33, 0.45, 0.05, -1.82, -1.41, -1.43]
    learn_confound = {k: dict(v) for k, v in learn.items()}
    learn_confound["R6"] = {s: learn_R5[s] + real_ldiff[s] for s in range(len(real_ldiff))}
    v4 = gate(forget, learn_confound, h3blocks, forget_rep)   # H3 present, but plasticity won't hold -> IUT fails
    print("    call:", v4["call"])
    print("    plast holds:", v4["plasticity_guard"]["holds"], "| inferior:", v4["plasticity_guard"]["inferior"],
          "| confound_r:", None if v4["confound_r"] is None else round(v4["confound_r"], 3))

    print("\n=== DEMO OK ===")


def main():
    ap = argparse.ArgumentParser(description="Hardened Stage-3 gate (forgetting + plasticity + Holm/IUT + H3)")
    ap.add_argument("--demo", action="store_true", help="run the stats + guards on synthetic data (CPU)")
    ap.add_argument("--delta_g", type=float, default=DELTA_G, help="GREENLIGHT SESOI on forgetting (pts)")
    ap.add_argument("--delta_e", type=float, default=DELTA_E, help="equivalence / plasticity margin (pts)")
    ap.add_argument("--r5", default=PRIMARY_R5, help="which R5 bracket is the PRIMARY decisive control")
    ap.add_argument("--scenario", default="class")
    ap.add_argument("--nexp", type=int, default=10, help="primary scenario n_experiences (10x10)")
    ap.add_argument("--nexp_rep", type=int, default=20, help="replication stream n_experiences (20x5)")
    ap.add_argument("--results", default=RESULTS, help="results dir (defaults to ./results)")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return

    forget, learn, h3blocks = load(args.scenario, args.nexp, args.results)
    forget_rep, _, _ = load(args.scenario, args.nexp_rep, args.results)
    if "R6" not in forget or args.r5 not in forget:
        print("Need both R6 and", args.r5, "results. Have:", sorted(forget))
        return
    v = gate(forget, learn, h3blocks, forget_rep,
             primary_r5=args.r5, delta_g=args.delta_g, delta_e=args.delta_e)
    print(json.dumps(v, indent=2, default=str))


if __name__ == "__main__":
    main()
