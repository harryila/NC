"""M2 RELATIONAL-DESCRIPTOR PROBE — is the screen's null an artifact of a synchrony-blind descriptor?

THE HYPOTHESIS (why the marginal screen may have measured the wrong thing):
  The screen + the original viability pre-check both decode task from `_pool_phase_state` =
  [mean, mean-square] of the oscillator group-directions AVERAGED OVER ALL SITES. That is the
  MARGINAL phase distribution (~ the global order parameter). It is, by construction, invariant to
  the RELATIONAL structure between oscillators -- WHICH sites synchronize with WHICH -- which is
  exactly what synchrony IS (and exactly what M1's mechanism analysis uses: spherical k-means phase
  clusters + cross-task AMI in h3.phase_cluster_stability). So a real R6>R5 task channel carried by
  the relational structure would be INVISIBLE to the marginal descriptor.

THE TEST: capture the raw phase state xs[layer][-1] per sample for R6 (ON) and R5:no_proj (OFF),
  and decode task with SEVERAL descriptors of increasing relational richness, then compare the
  R6-vs-R5 gap per descriptor:
    D0 marginal      : [mean, meansq] over sites (2n)         <- the screen's descriptor (baseline)
    D1 second_moment : full (U^T U)/S  (n^2)                  <- adds CROSS-AXIS coherence D0 lacks
    D2 coh_eig       : eigenvalues of (U^T U)/S  (n)          <- coherence anisotropy spectrum
    D3 cluster_occ   : sorted spherical-kmeans(k) occupancy   <- RELATIONAL: how sites group (gauge-inv)
    D4 spatial4x4    : coarse spatial map of mean direction   <- spatial pattern of phase (relational)

PRE-SPECIFIED DECISION RULE (locked before results; primary contrast = R6 - R5:no_proj):
  * RESCUE  (descriptor was the flaw): some RELATIONAL descriptor (D1-D4) shows mean Delta >= +0.05
            AND clearly exceeds D0's gap (by >= 0.04). => synchrony's channel is real but was hidden
            by marginal pooling -> rebuild the screen + C_ctx context on that descriptor.
  * NULL ROBUST: every descriptor (incl relational) gives R6 ~ R5 (no Delta >= +0.05). => the null
            survives a synchrony-sensitive readout -> PIVOT-NULL is real, not a measurement artifact.
  * (single round; this is a decisive disambiguation, not an open-ended search.)

Usage (GPU box):
    python m2_relational_probe.py --seeds 0 1 --epochs 30 --layers 1 2 --device cuda
    python m2_relational_probe.py --demo     # CPU: descriptor shapes + that they separate structure
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
if M1 not in sys.path:
    sys.path.insert(0, M1)
RESULTS = os.path.join(HERE, "results")

ARMS = {"R6": {}, "R5:no_proj": {"variant": "no_proj"}}
DESCRIPTORS = ["marginal", "2nd_moment", "coh_eig", "cluster_occ", "spatial4x4"]
RELATIONAL = ["2nd_moment", "coh_eig", "cluster_occ", "spatial4x4"]
RESCUE_DELTA = 0.05        # a relational descriptor must beat R5 by >= this to count as "channel hidden"
RESCUE_OVER_MARGINAL = 0.04
PRIMARY_LAYER = 2


def _sample_descriptors(osc_1, n, k=8):
    """All descriptors for ONE sample's oscillator state osc_1:(1,C,H,W). Returns {name: 1-D np vec}."""
    import numpy as np
    from h3 import group_directions, spherical_kmeans
    out = {}
    U = group_directions(osc_1, n=n)                      # (n_sites, n) unit vectors
    # D0 marginal (the screen's descriptor)
    out["marginal"] = np.concatenate([U.mean(0), (U ** 2).mean(0)])
    # D1 full second-moment matrix (diagonal == meansq in D0; OFF-diagonal == cross-axis coherence)
    M = (U.T @ U) / U.shape[0]
    out["2nd_moment"] = M.flatten()
    # D2 coherence eigenspectrum (gauge-invariant anisotropy of the direction cloud)
    ev = np.linalg.eigvalsh(M)
    out["coh_eig"] = ev[::-1].copy()
    # D3 phase-cluster occupancy, sorted (gauge-invariant relational summary of how sites group)
    asg = spherical_kmeans(U, k=k, seed=0)
    occ = np.bincount(asg, minlength=k).astype(float)
    occ = occ / max(occ.sum(), 1.0)
    out["cluster_occ"] = np.sort(occ)[::-1].copy()
    # D4 coarse spatial map of the per-cell mean direction (preserves WHERE phases point; relational)
    a = np.asarray(osc_1, float)[0]                        # (C,H,W)
    C, H, W = a.shape
    G = C // n
    md = a.reshape(G, n, H, W).mean(0)                     # (n,H,W) mean direction per spatial cell
    ph = 4
    if H >= ph and W >= ph:
        hs, ws = H // ph, W // ph
        md = md[:, :ph * hs, :ph * ws].reshape(n, ph, hs, ph, ws).mean(axis=(2, 4))   # (n,ph,ph)
    out["spatial4x4"] = md.flatten()
    return out


def _train_and_capture(rung, rung_kw, n_tasks, epochs, layers, device, eval_inits, seed,
                       max_samples_per_task, k=8):
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa: F401
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import InteractiveLogger
    from avalanche_backbone import LadderClassifier
    from h3 import _seeded as h3_seeded
    from torch.utils.data import DataLoader, Subset

    base = rung.split(":")[0]
    bench = SplitCIFAR100(n_experiences=n_tasks, return_task_id=False, seed=seed)
    model = LadderClassifier(base, num_classes=100, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)
    evalp = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[InteractiveLogger()])
    cl = Naive(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
               criterion=nn.CrossEntropyLoss(), train_mb_size=128, eval_mb_size=100,
               train_epochs=epochs, evaluator=evalp, device=device)
    for exp in bench.train_stream:
        cl.train(exp)
    n = int(getattr(model.net, "n", 4))
    # desc[layer][descriptor_name] = {task: [vec,...]}
    desc = {l: {d: {} for d in DESCRIPTORS} for l in layers}
    model.eval()
    with torch.no_grad():
        for i, exp in enumerate(bench.test_stream):
            ds = exp.dataset
            idx = list(range(min(max_samples_per_task, len(ds))))
            loader = DataLoader(Subset(ds, idx), batch_size=100, shuffle=False)
            for l in layers:
                for d in DESCRIPTORS:
                    desc[l][d][i] = []
            for batch in loader:
                x = batch[0].to(device)
                acc = {l: None for l in layers}
                for j in range(eval_inits):
                    with h3_seeded(seed + j):
                        _c, _x, xs, _es = model.net.feature(x)
                    for l in layers:
                        st = xs[l][-1].detach().float().cpu()
                        acc[l] = st if acc[l] is None else acc[l] + st
                for l in layers:
                    a = (acc[l] / float(eval_inits)).numpy()          # (B,C,H,W)
                    for b in range(a.shape[0]):
                        dd = _sample_descriptors(a[b:b + 1], n, k=k)
                        for name in DESCRIPTORS:
                            desc[l][name][i].append(dd[name])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # decode each descriptor at each layer
    import m2_primitives as m2
    out = {}
    for l in layers:
        out[l] = {}
        for name in DESCRIPTORS:
            cv, chance = m2.linear_task_decodability(desc[l][name], seed=seed)
            out[l][name] = {"cv": float(cv), "chance": float(chance), "margin": float(cv - chance)}
    return out


def _verdict(per_seed, layers):
    import numpy as np
    paired = [p for p in per_seed if all(a in p for a in ARMS)]
    summary = {}
    for l in layers:
        summary[l] = {}
        for d in DESCRIPTORS:
            r6 = [p["R6"][l][d]["cv"] for p in paired]
            r5 = [p["R5:no_proj"][l][d]["cv"] for p in paired]
            diffs = [a - b for a, b in zip(r6, r5)]
            summary[l][d] = {"mean_cv_R6": float(np.mean(r6)) if r6 else None,
                             "mean_cv_R5": float(np.mean(r5)) if r5 else None,
                             "mean_delta": float(np.mean(diffs)) if diffs else None,
                             "n_pos": int(sum(1 for x in diffs if x > 0)), "n": len(diffs),
                             "deltas": [round(x, 4) for x in diffs]}
    # decision: does any relational descriptor rescue at the primary layer (or any layer)?
    rescued = []
    for l in layers:
        d0 = summary[l]["marginal"]["mean_delta"] or 0.0
        for d in RELATIONAL:
            md = summary[l][d]["mean_delta"]
            if md is not None and md >= RESCUE_DELTA and (md - d0) >= RESCUE_OVER_MARGINAL:
                rescued.append({"layer": l, "descriptor": d, "mean_delta": md, "marginal_delta": d0})
    call = ("RESCUE (a relational descriptor reveals R6>>R5 the marginal pooling hid)"
            if rescued else
            "NULL ROBUST (no descriptor, incl relational, shows synchrony adding task-channel)")
    return {"call": call, "rescued": rescued, "summary": summary}


def _save(per_seed, layers, n_tasks, epochs, final=False):
    os.makedirs(RESULTS, exist_ok=True)
    v = _verdict(per_seed, layers) if any(all(a in p for a in ARMS) for p in per_seed) else {"call": "no paired seeds"}
    out = {"arms": list(ARMS), "descriptors": DESCRIPTORS, "relational": RELATIONAL,
           "rescue_delta": RESCUE_DELTA, "rescue_over_marginal": RESCUE_OVER_MARGINAL,
           "primary_layer": PRIMARY_LAYER, "n_tasks": n_tasks, "epochs": epochs,
           "per_seed": per_seed, "verdict": {str(k): vv for k, vv in v.items()} if isinstance(v, dict) else v}
    json.dump(out, open(os.path.join(RESULTS, "m2_relational_probe.json"), "w"), indent=2, default=str)
    if final and isinstance(v, dict) and "summary" in v:
        print("\n=== M2 RELATIONAL PROBE — R6 vs R5:no_proj task-decodability by descriptor ===")
        for l in layers:
            print("  -- layer %d --" % l)
            for d in DESCRIPTORS:
                s = v["summary"][l][d]
                tag = "(MARGINAL=screen)" if d == "marginal" else "(relational)"
                print("    %-12s %-16s R6=%.3f R5=%.3f  Δ=%+.4f  n_pos=%d/%d" % (
                    d, tag, s["mean_cv_R6"], s["mean_cv_R5"], s["mean_delta"], s["n_pos"], s["n"]))
        print("VERDICT:", v["call"])
        if v["rescued"]:
            print("RESCUED BY:", v["rescued"])
        print("PROBE_DONE")
    return out


def run(seeds, n_tasks=5, epochs=30, layers=(1, 2), device="cuda", eval_inits=4,
        max_samples_per_task=240, k=8):
    per_seed = []
    for s in seeds:
        rec = {"seed": s}
        for arm, kw in ARMS.items():
            rec[arm] = _train_and_capture(arm, kw, n_tasks, epochs, list(layers), device,
                                          eval_inits, s, max_samples_per_task, k=k)
            print("[seed %d] %s captured: %s" % (s, arm, " | ".join(
                "L%d %s" % (l, " ".join("%s=%.3f" % (d, rec[arm][l][d]["cv"]) for d in DESCRIPTORS))
                for l in layers)), flush=True)
        per_seed.append(rec)
        _save(per_seed, layers, n_tasks, epochs)
        print("[seed %d] saved (%d/%d)" % (s, len(per_seed), len(seeds)), flush=True)
    return _save(per_seed, layers, n_tasks, epochs, final=True)


def _demo():
    import numpy as np
    rng = np.random.default_rng(0)
    n, C, H, W = 4, 16, 8, 8
    # "synchronized/structured": directions cluster by spatial half -> relational+spatial signal,
    # but with the SAME global marginal as a random cloud (zero-mean) so D0 cannot see the difference.
    def structured():
        a = np.zeros((1, C, H, W))
        for g in range(C // n):
            v1 = rng.normal(size=n); v1 /= np.linalg.norm(v1)
            v2 = -v1
            a[0, g * n:(g + 1) * n, :, :W // 2] = v1[:, None, None]
            a[0, g * n:(g + 1) * n, :, W // 2:] = v2[:, None, None]
        return a
    def randomcloud():
        a = rng.normal(size=(1, C, H, W))
        return a
    ds = _sample_descriptors(structured(), n)
    dr = _sample_descriptors(randomcloud(), n)
    for name in DESCRIPTORS:
        print("  %-12s dim=%d  struct_norm=%.3f rand_norm=%.3f" % (
            name, ds[name].size, float(np.linalg.norm(ds[name])), float(np.linalg.norm(dr[name]))))
    # the structured one (two antipodal clusters) has marginal mean ~0 (antipodal cancels) but a
    # strongly anisotropic 2nd-moment / spatial pattern -> relational descriptors must differ more.
    print("=== M2 RELATIONAL PROBE DEMO OK (descriptors computed; relational ones carry the structure) ===")


def main():
    ap = argparse.ArgumentParser(description="M2 relational-descriptor probe (artifact-vs-null disambiguation)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    ap.add_argument("--max-samples-per-task", type=int, default=240)
    ap.add_argument("--k-clusters", type=int, default=8)
    a = ap.parse_args()
    if a.demo:
        _demo()
        return
    run(a.seeds, n_tasks=a.n_tasks, epochs=a.epochs, layers=tuple(a.layers), device=a.device,
        eval_inits=a.eval_inits, max_samples_per_task=a.max_samples_per_task, k=a.k_clusters)


if __name__ == "__main__":
    main()
