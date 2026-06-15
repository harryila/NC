"""
Metric-critique analysis (reproducible, LOCAL, reads committed result JSONs — no model/box).
Produces (1) the DITTADI RECONCILIATION: per-attribute Spearman rho(FG-ARI, retrieval utility) across the AKOrN
severance ladder + the T-sweep, showing FG-ARI tracks GLOBAL attributes (size/shape) with OPPOSITE sign to LOCAL
material -> reconciles our anti-correlation with Dittadi 2022's aggregate POSITIVE ARI<->property correlation; and
(2) the pooled FG-ARI-vs-material scatter (Fig 3) with BOTH all-points and converged-regime rho (honesty: T=1 is
degenerate, so the all-points rho is weaker than the converged one — report both).

USAGE: python analyze_metric_critique.py   (run from experiments/gateA; reads results/*.json)
"""
import json, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FG_LADDER = {"severed": 38.5, "itrsa": 63.5, "full": 75.5, "normclamp": 80.9}
ATTRS = ["material", "size", "shape"]
LOCAL = {"material"}


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, j in enumerate(s):
            r[j] = i
        return r
    rx, ry = rk(x), rk(y); n = len(x)
    return round(1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)), 3)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ladder_corr(d, fg=FG_LADDER, arms=None):
    arms = arms or [a for a in ["severed", "itrsa", "full", "normclamp"] if a in d]
    fgv = [fg[a] for a in arms]
    out = {}
    for at in ATTRS:
        if all(at in d[a] for a in arms):
            out[at] = {"rho_mAP": spearman(fgv, [d[a][at]["mAP"] for a in arms]),
                       "rho_R1": spearman(fgv, [d[a][at]["R1"] for a in arms]),
                       "class": "LOCAL" if at in LOCAL else "GLOBAL"}
    return {"arms": arms, "per_attribute": out}


def tsweep_corr(d, tmin=2):
    rows = [r for r in d["sweep"] if r["T"] >= tmin]
    fgv = [r["fgari"] for r in rows]
    return {"T_used": [r["T"] for r in rows],
            "per_attribute": {at: {"rho_mAP": spearman(fgv, [r[at]["mAP"] for r in rows]),
                                   "class": "LOCAL" if at in LOCAL else "GLOBAL"} for at in ATTRS}}


def main():
    res = {}
    ak_ood = load("retrieval_outd.json")
    ak_ind = load("retrieval_full.json")
    ts = load("t_sweep_outd.json")

    print("=== DITTADI RECONCILIATION: per-attribute rho(FG-ARI, utility) ===")
    print("(LOCAL material should be << 0; GLOBAL size/shape >= 0 -> Dittadi's positive aggregate is GLOBAL-carried)")
    for tag, d in [("AKOrN ladder OOD", ak_ood), ("AKOrN ladder IN-DIST", ak_ind)]:
        if d is None:
            print("  [%s] (not available)" % tag); continue
        c = ladder_corr(d); res[tag] = c
        print("  [%s] arms=%s" % (tag, c["arms"]))
        for at, v in c["per_attribute"].items():
            print("    %-9s %-7s rho_mAP=%+.2f rho_R1=%+.2f" % (at, v["class"], v["rho_mAP"], v["rho_R1"]))
    if ts is not None:
        c = tsweep_corr(ts); res["T-sweep (T>=2)"] = c
        print("  [T-sweep T>=2] T=%s" % c["T_used"])
        for at, v in c["per_attribute"].items():
            print("    %-9s %-7s rho_mAP=%+.2f" % (at, v["class"], v["rho_mAP"]))

    # ---- pooled FG-ARI vs material scatter (Fig 3): AKOrN ladder + T-sweep ----
    pts = []
    if ak_ood:
        for a in ["severed", "itrsa", "full", "normclamp"]:
            if a in ak_ood:
                pts.append((FG_LADDER[a] / 100.0, ak_ood[a]["material"]["mAP"], "AKOrN:" + a))
    if ts:
        for r in ts["sweep"]:
            pts.append((r["fgari"], r["material"]["mAP"], "T=%d" % r["T"]))
    fg_all = [p[0] for p in pts]; mat_all = [p[1] for p in pts]
    conv = [p for p in pts if p[2] != "T=1"]               # drop degenerate (under-iterated) T=1
    res["pooled_scatter"] = {
        "n_points": len(pts),
        "rho_all_points": spearman(fg_all, mat_all),
        "rho_excl_T1_degenerate": spearman([p[0] for p in conv], [p[1] for p in conv]),
        "points": [{"fgari": round(f, 3), "material_mAP": round(m, 3), "label": l} for f, m, l in pts],
    }
    print("\n=== POOLED FG-ARI vs material-mAP scatter (Fig 3) ===")
    print("  n=%d points | rho(all)=%.3f | rho(excl T=1 degenerate)=%.3f" % (
        res["pooled_scatter"]["n_points"], res["pooled_scatter"]["rho_all_points"],
        res["pooled_scatter"]["rho_excl_T1_degenerate"]))
    print("  (honesty: report BOTH; T=1 is under-iterated/degenerate)")

    out = os.path.join(R, "metric_critique.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
