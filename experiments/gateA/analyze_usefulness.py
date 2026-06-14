"""
GATE A — usefulness probe-variant analysis on SAVED pooled features (cheap, frozen, seconds).
Loads native_usefulness_controls_features.npz and runs the rigor variants the validation workflow demanded:
  - multi-split (>=10 scene splits) LogReg on RAW and L2-normalized features -> mean +/- std (split-noise) + CI-gap
  - balanced accuracy (macro-recall) for material (kills majority-class artifact on ~60-way)
  - kNN-retrieval (leave-scene-out, k=5 cosine) — probe-free downstream measure
  - KMeans(n_classes)-AMI — unsupervised clustering quality of per-object property
  - drops dead `color` from the headline (near-chance for all arms incl raw pixels = degenerate label)
Prints the full arm x attr table + the W1/W2/W3 verdict on the full-vs-severed material reallocation.
USAGE (box): python analyze_usefulness.py --npz /root/NC/experiments/gateA/results/native_usefulness_controls_features.npz
"""
import argparse, json
import numpy as np

ARMS = ["rawpixels", "patchfy_full", "full_readout", "randinit", "severed_readout", "itrsa_readout"]
ATTRS = ["shape", "size", "material"]   # color dropped (dead: near-chance for raw pixels too)


def load(npz):
    d = np.load(npz, allow_pickle=True)
    arms = {}
    for nm in ARMS:
        k = nm + "__X"
        if k not in d.files:
            continue
        arms[nm] = dict(X=d[k].astype("float32"), sc=d[nm + "__sc"],
                        labs={a: d[nm + "__lab_" + a] for a in ATTRS})
    return arms


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-9, None)


def logreg_multisplit(X, y, sc, n_splits=3):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    yc = LabelEncoder().fit_transform(y)
    uniq = np.unique(sc)
    accs, bals = [], []
    for s in range(n_splits):
        rng = np.random.RandomState(s)
        u = uniq.copy(); rng.shuffle(u)
        tr_sc = set(u[:int(0.6 * len(u))].tolist())
        tr = np.array([x in tr_sc for x in sc]); te = ~tr
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=1.0, solver="saga", n_jobs=4))
        clf.fit(X[tr], yc[tr]); pred = clf.predict(X[te])
        accs.append(float((pred == yc[te]).mean()))
        bals.append(float(balanced_accuracy_score(yc[te], pred)))
    vals, cnts = np.unique(yc, return_counts=True)
    chance = float(cnts.max() / len(yc))
    return dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                bal=float(np.mean(bals)), chance=chance, n_classes=int(len(vals)))


def knn_retrieval(X, y, sc, k=5, n_splits=5):
    """leave-scene-out cosine kNN attribute retrieval (probe-free)."""
    from sklearn.preprocessing import LabelEncoder
    Xn = l2(X); yc = LabelEncoder().fit_transform(y); uniq = np.unique(sc)
    accs = []
    for s in range(n_splits):
        rng = np.random.RandomState(100 + s)
        u = uniq.copy(); rng.shuffle(u)
        tr_sc = set(u[:int(0.6 * len(u))].tolist())
        tr = np.array([x in tr_sc for x in sc]); te = ~tr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sim = Xn[te] @ Xn[tr].T                                  # cosine (unit vectors)
        nn = np.argsort(-sim, axis=1)[:, :k]
        ytr = yc[tr]
        pred = np.array([np.bincount(ytr[row]).argmax() for row in nn])
        accs.append(float((pred == yc[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def kmeans_ami(X, y):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.preprocessing import LabelEncoder
    yc = LabelEncoder().fit_transform(y); ncls = len(np.unique(yc))
    km = KMeans(n_clusters=max(2, min(ncls, 40)), random_state=0, n_init=5).fit(l2(X))
    return float(adjusted_mutual_info_score(yc, km.labels_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/root/NC/experiments/gateA/results/native_usefulness_controls_features.npz")
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/usefulness_analysis.json")
    args = ap.parse_args()
    arms = load(args.npz)
    print("loaded arms:", list(arms.keys()), "| n_obj:", {k: len(v["sc"]) for k, v in arms.items()})

    res = {}
    for nm, a in arms.items():
        X, sc = a["X"], a["sc"]
        res[nm] = {}
        for at in ATTRS:
            y = a["labs"][at]
            l2r = logreg_multisplit(l2(X), y, sc, args.n_splits)   # L2 only (raw is in controls2.json already)
            res[nm][at] = dict(lr_l2=l2r["acc"], lr_l2_std=l2r["acc_std"],
                               bal_l2=l2r["bal"], knn=knn_retrieval(X, y, sc),
                               kmeans_ami=kmeans_ami(X, y), chance=l2r["chance"], n_classes=l2r["n_classes"])
        print("%-16s | " % nm + " || ".join(
            "%s: lrL2 %.3f±%.3f bal %.3f knn %.3f ami %.3f (c%.2f)" % (
                at, res[nm][at]["lr_l2"], res[nm][at]["lr_l2_std"], res[nm][at]["bal_l2"],
                res[nm][at]["knn"], res[nm][at]["kmeans_ami"], res[nm][at]["chance"]) for at in ATTRS), flush=True)

    # ---- W-verdict on material reallocation (full vs severed), L2 + floors + CI(std-gap) ----
    v = {}
    if "full_readout" in res and "severed_readout" in res:
        f, s = res["full_readout"]["material"], res["severed_readout"]["material"]
        floors = {k: res[k]["material"]["lr_l2"] for k in ("rawpixels", "patchfy_full", "randinit") if k in res}
        gap = s["lr_l2"] - f["lr_l2"]
        ci_disjoint = gap > 2 * (f["lr_l2_std"] + s["lr_l2_std"])      # crude split-noise separation
        sev_above_floors = all(s["lr_l2"] > fl + 0.02 for fl in floors.values())
        knn_holds = res["severed_readout"]["material"]["knn"] > res["full_readout"]["material"]["knn"]
        if ci_disjoint and sev_above_floors and knn_holds:
            label = "W1_REALLOCATION (severed builds genuine material rep > all floors, survives L2+kNN; full spends capacity on grouping+size)"
        elif ci_disjoint and sev_above_floors:
            label = "W1_minus (LogReg/L2 strong; recheck kNN)"
        elif ci_disjoint:
            label = "W2 (material gap real but severed not clearly above floors)"
        else:
            label = "WEAK/ARTIFACT (gap within split-noise)"
        # size = global control (expect full > severed)
        fs, ss = res["full_readout"]["size"]["lr_l2"], res["severed_readout"]["size"]["lr_l2"]
        v = dict(label=label, material_full=f["lr_l2"], material_severed=s["lr_l2"], material_gap=gap,
                 material_floors=floors, ci_disjoint=bool(ci_disjoint), sev_above_floors=bool(sev_above_floors),
                 knn_holds=bool(knn_holds), size_full=fs, size_severed=ss, size_reallocation=bool(fs > ss + 0.02))
    res["_verdict"] = v
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n=== W-VERDICT ===", json.dumps(v, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
