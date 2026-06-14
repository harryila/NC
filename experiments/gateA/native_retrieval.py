"""
GATE A — downstream-consequence DECIDER: cross-scene object retrieval by attribute (PREREG-downstream.md).
Turns the decodability reallocation into a real USE task: is FG-ARI ANTI-correlated with downstream retrieval utility?
For each arm: GT-mask-pooled object vectors, L2-norm; query=each object, gallery=objects in OTHER scenes (same-scene
masked to kill co-occurrence), rank by cosine, relevance=same attribute -> mAP@R + Recall@1 + 1000x query-bootstrap CI.
LOCKED rule (PREREG-downstream.md): INVERSION iff mAP_material(severed) > full & ItrSA (CI-sep) AND
rho(FG-ARI, mAP_material) <= -0.5 AND size-control opposite (mAP_size(full) >= severed).

Two modes:
  --npz <features.npz>           : score on PRECOMPUTED features (instant; full-test from native_usefulness_controls).
  --data_type outd --data_root  : EXTRACT fresh (OOD decider) for arms {full,itrsa,severed,normclamp} then score.
USAGE (OOD decider, box): python native_retrieval.py --data_type outd \
  --data_root /root/data/clevrtex/clevrtex_outd --n_scenes 2000 --device cpu --out results/retrieval_outd.json
"""
import os, sys, json, argparse
import numpy as np

_GA = "/root/NC/experiments/gateA"
for p in (_GA, "/root/NC"):
    if p not in sys.path:
        sys.path.insert(0, p)

ATTRS = ["material", "size", "shape"]                 # color dropped (dead)
FGARI = {"full_readout": 75.5, "full": 75.5, "severed_readout": 38.5, "severed": 38.5,
         "itrsa_readout": 63.5, "itrsa": 63.5, "normclamp": 80.9, "normclamp_readout": 80.9,
         "projoff": 76.7, "rawpixels": 0.0, "patchfy_full": 0.0, "randinit": 0.0}


def l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def retrieval(X, y, sc, n_boot=1000, seed=0):
    """leave-one-scene-out cosine retrieval. mAP (relevance=same label) + Recall@1 + bootstrap CI over queries."""
    Xn = l2(X.astype("float32"))
    S = Xn @ Xn.T                                       # [N,N] cosine
    N = len(X)
    aps, r1 = [], []
    for i in range(N):
        gal = sc != sc[i]                               # gallery = other scenes
        rel = (y[gal] == y[i])
        nrel = int(rel.sum())
        if nrel == 0:
            continue
        order = np.argsort(-S[i][gal])
        rs = rel[order].astype("float32")
        cum = np.cumsum(rs)
        prec = cum / (np.arange(len(rs)) + 1)
        aps.append(float((prec * rs).sum() / nrel))
        r1.append(float(rs[0]))
    aps = np.array(aps); r1 = np.array(r1)
    rng = np.random.RandomState(seed)
    boots = [aps[rng.randint(0, len(aps), len(aps))].mean() for _ in range(n_boot)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # chance mAP = sum over classes of frac^2 (prob a random gallery item is relevant, ~)
    vals, cnts = np.unique(y, return_counts=True); fr = cnts / cnts.sum()
    return dict(mAP=float(aps.mean()), ci=[lo, hi], R1=float(r1.mean()),
                chance=float((fr ** 2).sum()), n_q=int(len(aps)), n_classes=int(len(vals)))


# ---------- feature sources ----------
def from_npz(npz):
    d = np.load(npz, allow_pickle=True)
    arms = {}
    for k in d.files:
        if k.endswith("__X"):
            nm = k[:-3]
            arms[nm] = dict(X=d[nm + "__X"].astype("float32"), sc=d[nm + "__sc"],
                            labs={a: d[nm + "__lab_" + a] for a in ATTRS if nm + "__lab_" + a in d.files})
    return arms


def extract_fresh(args):
    """Extract GT-pooled features for {full,itrsa,severed,normclamp} on a given data split (e.g. OOD)."""
    import torch
    import native_usefulness as NU
    import native_usefulness_controls as NUC

    # load scenes for the requested data_type (full/outd/camo)
    from source.data.datasets.objs.clevr_tex import get_clevrtex
    ds = get_clevrtex(args.data_root, split="test", data_type=args.data_type, imsize=128, return_meta_data=True)
    scenes = []
    for i in range(min(args.n_scenes, len(ds))):
        ind, img, msk = ds[i][0], ds[i][1], ds[i][2]
        msk = np.asarray(msk)[0].astype(np.int64)
        raw = json.load(open(ds.metadata_index[ind]))
        by_id = {int(o["index"]): {a: str(o.get(a, "NA")) for a in NU.ATTRS} for o in raw["objects"]}
        scenes.append((img.float(), msk, by_id))
    print("[data] %s: %d scenes, %d objects" % (args.data_type, len(scenes), sum(len(s[2]) for s in scenes)), flush=True)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    arms = {}
    def grab(nm, model):
        X, labs, sc = NUC.extract_X(scenes, lambda im: NU.capture_readout(model, im, dev), args, dev)
        arms[nm] = dict(X=X, sc=sc, labs={a: labs[a] for a in ATTRS})
        del model
    grab("full", NUC.build_full(args, dev, load=True))
    grab("itrsa", NUC.build_itrsa(args, dev))
    grab("severed", NU.build_arm(dict(name="severed", J="none", norm_ablate="none", project=True, run="clvtex_severed_none"), args, dev)[0])
    grab("normclamp", NU.build_arm(dict(name="normclamp", J="attn", norm_ablate="clamp", project=True, run="clvtex_normclamp"), args, dev)[0])
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--data_type", default=None, choices=[None, "full", "outd", "camo"])
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_outd")
    ap.add_argument("--runs_root", default="/root/NC/external/akorn/runs")
    ap.add_argument("--ckpt_name", default="ema_model.pth")
    ap.add_argument("--n_scenes", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/retrieval.json")
    # model cfg (for fresh extraction)
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--psize", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", default="gn"); ap.add_argument("--model_imsize", type=int, default=128)
    for nm in ["use_omega", "global_omg", "maxpool", "use_ro_x", "no_ro", "gta", "autorescale"]:
        ap.add_argument("--" + nm, type=lambda s: s.lower() == "true", default=(nm in ("maxpool", "gta")))
    args = ap.parse_args()

    arms = from_npz(args.npz) if args.npz else extract_fresh(args)
    res = {}
    for nm, a in arms.items():
        if not a["labs"]:
            continue
        res[nm] = {at: retrieval(a["X"], a["labs"][at], a["sc"], seed=args.seed) for at in a["labs"]}
        print("%-16s | " % nm + " | ".join("%s mAP=%.3f[%.3f,%.3f] R@1=%.3f (c%.3f)" % (
            at, res[nm][at]["mAP"], res[nm][at]["ci"][0], res[nm][at]["ci"][1], res[nm][at]["R1"], res[nm][at]["chance"])
            for at in res[nm]), flush=True)

    # ---- locked decision rule (material = local, size = global control) ----
    def key(nm):  # normalize arm names across npz/fresh
        return {"full_readout": "full", "severed_readout": "severed", "itrsa_readout": "itrsa"}.get(nm, nm)
    R = {key(nm): v for nm, v in res.items()}
    verdict = {}
    if all(k in R and "material" in R[k] for k in ("full", "severed")):
        fm, sm = R["full"]["material"], R["severed"]["material"]
        im = R.get("itrsa", {}).get("material")
        mat_sev_gt_full = sm["mAP"] - (sm["ci"][1] - sm["mAP"]) > fm["mAP"] + (fm["mAP"] - fm["ci"][0])
        mat_sev_gt_itrsa = (im is None) or (sm["ci"][0] > im["ci"][1])
        # spearman(FG-ARI, mAP_material) over arms present
        arms_fg = [(FGARI.get(k, FGARI.get(k + "_readout", 0)), R[k]["material"]["mAP"]) for k in R if "material" in R[k] and FGARI.get(k, FGARI.get(k+"_readout")) ]
        rho = float("nan")
        if len(arms_fg) >= 3:
            from scipy.stats import spearmanr
            rho = float(spearmanr([x[0] for x in arms_fg], [x[1] for x in arms_fg]).correlation)
        size_ctrl = ("size" in R.get("full", {}) and "size" in R.get("severed", {}) and
                     R["full"]["size"]["ci"][0] >= R["severed"]["size"]["mAP"])
        inversion = bool(mat_sev_gt_full and mat_sev_gt_itrsa and (rho <= -0.5 if rho == rho else False) and size_ctrl)
        verdict = dict(INVERSION_CONFIRMED=inversion, mat_severed_gt_full=bool(mat_sev_gt_full),
                       mat_severed_gt_itrsa=bool(mat_sev_gt_itrsa), spearman_fg_material=rho,
                       size_control_full_ge_severed=bool(size_ctrl),
                       material_full=fm["mAP"], material_severed=sm["mAP"], material_itrsa=(im["mAP"] if im else None),
                       size_full=R["full"].get("size", {}).get("mAP"), size_severed=R["severed"].get("size", {}).get("mAP"))
    res["_verdict"] = verdict
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n=== VERDICT ===", json.dumps(verdict, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
