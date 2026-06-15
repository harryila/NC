"""
FLOORS on the HEADLINE (retrieval) metric (reviewer ask): put the model-agnostic floors raw-pixel / conv-stem(patchfy) /
random-init INTO the OOD material-retrieval table, not just the decode probe. Proves severed's retrieval advantage is
"preserved INPUT info, destroyed-then-restored" -- severed should sit BETWEEN the conv-stem/randinit MODEL floors and raw
pixels -- NOT "magically better than the input". Compute-light: raw = no model; patchfy = conv-stem (pre-Kuramoto);
randinit = one untrained forward. (ItrSA + severed/full already in retrieval_outd.json; merged in the printout.)

USAGE (box): python floors_retrieval.py --data_type outd --data_root /root/data/clevrtex/clevrtex_outd \
  --n_scenes 2000 --device cuda --bs 8 --out results/retrieval_floors_outd.json
"""
import os, sys, json, argparse
import numpy as np

_GA, _AK = "/root/NC/experiments/gateA", "/root/NC/external/akorn"
for p in (_GA, _AK, "/root/NC"):
    if p not in sys.path:
        sys.path.insert(0, p)

import native_usefulness as NU
import native_usefulness_controls as NUC
import native_retrieval as NR
ATTRS = ["material", "size", "shape"]


def load_scenes(args):
    from source.data.datasets.objs.clevr_tex import get_clevrtex
    ds = get_clevrtex(args.data_root, split="test", data_type=args.data_type, imsize=128, return_meta_data=True)
    scenes = []
    for i in range(min(args.n_scenes, len(ds))):
        ind, img, msk = ds[i][0], ds[i][1], ds[i][2]
        msk = np.asarray(msk)[0].astype(np.int64)
        raw = json.load(open(ds.metadata_index[ind]))
        by_id = {int(o["index"]): {a: str(o.get(a, "NA")) for a in NU.ATTRS} for o in raw["objects"]}
        scenes.append((img.float(), msk, by_id))
    print("[data] %s: %d scenes" % (args.data_type, len(scenes)), flush=True)
    return scenes


def score(nm, X, labs, sc, seed):
    out = {}
    for a in ATTRS:
        y = labs[a]; _, yi = np.unique(y, return_inverse=True)
        r = NR.retrieval(X, yi, sc, seed=seed)
        out[a] = {"mAP": r["mAP"], "ci": r["ci"], "R1": r["R1"], "chance": r["chance"]}
    print("%-10s | " % nm + " | ".join("%s mAP=%.3f R@1=%.3f (c%.3f)" % (
        a, out[a]["mAP"], out[a]["R1"], out[a]["chance"]) for a in ATTRS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_type", default="outd"); ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_outd")
    ap.add_argument("--runs_root", default="/root/NC/external/akorn/runs"); ap.add_argument("--ckpt_name", default="ema_model.pth")
    ap.add_argument("--n_scenes", type=int, default=2000); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/retrieval_floors_outd.json")
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--psize", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", default="gn"); ap.add_argument("--model_imsize", type=int, default=128)
    for nm in ["use_omega", "global_omg", "maxpool", "use_ro_x", "no_ro", "gta", "autorescale"]:
        ap.add_argument("--" + nm, type=lambda s: s.lower() == "true", default=(nm in ("maxpool", "gta")))
    args = ap.parse_args()

    import torch
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    scenes = load_scenes(args)
    res = {"_meta": {"data_type": args.data_type, "n_scenes": args.n_scenes}}

    # raw-pixel floor (no model): pool the raw image as the "feature map"
    Xr, labs, sc = NUC.extract_X(scenes, lambda im: im.to(dev), args, dev)
    res["rawpixels"] = score("rawpixels", Xr, labs, sc, args.seed)
    # conv-stem (patchfy) floor: the FULL model's patch-embed output, pre-Kuramoto
    full = NUC.build_full(args, dev, load=True)
    Xp, labs, sc = NUC.extract_X(scenes, lambda im: NUC.capture_from(full, im, dev, "patchfy"), args, dev)
    res["patchfy"] = score("patchfy", Xp, labs, sc, args.seed)
    del full
    # random-init floor: architecture, no learning (one untrained forward)
    rnd = NUC.build_full(args, dev, load=False)
    Xz, labs, sc = NUC.extract_X(scenes, lambda im: NU.capture_readout(rnd, im, dev), args, dev)
    res["randinit"] = score("randinit", Xz, labs, sc, args.seed)
    del rnd

    # ---- merge with existing severed/full/itrsa from retrieval_outd.json for the headline floor-anchored ladder ----
    p = os.path.join(os.path.dirname(args.out), "retrieval_%s.json" % ("outd" if args.data_type == "outd" else "full"))
    pp = p if os.path.exists(p) else os.path.join(os.path.dirname(args.out), "retrieval_outd.json")
    merged = {}
    if os.path.exists(pp):
        d = json.load(open(pp))
        for a in ["severed", "itrsa", "full", "normclamp"]:
            if a in d and "material" in d[a]:
                merged[a] = {"mAP": d[a]["material"]["mAP"], "R1": d[a]["material"]["R1"]}
    print("\n=== FLOOR-ANCHORED MATERIAL LADDER (OOD mAP) ===", flush=True)
    order = [("rawpixels", res["rawpixels"]["material"]["mAP"]),
             ("severed", merged.get("severed", {}).get("mAP")),
             ("patchfy", res["patchfy"]["material"]["mAP"]),
             ("randinit", res["randinit"]["material"]["mAP"]),
             ("itrsa", merged.get("itrsa", {}).get("mAP")),
             ("full", merged.get("full", {}).get("mAP"))]
    for nm, v in sorted([o for o in order if o[1] is not None], key=lambda x: -x[1]):
        print("  %-10s material mAP=%.3f" % (nm, v), flush=True)
    res["_severed_between_floors"] = dict(
        severed=merged.get("severed", {}).get("mAP"),
        raw=res["rawpixels"]["material"]["mAP"], patchfy=res["patchfy"]["material"]["mAP"],
        randinit=res["randinit"]["material"]["mAP"], full=merged.get("full", {}).get("mAP"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
