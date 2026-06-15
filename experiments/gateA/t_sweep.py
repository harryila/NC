"""
DOSE-RESPONSE T-sweep (eval-only, single trained full AKOrN checkpoint): vary the number of Kuramoto recurrent steps
T in {1,2,4,6,8} at EVAL time (AKOrN is built for variable-T eval) -> grouping STRENGTH knob WITHOUT retraining.
For each T, on the SAME OOD split, from the SAME canonical readout activation (model.out[0] output, eval_obj-normalized):
  - FG-ARI  : eval_obj.eval (agglomerative, n_clusters=11) — the field's segmentation metric.
  - utility : GT-mask-pooled object vectors -> cross-scene material/size retrieval (mAP@R + R@1), NR.retrieval.
Turns the discrete severance ladder into a CONTINUOUS within-model dose-response: does FG-ARI rise with T while
downstream material utility falls? (the strongest cheap version of the causal anti-correlation claim).

USAGE (box): python t_sweep.py --data_type outd --data_root /root/data/clevrtex/clevrtex_outd \
  --n_scenes 320 --Ts 1,2,4,6,8 --device cuda --bs 16 --out results/t_sweep_outd.json
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


def sweep_T(args, t, scenes, dev):
    import torch
    import eval_obj as EO
    EO.USE_GPU_FOR_PCA = False                                  # keep GPU memory frugal (seeds are training)
    args.T = t
    model = NUC.build_full(args, dev, load=True)                # trained full ckpt, rebuilt at this T
    fgaris, Xrows, sids, labs = [], [], [], {a: [] for a in ATTRS}
    sc = 0
    bs = args.bs
    for b0 in range(0, len(scenes), bs):
        batch = scenes[b0:b0 + bs]
        imgs = torch.stack([s[0] for s in batch])              # [b,3,H,W]
        gts = torch.from_numpy(np.stack([s[1] for s in batch])).long()
        # canonical FG-ARI (reuses eval_obj.eval: saccade+PCA(cpu)+agglomerative n_clusters=11)
        with torch.no_grad():
            scores, _ = EO.eval(model, imgs, gts, method="agglomerative", n_clusters=11, saccade_r=1, pca=True)
            fgaris.extend(list(np.array(scores["fgari"]).reshape(-1)))
            v = EO.model_preds(model, imgs).cpu()               # [b,C,h,w] canonical normalized readout
        masks = [s[1] for s in batch]
        X, s_ids, o_ids = NU.pool_objects(v, masks)             # GT-mask-pooled per object
        for k, (sid, oid) in enumerate(zip(s_ids, o_ids)):
            byid = batch[sid][2]
            if oid not in byid:
                continue
            Xrows.append(X[k]); sids.append(sc + sid)
            for a in ATTRS:
                labs[a].append(byid[oid].get(a, "NA"))
        sc += len(batch)
    del model
    Xrows = np.stack(Xrows).astype("float32"); scarr = np.array(sids)
    out = {"T": t, "fgari": float(np.mean(fgaris)), "n_fgari": len(fgaris), "n_obj": len(Xrows)}
    for a in ATTRS:
        y = np.array(labs[a]); _, yi = np.unique(y, return_inverse=True)
        r = NR.retrieval(Xrows, yi, scarr, seed=args.seed)
        out[a] = {"mAP": r["mAP"], "ci": r["ci"], "R1": r["R1"], "chance": r["chance"]}
    print("T=%d | FG-ARI=%.3f | material mAP=%.3f R@1=%.3f | size mAP=%.3f R@1=%.3f" % (
        t, out["fgari"], out["material"]["mAP"], out["material"]["R1"],
        out["size"]["mAP"], out["size"]["R1"]), flush=True)
    return out


def spearman(x, y):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for rk, i in enumerate(s):
            r[i] = rk
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    if n < 3:
        return float("nan")
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_type", default="outd", choices=["full", "outd", "camo"])
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_outd")
    ap.add_argument("--runs_root", default="/root/NC/external/akorn/runs")
    ap.add_argument("--ckpt_name", default="ema_model.pth")
    ap.add_argument("--Ts", default="1,2,4,6,8")
    ap.add_argument("--n_scenes", type=int, default=320)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/t_sweep_outd.json")
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--psize", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", default="gn"); ap.add_argument("--model_imsize", type=int, default=128)
    for nm in ["use_omega", "global_omg", "maxpool", "use_ro_x", "no_ro", "gta", "autorescale"]:
        ap.add_argument("--" + nm, type=lambda s: s.lower() == "true", default=(nm in ("maxpool", "gta")))
    args = ap.parse_args()

    import torch
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    Ts = [int(x) for x in args.Ts.split(",")]
    scenes = load_scenes(args)
    res = {"_meta": {"data_type": args.data_type, "n_scenes": args.n_scenes, "Ts": Ts, "device": dev}}
    rows = [sweep_T(args, t, scenes, dev) for t in Ts]
    res["sweep"] = rows
    fg = [r["fgari"] for r in rows]
    res["dose_response"] = {
        "spearman_fgari_material_mAP": spearman(fg, [r["material"]["mAP"] for r in rows]),
        "spearman_fgari_material_R1": spearman(fg, [r["material"]["R1"] for r in rows]),
        "spearman_fgari_size_mAP": spearman(fg, [r["size"]["mAP"] for r in rows]),
        "fgari_rises_with_T": bool(fg[-1] > fg[0]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n=== DOSE-RESPONSE ===", json.dumps(res["dose_response"], indent=1), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
