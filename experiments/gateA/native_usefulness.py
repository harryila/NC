"""
GATE A — FG-ARI x downstream-USEFULNESS dissociation on the native AKOrN ladder.
===============================================================================
Tests the paper's sharpest surviving hook (OCL evaluation crisis, 2602.07532 / 2504.07092):
can a CLEVRTex arm be HIGH on FG-ARI (localization / "where") yet differ on representation
USEFULNESS (object-property decoding / "what")? See PREREG-usefulness.md for the locked rule.

METHOD (clean by construction):
  * representation probed = the EXACT feature eval_obj clusters for FG-ARI: the readout map at
    model.out[0] input, [B, ch, Hf, Wf].
  * GT-mask-pooled object vectors: per GT object (flat-mask id), mean-pool the readout feature over
    that object's cells (mask nearest-downsampled to feature grid). Uses GROUND-TRUTH masks, so it
    measures property-encoding INDEPENDENT of the model's own segmentation -> decouples what from where.
  * targets: shape, size, color, material (object-intrinsic categoricals from the raw scene JSON).
  * probe: StandardScaler -> multinomial LogisticRegression; split BY SCENE 60/40; per-attribute test
    acc + 1000x object-bootstrap 95% CI + majority chance; U = mean chance-normalized acc.

Per-arm architecture is rebuilt EXACTLY as eval_obj.py does (verified against eval_obj.py:428-497):
  AKOrN(J = "attn" if J in {none,identity} else J, ...) -> sever_akorn (if J none) ->
  norm_ablate_akorn (if norm_ablate!=none, no_proj=True) -> EMA(net).load -> .ema_model.

OOM-SAFE (runs alongside a training job on the shared 4090): small --bs, torch.no_grad, per-object
pooling on CPU, empty_cache + del model between arms.

USAGE (on the box):
  cd /root/NC/external/akorn && PYTHONPATH=/root/NC:/root/NC/external/akorn \
  /usr/bin/python3.11 /root/NC/experiments/gateA/native_usefulness.py \
     --data_root /root/data/clevrtex/clevrtex_full --n_scenes 300 --bs 8 \
     --out /root/NC/experiments/gateA/results/native_usefulness.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F


# ---- per-arm configs: (name, J, norm_ablate, project, run_dir) ; ckpt = <run_dir>/ema_model.pth ----
DEFAULT_ARMS = [
    dict(name="full",     J="attn", norm_ablate="none",  project=True,  run="clvtex_akorn_attn_repro", fgari=75.5),
    dict(name="projoff",  J="attn", norm_ablate="none",  project=False, run="clvtex_projoff",           fgari=76.7),
    dict(name="normclamp",J="attn", norm_ablate="clamp", project=True,  run="clvtex_normclamp",         fgari=80.9),
    dict(name="severed",  J="none", norm_ablate="none",  project=True,  run="clvtex_severed_none",      fgari=38.5),
]
ATTRS = ["shape", "size", "color", "material"]


def build_arm(cfg, args, device):
    """Rebuild EXACTLY as eval_obj.py:428-497, then EMA-load the arm's ema_model.pth."""
    from source.models.objs.knet import AKOrN
    from ema_pytorch import EMA
    net = AKOrN(
        args.N, ch=args.ch, L=args.L, T=args.T,
        J=("attn" if cfg["J"] in ("none", "identity") else cfg["J"]),
        use_omega=args.use_omega, global_omg=args.global_omg, c_norm=args.c_norm,
        psize=args.psize, imsize=args.model_imsize, autorescale=args.autorescale,
        maxpool=args.maxpool, project=cfg["project"], heads=args.heads,
        use_ro_x=args.use_ro_x, no_ro=args.no_ro, gta=args.gta,
    ).to(device)
    if cfg["J"] in ("none", "identity"):
        if "/root/NC" not in sys.path:
            sys.path.insert(0, "/root/NC")
        from experiments.gateA.native_severance import sever_akorn
        net = sever_akorn(net)
    if cfg["norm_ablate"] != "none":
        if "/root/NC" not in sys.path:
            sys.path.insert(0, "/root/NC")
        from experiments.gateA.native_norm_ablate import norm_ablate_akorn
        net = norm_ablate_akorn(net, variant=cfg["norm_ablate"], no_proj=True)
    ckpt = os.path.join(args.runs_root, cfg["run"], args.ckpt_name)
    model = EMA(net)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not (k.endswith("initted") or k.endswith("step"))]
    if real_missing:
        raise RuntimeError(f"[{cfg['name']}] {len(real_missing)} param keys MISSING (arch mismatch): {real_missing[:6]}")
    model = model.ema_model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_scenes(args):
    """Return list of (img[3,128,128], msk[128,128] int, objects[list of attr dicts keyed by mask id])."""
    from source.data.datasets.objs.clevr_tex import get_clevrtex
    ds = get_clevrtex(args.data_root, split="test", data_type="full",
                      imsize=128, return_meta_data=True)
    scenes = []
    n = min(args.n_scenes, len(ds))
    for i in range(n):
        item = ds[i]
        ind, img, msk = item[0], item[1], item[2]
        msk = np.asarray(msk)[0].astype(np.int64)        # [128,128], 0=bg, 1..N = object index
        raw = json.load(open(ds.metadata_index[ind]))    # full attributes (color etc. survive here)
        by_id = {}
        for o in raw["objects"]:
            by_id[int(o["index"])] = {a: str(o.get(a, "NA")) for a in ATTRS}
        scenes.append((img.float(), msk, by_id))
    return scenes, n


def capture_readout(model, imgs, device):
    """Feature fed into model.out[0] (the eval_obj readout map), [B, ch, Hf, Wf]."""
    holder = {}
    h = model.out[0].register_forward_hook(lambda _m, inp, _o: holder.__setitem__("f", inp[0].detach()))
    with torch.no_grad():
        _ = model(imgs.to(device))
    h.remove()
    return holder["f"]


def pool_objects(feat, msks):
    """feat [B,ch,Hf,Wf]; msks list of [H,W] int. Upsample feat to mask resolution (bilinear) and
    mean-pool over each object's full-res GT cells, so small objects get adequate support.
    Returns (X [n_obj, ch], scene_ids, obj_ids)."""
    B, ch, Hf, Wf = feat.shape
    H, W = msks[0].shape
    feat_up = F.interpolate(feat.float(), size=(H, W), mode="bilinear", align_corners=False).cpu().numpy()
    X, sids, oids = [], [], []
    for b in range(B):
        fb = feat_up[b].reshape(ch, H * W).T                                              # [H*W, ch]
        flat = msks[b].reshape(-1)
        for v in np.unique(flat):
            if v == 0:
                continue
            cells = flat == v
            if cells.sum() < 4:                                                           # skip near-invisible objects
                continue
            X.append(fb[cells].mean(0))
            sids.append(b)
            oids.append(int(v))
    return (np.stack(X) if X else np.zeros((0, ch))), sids, oids


def run_arm(model, scenes, scene_offset, args, device):
    """Extract pooled object vectors + attribute labels for one arm over all scenes."""
    X_all, lab_all, scene_all = [], {a: [] for a in ATTRS}, []
    imgs = torch.stack([s[0] for s in scenes])
    for start in range(0, len(scenes), args.bs):
        sl = slice(start, start + args.bs)
        feat = capture_readout(model, imgs[sl], device)
        msks = [scenes[j][1] for j in range(start, min(start + args.bs, len(scenes)))]
        X, sids, oids = pool_objects(feat, msks)
        for k, (sid, oid) in enumerate(zip(sids, oids)):
            gobj = scenes[start + sid][2].get(oid)
            if gobj is None:
                continue
            X_all.append(X[k])
            scene_all.append(scene_offset + start + sid)
            for a in ATTRS:
                lab_all[a].append(gobj[a])
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.stack(X_all), {a: np.array(lab_all[a]) for a in ATTRS}, np.array(scene_all)


def probe_attribute(X, y, scenes, seed=0, n_boot=1000):
    """Scene-split 60/40 logistic-regression probe; return test acc, bootstrap CI, chance."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    rng = np.random.RandomState(seed)
    uniq_sc = np.unique(scenes)
    rng.shuffle(uniq_sc)
    cut = int(0.6 * len(uniq_sc))
    train_sc, test_sc = set(uniq_sc[:cut].tolist()), set(uniq_sc[cut:].tolist())
    tr = np.array([s in train_sc for s in scenes])
    te = ~tr
    le = LabelEncoder().fit(y)
    yc = le.transform(y)
    # drop test classes unseen in train? LogisticRegression handles; chance = majority over all
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[tr], yc[tr])
    pred = clf.predict(X[te])
    correct = (pred == yc[te]).astype(float)
    acc = float(correct.mean())
    # majority-class chance from train distribution
    vals, cnts = np.unique(yc[tr], return_counts=True)
    maj = vals[cnts.argmax()]
    chance = float((yc[te] == maj).mean())
    # bootstrap CI over test objects
    boots = []
    idx = np.arange(len(correct))
    for _ in range(n_boot):
        bi = rng.choice(idx, len(idx), replace=True)
        boots.append(correct[bi].mean())
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return dict(acc=acc, ci=[lo, hi], chance=chance, n_classes=int(len(le.classes_)),
                n_train=int(tr.sum()), n_test=int(te.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_full")
    ap.add_argument("--runs_root", default="/root/NC/external/akorn/runs")
    ap.add_argument("--ckpt_name", default="ema_model.pth")
    ap.add_argument("--n_scenes", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/native_usefulness.json")
    # model hyperparams (AKOrN^attn objs config; must match the checkpoints)
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--psize", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", default="gn"); ap.add_argument("--model_imsize", type=int, default=128)
    ap.add_argument("--use_omega", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--global_omg", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--maxpool", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--use_ro_x", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--no_ro", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--gta", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--autorescale", type=lambda s: s.lower() == "true", default=False)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    scenes, n = load_scenes(args)
    print(f"[data] {n} scenes loaded ({sum(len(s[2]) for s in scenes)} objects)", flush=True)

    results = {}
    for cfg in DEFAULT_ARMS:
        try:
            model, ckpt = build_arm(cfg, args, device)
        except Exception as e:
            print(f"[{cfg['name']}] BUILD/LOAD FAILED: {e}", flush=True)
            results[cfg["name"]] = dict(error=str(e), fgari=cfg["fgari"])
            continue
        X, labs, sc = run_arm(model, scenes, 0, args, device)
        arm_res = dict(fgari=cfg["fgari"], n_objects=int(X.shape[0]), ckpt=ckpt, attrs={})
        unorms = []
        for a in ATTRS:
            pr = probe_attribute(X, labs[a], sc, seed=args.seed)
            arm_res["attrs"][a] = pr
            denom = max(1e-9, 1.0 - pr["chance"])
            unorms.append((pr["acc"] - pr["chance"]) / denom)
        arm_res["U_norm"] = float(np.mean(unorms))            # chance-normalized usefulness
        arm_res["U_mean_acc"] = float(np.mean([arm_res["attrs"][a]["acc"] for a in ATTRS]))
        results[cfg["name"]] = arm_res
        print(f"[{cfg['name']}] FG-ARI {cfg['fgari']}  nobj {arm_res['n_objects']}  U_norm {arm_res['U_norm']:.3f}  "
              f"accs " + " ".join(f"{a}={arm_res['attrs'][a]['acc']:.3f}(c{arm_res['attrs'][a]['chance']:.2f})" for a in ATTRS),
              flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- pre-registered decision rule (PREREG-usefulness.md) ----
    verdict = {}
    if all(k in results and "U_norm" in results[k] for k in ("full", "normclamp", "severed")):
        Uf, Uc, Us = results["full"]["U_norm"], results["normclamp"]["U_norm"], results["severed"]["U_norm"]
        d1 = Uc <= Uf  # clamp's higher FG-ARI is hollow (usefulness not higher)
        d2 = Us >= 0.60 * Uf  # severance preserves >=60% usefulness despite -37 FG-ARI (where/what split)
        arms_present = [a for a in ("severed", "full", "projoff", "normclamp") if "U_norm" in results.get(a, {})]
        fgs = [results[a]["fgari"] for a in arms_present]
        us = [results[a]["U_norm"] for a in arms_present]
        from scipy.stats import spearmanr
        rho = float(spearmanr(fgs, us).correlation) if len(arms_present) >= 3 else float("nan")
        if d1 or d2:
            label = "DISSOCIATION-REAL"
        elif rho >= 0.9:
            label = "NO-DISSOCIATION"
        else:
            label = "AMBIGUOUS"
        verdict = dict(label=label, D1_hollow_clamp=bool(d1), D2_whatwhere_split=bool(d2),
                       spearman_fg_u=rho, U_full=Uf, U_clamp=Uc, U_severed=Us)
    results["_verdict"] = verdict
    results["_meta"] = dict(n_scenes=n, seconds=round(time.time() - t0, 1), device=device, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("\n=== VERDICT ===", json.dumps(verdict, indent=2), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
