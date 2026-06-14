"""
GATE A — USEFULNESS CONTROL ARMS (kills the "severed is just less-processed" confound).
=======================================================================================
The where/what finding (native_usefulness.py): severing coupling halves FG-ARI (75.5->38.5) yet GT-mask-pooled
attribute decode does NOT drop — material ~DOUBLES (0.262->0.469), size drops (0.604->0.475), shape flat. Lead
confound: does severed's higher material-decode just mean it is LESS PROCESSED (closer to raw conv texture), not a
better object representation? This adds the missing REFERENCE arms, all probed with the EXACT same GT-mask-pool +
scene-split logistic probe, same scenes/seed:

  rawpixels   : pool the raw image (3-ch). Absolute floor of material-in-pixels.
  patchfy     : the FULL model's patch-embedding output (pre-Kuramoto, same 16x16 grid as the readout) — "raw conv
                texture before any coupling". THE decisive baseline.
  randinit    : a random-init AKOrN readout (architecture, no learning) at the readout grid.
  full        : full AKOrN readout (post-Kuramoto) [re-run here for same-scenes comparison].
  severed     : J=none readout [re-run here for same-scenes comparison].

DECISIVE READOUT (pre-registered logic):
  - If patchfy_material ≈ severed_material > full_material  => the coupling DESTROYS material info present in the
    early feature; severing it merely RESTORES it. Cleanest, strongest framing ("synchrony trades recoverable
    per-object detail for grouping"). 'severed better' is then 'less destructive', stated honestly.
  - If severed_material > patchfy_material too => severance genuinely BUILDS material rep beyond the early feature
    (stronger 'representation' wording allowed).
  - If full_material ≈ patchfy_material (coupling does NOT destroy material) => the doubling is something else;
    re-examine. (Would weaken the framing.)
  Size is the global control (expect smoothing to HELP it: full/patchfy >= severed).

Frozen + CPU-safe (runs alongside the rigor campaign on the shared box).
USAGE (box): PYTHONPATH=/root/NC:/root/NC/external/akorn CUDA_VISIBLE_DEVICES="" \
  python /root/NC/experiments/gateA/native_usefulness_controls.py --n_scenes 500 --device cpu \
  --out /root/NC/experiments/gateA/results/native_usefulness_controls.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch

_GA = "/root/NC/experiments/gateA"
if _GA not in sys.path:
    sys.path.insert(0, _GA)
if "/root/NC" not in sys.path:
    sys.path.insert(0, "/root/NC")
import native_usefulness as NU   # reuse load_scenes, pool_objects, probe_attribute, build_arm, ATTRS


def capture_from(model, imgs, device, module_attr):
    """Hook a named module's OUTPUT (e.g. 'patchfy') -> [B,ch,Hf,Wf]."""
    holder = {}
    mod = getattr(model, module_attr)
    h = mod.register_forward_hook(lambda _m, _i, o: holder.__setitem__("f", (o[0] if isinstance(o, (tuple, list)) else o).detach()))
    with torch.no_grad():
        _ = model(imgs.to(device))
    h.remove()
    return holder["f"]


def build_full(args, device, load=True):
    """Full AKOrN (J=attn, project=True). load=False -> random init (no-learning floor)."""
    from source.models.objs.knet import AKOrN
    from ema_pytorch import EMA
    net = AKOrN(args.N, ch=args.ch, L=args.L, T=args.T, J="attn",
                use_omega=args.use_omega, global_omg=args.global_omg, c_norm=args.c_norm,
                psize=args.psize, imsize=args.model_imsize, autorescale=args.autorescale,
                maxpool=args.maxpool, project=True, heads=args.heads,
                use_ro_x=args.use_ro_x, no_ro=args.no_ro, gta=args.gta).to(device)
    if not load:
        return net.eval()
    ckpt = os.path.join(args.runs_root, "clvtex_akorn_attn_repro", args.ckpt_name)
    model = EMA(net)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model.load_state_dict(sd, strict=False)
    return model.ema_model.to(device).eval()


def build_itrsa(args, device):
    """Trained ItrSA (ViT, T-step iterative self-attention, NO Kuramoto). The non-oscillator cross-token floor:
    does ANY cross-token grouping op show the where/what reallocation, or is it Kuramoto-specific?"""
    from source.models.objs.vit import ViT
    from ema_pytorch import EMA
    net = ViT(psize=args.psize, imsize=args.model_imsize, autorescale=args.autorescale, ch=args.ch,
              blocks=args.L, heads=args.heads, mlp_dim=2 * args.ch, T=args.T, maxpool=args.maxpool, gta=False).to(device)
    ckpt = os.path.join(args.runs_root, "clvtex_itrsa", args.ckpt_name)
    model = EMA(net)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model.load_state_dict(sd, strict=False)
    return model.ema_model.to(device).eval()


def extract_X(scenes, feat_fn, args, device):
    """Generic: feat_fn(imgs_slice)->[B,ch,Hf,Wf]; pool GT masks; align attribute labels. Mirrors NU.run_arm."""
    X_all, lab_all, scene_all = [], {a: [] for a in NU.ATTRS}, []
    imgs = torch.stack([s[0] for s in scenes])
    for start in range(0, len(scenes), args.bs):
        feat = feat_fn(imgs[start:start + args.bs])
        msks = [scenes[j][1] for j in range(start, min(start + args.bs, len(scenes)))]
        X, sids, oids = NU.pool_objects(feat, msks)
        for k, (sid, oid) in enumerate(zip(sids, oids)):
            gobj = scenes[start + sid][2].get(oid)
            if gobj is None:
                continue
            X_all.append(X[k]); scene_all.append(start + sid)
            for a in NU.ATTRS:
                lab_all[a].append(gobj[a])
    return np.stack(X_all), {a: np.array(lab_all[a]) for a in NU.ATTRS}, np.array(scene_all)


def probe_all(name, X, labs, sc, seed):
    """Probe RAW features AND L2-normalized (cosine, the geometry FG-ARI clusters) -> kills the
    common-mode variance artifact. If the material gap survives L2-norm it is content, not geometry."""
    Xl2 = torch.nn.functional.normalize(torch.from_numpy(X.astype("float32")), dim=1).numpy()
    row = {"name": name, "n_objects": int(X.shape[0]), "attrs": {}, "attrs_l2": {}}
    for a in NU.ATTRS:
        row["attrs"][a] = NU.probe_attribute(X, labs[a], sc, seed=seed)
        row["attrs_l2"][a] = NU.probe_attribute(Xl2, labs[a], sc, seed=seed)
    accs = {a: row["attrs"][a]["acc"] for a in NU.ATTRS}
    a2 = {a: row["attrs_l2"][a]["acc"] for a in NU.ATTRS}
    print("%-14s nobj=%d | RAW %s | L2 %s" % (name, row["n_objects"],
          " ".join("%s=%.3f" % (a, accs[a]) for a in NU.ATTRS),
          " ".join("%s=%.3f" % (a, a2[a]) for a in NU.ATTRS)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_full")
    ap.add_argument("--runs_root", default="/root/NC/external/akorn/runs")
    ap.add_argument("--ckpt_name", default="ema_model.pth")
    ap.add_argument("--n_scenes", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/native_usefulness_controls.json")
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--psize", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--c_norm", default="gn"); ap.add_argument("--model_imsize", type=int, default=128)
    for nm in ["use_omega", "global_omg", "maxpool", "use_ro_x", "no_ro", "gta", "autorescale"]:
        ap.add_argument("--" + nm, type=lambda s: s.lower() == "true",
                        default=(nm in ("maxpool", "gta")))
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    t0 = time.time()
    scenes, n = NU.load_scenes(args)
    print("[data] %d scenes, %d objects" % (n, sum(len(s[2]) for s in scenes)), flush=True)
    rows = {}
    feats = {}

    def do(name, feat_fn):
        X, labs, sc = extract_X(scenes, feat_fn, args, device)
        feats[name] = (X, labs, sc)
        rows[name] = probe_all(name, X, labs, sc, args.seed)

    do("rawpixels", lambda im: im)                                           # raw pixel floor (texture/color)
    full = build_full(args, device, load=True)
    do("patchfy_full", lambda im: capture_from(full, im, device, "patchfy"))  # conv-stem, pre-Kuramoto
    do("full_readout", lambda im: NU.capture_readout(full, im, device))       # post-coupling readout
    del full
    rnd = build_full(args, device, load=False)
    do("randinit", lambda im: NU.capture_readout(rnd, im, device))            # architecture, no learning
    del rnd
    sev, _ = NU.build_arm(dict(name="severed", J="none", norm_ablate="none", project=True, run="clvtex_severed_none"), args, device)
    do("severed_readout", lambda im: NU.capture_readout(sev, im, device))     # coupling-severed readout
    del sev
    it = build_itrsa(args, device)
    do("itrsa_readout", lambda im: NU.capture_readout(it, im, device))        # non-oscillator cross-token floor
    del it

    # save pooled features so downstream probe variants (multi-split, balanced-acc, kNN, KMeans, n>=3) are free
    npz = {}
    for nm, (X, labs, sc) in feats.items():
        npz[nm + "__X"] = X
        npz[nm + "__sc"] = sc
        for a in NU.ATTRS:
            npz[nm + "__lab_" + a] = labs[a]
    feat_path = args.out.replace(".json", "_features.npz")
    np.savez(feat_path, **npz)
    print("[saved features]", feat_path, flush=True)

    # --- decisive comparison on material (local) + size (global) ---
    def mat(k): return rows[k]["attrs"]["material"]["acc"]
    def sz(k):  return rows[k]["attrs"]["size"]["acc"]
    cmp = dict(
        material=dict(rawpixels=mat("rawpixels"), patchfy=mat("patchfy_full"), full=mat("full_readout"),
                      randinit=mat("randinit"), severed=mat("severed_readout")),
        size=dict(rawpixels=sz("rawpixels"), patchfy=sz("patchfy_full"), full=sz("full_readout"),
                  randinit=sz("randinit"), severed=sz("severed_readout")),
    )
    # verdict on the confound
    m = cmp["material"]
    if m["full"] < m["patchfy"] - 0.02 and abs(m["severed"] - m["patchfy"]) < 0.06:
        confound_verdict = "COUPLING_DESTROYS_MATERIAL (full<patchfy~=severed): cleanest framing, severed=less-destructive"
    elif m["severed"] > m["patchfy"] + 0.02:
        confound_verdict = "SEVERED_BUILDS_MATERIAL (severed>patchfy): stronger 'representation' wording allowed"
    elif abs(m["full"] - m["patchfy"]) < 0.02:
        confound_verdict = "COUPLING_DOES_NOT_DESTROY (full~=patchfy): doubling is something else; re-examine framing"
    else:
        confound_verdict = "MIXED: inspect numbers manually"
    out = dict(rows=rows, compare=cmp, confound_verdict=confound_verdict,
               meta=dict(n_scenes=n, seconds=round(time.time() - t0, 1), device=device, seed=args.seed))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== MATERIAL (local) ===", json.dumps(cmp["material"], indent=1))
    print("=== SIZE (global) ===", json.dumps(cmp["size"], indent=1))
    print("=== CONFOUND VERDICT ===", confound_verdict)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
