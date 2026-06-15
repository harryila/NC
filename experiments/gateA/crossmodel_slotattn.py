"""
CROSS-MODEL arm (PREREG-downstream.md): does the local->global grouping reallocation reproduce in Slot Attention?
SA is a 2nd, architecturally-distinct grouping mechanism (iterative cross-attention + GRU bottleneck, NO Kuramoto).
Pretrained CLEVRTex checkpoint (HHousen/object-discovery-pytorch, clevrtex_sa, 7 slots/3 iters/128px) -> EVAL ONLY.

SA grouping-strength ladder (each on SA's OWN no-grouping->grouped axis, fixed params; prereg-sanctioned):
  - sa_encoder : pre-grouping CNN encoder feature map, GT-mask-pooled  (UNGROUPED floor; analog of AKOrN severed/raw)
  - sa_slots_i1: slot vectors with num_iterations=1 (WEAK grouping)
  - sa_slots_i3: slot vectors with num_iterations=3 (FULL grouping, trained value; analog of AKOrN full)
Each object's slot = the slot whose decoder mask most overlaps that object's GT region (assignment decoupled from FG-ARI).
SAME GT masks given to every arm => conservative (grouped arms get free localization yet must still lose material).

TRUST GATE: FG-ARI of slots_i3 (argmax decoder mask vs GT, foreground only) ~= SA's reported CLEVRTex (~0.6 in-dist).
DECISION (cross-model, "same DIRECTION" per prereg): mAP_material(encoder) > mAP_material(slots_i3) AND
  mAP_size(slots_i3) >= mAP_size(encoder)  [+ monotone enc>i1>i3 material, FG-ARI i3>i1, Spearman(FG-ARI,mAP_mat)<=-0.5].

USAGE (box, CPU, no GPU contention with the n>=3 campaign):
  python crossmodel_slotattn.py --data_type outd --data_root /root/data/clevrtex/clevrtex_outd \
    --n_scenes 2000 --device cpu --out results/crossmodel_sa_outd.json
"""
import os, sys, json, argparse
from argparse import Namespace
import numpy as np

_GA = "/root/NC/experiments/gateA"
_SA = "/root/NC/external/object-discovery-pytorch"
_AK = "/root/NC/external/akorn"
for p in (_GA, _SA, _AK, "/root/NC"):
    if p not in sys.path:
        sys.path.insert(0, p)

import native_retrieval as NR                                   # reuse the LOCKED retrieval() verbatim
ATTRS = ["material", "size", "shape"]


def load_sa(ckpt_path, device):
    """Bypass pytorch_lightning: load the state_dict straight into the bare SlotAttentionModel (plain nn.Module).
    All ckpt keys are prefixed 'model.' (LightningModule.self.model = SlotAttentionModel); strip it.
    object_discovery.utils does `from pytorch_lightning import Callback` at import time -> stub it (we never train)."""
    import types, torch
    if "pytorch_lightning" not in sys.modules:
        _pl = types.ModuleType("pytorch_lightning")
        _pl.Callback = type("Callback", (), {})
        sys.modules["pytorch_lightning"] = _pl
    from object_discovery.slot_attention_model import SlotAttentionModel
    ck = torch.load(ckpt_path, map_location="cpu")
    hp = Namespace(**ck["hyper_parameters"])
    sa = SlotAttentionModel(resolution=hp.resolution, num_slots=hp.num_slots,
                            num_iterations=hp.num_iterations, slot_size=hp.slot_size)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = sa.load_state_dict(sd, strict=False)
    assert not unexpected, "unexpected keys: %s" % unexpected[:5]
    assert not missing, "missing keys: %s" % missing[:5]
    sa.eval().to(device)
    return sa, hp


def load_scenes(args):
    """Reuse AKOrN's loader EXACTLY as native_retrieval.extract_fresh does -> identical GT masks + attribute labels."""
    import native_usefulness as NU
    from source.data.datasets.objs.clevr_tex import get_clevrtex
    ds = get_clevrtex(args.data_root, split="test", data_type=args.data_type, imsize=128, return_meta_data=True)
    scenes = []
    for i in range(min(args.n_scenes, len(ds))):
        ind, img, msk = ds[i][0], ds[i][1], ds[i][2]
        msk = np.asarray(msk)[0].astype(np.int64)
        raw = json.load(open(ds.metadata_index[ind]))
        by_id = {int(o["index"]): {a: str(o.get(a, "NA")) for a in NU.ATTRS} for o in raw["objects"]}
        scenes.append((img.float(), msk, by_id))
    print("[data] %s: %d scenes, %d GT objects" % (args.data_type, len(scenes),
          sum(len(s[2]) for s in scenes)), flush=True)
    return scenes


def fg_ari(gt, pred):
    """FG-ARI on foreground pixels only (gt>0). gt,pred = [H,W] int label maps."""
    from sklearn.metrics import adjusted_rand_score
    fg = gt.reshape(-1) > 0
    if fg.sum() < 2:
        return float("nan")
    g, p = gt.reshape(-1)[fg], pred.reshape(-1)[fg]
    if len(np.unique(g)) < 2:
        return float("nan")
    return float(adjusted_rand_score(g, p))


def extract(model, scenes, device, iters=(1, 3)):
    """One pass: encoder GT-pool + slot vectors at each iter count + per-arm slot-assignment + FG-ARI. Aligned oids."""
    import torch
    # hook the encoder feature map (input to slot attention) [B,HW,C] AND the slot_attention output [B,S,C]
    # (model.forward RETURNS slots reshaped to [B*S,C,1,1] for the decoder, so grab the real slots via hook)
    cap = {}
    h = model.encoder_out_layer.register_forward_hook(lambda m, i, o: cap.__setitem__("enc", o.detach()))
    hs = model.slot_attention.register_forward_hook(lambda m, i, o: cap.__setitem__("slots", o.detach()))

    enc_rows, slot_rows = [], {it: [] for it in iters}
    raw_rows, encmask_rows, enchard_rows = [], [], []   # CONTROLS: raw floor; encoder pooled by SOFT slot mask; by HARD slot region
    labs = {a: [] for a in ATTRS}
    sids = []
    aris = {it: [] for it in iters}
    sc = 0
    with torch.no_grad():
        for img, msk, by_id in scenes:
            x = (img * 2.0 - 1.0).unsqueeze(0).to(device)               # AKOrN [0,1] -> SA [-1,1]
            H, W = msk.shape
            slots_by_it, masks_by_it = {}, {}
            for it in iters:
                model.slot_attention.num_iterations = it
                _, _, masks, _ = model.forward(x)                       # masks [1,S,1,H,W]
                slots_by_it[it] = cap["slots"][0].cpu().numpy()        # [S,C] (true slots, via hook)
                masks_by_it[it] = masks[0, :, 0].cpu().numpy()          # [S,H,W]
            enc_map = cap["enc"][0].cpu().numpy().reshape(H, W, -1)     # [H,W,C] (encoder is stride-1, no upsample)
            # FG-ARI per iter (argmax slot mask = predicted segmentation)
            for it in iters:
                pred = masks_by_it[it].argmax(0)                        # [H,W]
                a = fg_ari(msk, pred)
                if a == a:
                    aris[it].append(a)
            hardseg3 = masks_by_it[3].argmax(0)                        # [H,W] hard slot segmentation at i3
            # per GT object (identical skip logic to pool_objects -> aligned across all arms)
            flat = msk.reshape(-1)
            for v in np.unique(flat):
                if v == 0:
                    continue
                cells = (msk == v)
                if cells.sum() < 4:
                    continue
                vid = int(v)
                if vid not in by_id:
                    continue
                enc_rows.append(enc_map[cells].reshape(-1, enc_map.shape[-1]).mean(0))
                raw_rows.append(img.numpy()[:, cells].mean(1))          # CONTROL: raw-pixel floor (3-dim GT-pooled RGB)
                assigned = {}
                for it in iters:
                    ov = masks_by_it[it][:, cells].sum(1)               # [S] overlap of each slot mask w/ object
                    assigned[it] = int(ov.argmax())
                    slot_rows[it].append(slots_by_it[it][assigned[it]])
                # CONTROL: pool the SAME encoder features by the assigned slot's decoder mask (grouped AGGREGATION,
                # no GRU/MLP bottleneck) -> isolates "grouped aggregation footprint" from "slot bottleneck transform".
                ef = enc_map.reshape(-1, enc_map.shape[-1])             # [HW,C]
                w = masks_by_it[3][assigned[3]].reshape(-1)             # [HW] assigned-slot SOFT alpha mask (i3)
                encmask_rows.append((w[:, None] * ef).sum(0) / (w.sum() + 1e-9))
                # CONTROL 2 (HARD): pool encoder by the slot's HARD region (pixels where this slot WINS the argmax).
                # Separates grouped-AGGREGATION-FOOTPRINT from SOFT-mask off-object dilution. Falls back to soft if empty.
                hw = (hardseg3 == assigned[3]).astype("float32").reshape(-1)
                if hw.sum() < 4:
                    hw = w
                enchard_rows.append((hw[:, None] * ef).sum(0) / (hw.sum() + 1e-9))
                for a in ATTRS:
                    labs[a].append(by_id[vid].get(a, "NA"))
                sids.append(sc)
            sc += 1
    h.remove(); hs.remove()
    arms = {"sa_raw": np.stack(raw_rows).astype("float32"),
            "sa_encoder": np.stack(enc_rows).astype("float32"),
            "sa_encmask_i3": np.stack(encmask_rows).astype("float32"),
            "sa_enchard_i3": np.stack(enchard_rows).astype("float32")}
    for it in iters:
        arms["sa_slots_i%d" % it] = np.stack(slot_rows[it]).astype("float32")
    sc_arr = np.array(sids)
    lab_arr = {a: np.array(labs[a]) for a in ATTRS}
    fgari = {("sa_slots_i%d" % it): (float(np.mean(aris[it])) if aris[it] else float("nan")) for it in iters}
    for k in ("sa_encoder", "sa_raw", "sa_encmask_i3", "sa_enchard_i3"):
        fgari[k] = 0.0                                                  # ungrouped floors
    return arms, lab_arr, sc_arr, fgari


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/NC/external/object-discovery-pytorch/checkpoints/clevrtex_sa.ckpt")
    ap.add_argument("--data_type", default="outd", choices=["full", "outd", "camo"])
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_outd")
    ap.add_argument("--n_scenes", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/results/crossmodel_sa.json")
    ap.add_argument("--rescore_npz", default=None,
                    help="re-score saved features (skip model): recompute retrieval+verdict offline w/ current logic")
    args = ap.parse_args()

    if args.rescore_npz:                                                # OFFLINE re-score: no model, reuse saved features
        d = np.load(args.rescore_npz, allow_pickle=True)
        nms = sorted({k.rsplit("__", 1)[0] for k in d.files})
        arms = {nm: d[nm + "__X"] for nm in nms}
        sc = d[nms[0] + "__sc"]
        labs = {a: d[nms[0] + "__lab_" + a] for a in ATTRS}
        fgari = json.load(open(args.out))["_verdict"]["fgari"]          # carry FG-ARI from the existing json
        print("[rescore] %d arms from %s" % (len(arms), args.rescore_npz), flush=True)
    else:
        import torch
        dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        model, hp = load_sa(args.ckpt, dev)
        print("[model] SA loaded: %d slots / %d iters / slot_size %d / %s" % (
            hp.num_slots, hp.num_iterations, hp.slot_size, hp.resolution), flush=True)
        scenes = load_scenes(args)
        arms, labs, sc, fgari = extract(model, scenes, dev, iters=(1, 3))
        print("[FG-ARI trust gate] " + " ".join("%s=%.3f" % (k, fgari[k]) for k in fgari), flush=True)

    # encode labels per attr (factorize strings -> ints) and run the LOCKED retrieval per arm/attr
    res = {}
    for nm, X in arms.items():
        print("[shape] %s X=%s sc=%s" % (nm, X.shape, sc.shape), flush=True)
        res[nm] = {}
        for a in ATTRS:
            y = labs[a]
            _, yi = np.unique(y, return_inverse=True)
            res[nm][a] = NR.retrieval(X, yi, sc, seed=args.seed)
        print("%-14s | " % nm + " | ".join("%s mAP=%.3f[%.3f,%.3f] R@1=%.3f (c%.3f)" % (
            a, res[nm][a]["mAP"], res[nm][a]["ci"][0], res[nm][a]["ci"][1],
            res[nm][a]["R1"], res[nm][a]["chance"]) for a in ATTRS), flush=True)

    # ---- SA verdict: HONEST SCOPE = DESTRUCTION LEG ONLY (mAP-primary; harmonized w/ AKOrN prereg rule) ----
    # Post adversarial-review (wm01luswy): SA corroborates ONLY the local-destruction direction. We apply the SAME
    # prereg gate AKOrN uses (native_retrieval.py): material destruction CI-separated AND the size-BUILD gate (which SA
    # is EXPECTED TO FAIL) — reported explicitly so the asymmetry is disclosed, not hidden. mAP is primary; R@1 secondary.
    enc, s3, s1 = res["sa_encoder"], res["sa_slots_i3"], res["sa_slots_i1"]

    def ci_sep_gt(a, b):  # a's mAP CI-low strictly above b's mAP CI-high
        return a["ci"][0] > b["ci"][1]
    # CIs are anti-conservative at nq~12617 (i.i.d.-query bootstrap); require a real EFFECT SIZE + cross-metric agreement,
    # not bare CI-separation (per adversarial review). MIN_DELTA on mAP guards against tight-CI false positives.
    MIN_DELTA = 0.01
    def real_gt(a, b):  # a meaningfully beats b: CI-sep AND mAP-delta>=MIN_DELTA AND R@1 agrees
        return ci_sep_gt(a, b) and (a["mAP"] - b["mAP"] >= MIN_DELTA) and (a["R1"] > b["R1"])
    # DESTRUCTION leg (the claim we make for SA): ungrouped encoder material > grouped slots
    destruction_mAP = enc["material"]["mAP"] > s3["material"]["mAP"]
    destruction_ci = real_gt(enc["material"], s3["material"])
    destruction_R1 = enc["material"]["R1"] > s3["material"]["R1"]   # secondary corroboration
    # SIZE-BUILD gate (AKOrN-style, prereg cond iii) applied to SA — EXPECTED FALSE (size is a spatial-context confound).
    # Now uses real_gt (effect-size + R@1 agreement) so a +0.002 mAP tight-CI artifact does NOT pass.
    size_build_gate = real_gt(s3["size"], enc["size"])
    # n_iters monotonicity (prereg SA gate) — EXPECTED FALSE (slots_i1 < slots_i3)
    monotone_mat = enc["material"]["mAP"] >= s1["material"]["mAP"] >= s3["material"]["mAP"]
    order = ["sa_encoder", "sa_slots_i1", "sa_slots_i3"]
    rho = float("nan")
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr([fgari[k] for k in order], [res[k]["material"]["mAP"] for k in order]).correlation)
    except Exception:
        pass
    verdict = dict(
        SA_DESTRUCTION_LEG_CONFIRMED=bool(destruction_mAP and destruction_ci),   # the ONLY claim made for SA
        destruction_material_mAP_enc_gt_slots=bool(destruction_mAP), destruction_mAP_ci_separated=bool(destruction_ci),
        destruction_material_R1_corroborates=bool(destruction_R1),
        size_build_gate_passed=bool(size_build_gate),               # EXPECTED FALSE: size is not grouping-specific (confound)
        n_iters_monotonicity_passed=bool(monotone_mat),             # EXPECTED FALSE: slots_i1 < slots_i3
        spearman_fgari_material=rho,
        NOTE="SA = local-DESTRUCTION corroboration only. size-build + monotonicity gates expected FALSE (retired as confounds). "
             "Lead with mAP; R@1 secondary. NOT equivalent to AKOrN INVERSION_CONFIRMED.",
        mAP_material=dict(encoder=enc["material"]["mAP"], slots_i1=s1["material"]["mAP"], slots_i3=s3["material"]["mAP"]),
        R1_material=dict(encoder=enc["material"]["R1"], slots_i1=s1["material"]["R1"], slots_i3=s3["material"]["R1"]),
        mAP_size=dict(encoder=enc["size"]["mAP"], slots_i1=s1["size"]["mAP"], slots_i3=s3["size"]["mAP"]),
        fgari=fgari)
    res["_verdict"] = verdict
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    # save pooled features so CIs/floors/cluster-bootstrap can be RE-SCORED offline without re-running the model
    npz_path = args.out.replace(".json", "_features.npz")
    save = {}
    for nm, X in arms.items():
        save[nm + "__X"] = X
        save[nm + "__sc"] = sc
        for a in ATTRS:
            save[nm + "__lab_" + a] = labs[a]
    np.savez_compressed(npz_path, **save)
    print("\n=== SA VERDICT (destruction-leg scope) ===", json.dumps(verdict, indent=1), flush=True)
    print("wrote", args.out, "and", npz_path, flush=True)


if __name__ == "__main__":
    main()
