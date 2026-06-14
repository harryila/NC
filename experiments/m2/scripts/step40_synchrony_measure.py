"""STEP 40 — SYNCHRONY MEASURE (the dissociation test). On the TRAINED oscillator of each arm, capture the
layer-1 phase state on the conjunction test images (known 3-object color masks) and measure whether the phases are
OBJECT-STRUCTURED-SYNCHRONIZED. Verified-correct design (4-lens workflow):
  - obj_ami_minus_null  : spherical-kmeans phase clusters vs object masks (AMI), minus a label-shuffle null. PRIMARY.
  - deltaR = R_within - R_between : within-object phase coherence minus across-object (binding-by-synchrony signature).
  - R_global : global Kuramoto order parameter (supporting; ambiguous alone).
PREDICTION if dissociation real: R6/R6s HIGH obj_ami_minus_null & deltaR>0; R5d ~=0 (coupling severed -> no
object-structured synchrony) -- WHILE all bind equally. FALSIFIER: R5d also object-aligned -> recurrence itself
synchronizes. eval_inits=1 in step9 -> single clean forward (no init-averaging hazard).
Usage: python step40_synchrony_measure.py --arm R6 --seeds 0 1 2 3 4 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2/scripts"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch
import step27_conjunction_binding as s27   # patches shp -> conjunction data (3-obj colorxshape, 6cls/3task)
import h3
import step9_fully_online as s9

def object_labels_16(img32):
    """img (3,32,32) -> (16,16) object label: 0/1/2 = R/G/B object, -1 = background. majority-pool 2x2 (32->16)."""
    c = np.asarray(img32); maxc = c.max(0); arg = c.argmax(0)
    lab32 = np.where(maxc > 0.5, arg, -1)
    lab16 = np.full((16, 16), -1, int)
    for i in range(16):
        for j in range(16):
            blk = lab32[2*i:2*i+2, 2*j:2*j+2].ravel(); blk = blk[blk >= 0]
            if len(blk):
                v, cnt = np.unique(blk, return_counts=True); lab16[i, j] = v[cnt.argmax()]
    return lab16

def _group_dirs(state, n=4):
    B, C, H, W = state.shape; G = C // n
    a = state.reshape(B, G, n, H*W).transpose(0, 1, 3, 2)            # (B,G,HW,n)
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)

def _measures(state, labels16, n=4, seed=0):
    """PER-GROUP measures (each of the G oscillator-groups has its OWN sphere gauge, so compute within-group then
    average over groups). For each image b and group g: cluster that group's spatial oscillators (one per site) by
    phase and compare to the object masks; within/between-object resultant lengths; global order parameter."""
    B, C, H, W = state.shape; G = C // n
    U = _group_dirs(state, n); lab = labels16.reshape(B, H*W); rng = np.random.default_rng(seed)
    Rw_l, Rb_l, ami_l, amin_l, Rg_l = [], [], [], [], []
    for b in range(B):
        labb = lab[b]; objs = [o for o in np.unique(labb) if o >= 0]
        if len(objs) < 2: continue
        sites = np.where(labb >= 0)[0]; obj_at = labb[sites]            # object label per colored site
        perm = rng.permutation(len(sites))                              # shared shuffle null for this image
        for g in range(G):
            Ug = U[b, g]                                               # (HW, n) one oscillator per spatial site
            cents, Rw = [], []
            for o in objs:
                V = Ug[np.where(labb == o)[0]]; mv = V.mean(0)
                Rw.append(float(np.linalg.norm(mv))); cents.append(mv / max(np.linalg.norm(mv), 1e-12))
            cents = np.array(cents)
            Rw_l.append(np.mean(Rw)); Rb_l.append(float(np.linalg.norm(cents.mean(0))))
            Vc = Ug[sites]; Rg_l.append(float(np.linalg.norm(Vc.mean(0))))
            assign = h3.spherical_kmeans(Vc, k=len(objs), seed=seed)    # phase-cluster the colored sites
            ami_l.append(h3._ami(assign, obj_at)); amin_l.append(h3._ami(assign, obj_at[perm]))
    return {"R_within": float(np.mean(Rw_l)), "R_between": float(np.mean(Rb_l)),
            "deltaR": float(np.mean(Rw_l) - np.mean(Rb_l)), "R_global": float(np.mean(Rg_l)),
            "obj_ami": float(np.mean(ami_l)), "obj_ami_null": float(np.mean(amin_l)),
            "obj_ami_minus_null": float(np.mean(ami_l) - np.mean(amin_l)), "n_groupimgs": int(len(Rw_l))}

def capture_fn(arm, cg, n, te_imgs, device):
    Xk = te_imgs[0][0][:32]                                          # 32 test images from task 0
    from h3 import _seeded as hs
    with torch.no_grad(), hs(0):
        _c, _x, xs, _e = cg.net.feature(Xk.to(device))
    state = xs[1][-1].detach().cpu().numpy()
    labels = np.stack([object_labels_16(Xk[b].numpy()) for b in range(len(Xk))])
    return _measures(state, labels, n=n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_tasks", type=int, default=3); ap.add_argument("--out", default="step40_sync.json")
    a = ap.parse_args(); path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=a.n_tasks, device=a.device, capture_fn=capture_fn); recs.append(r)
        json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        m = r.get("measure", {}) or {}
        print(f"[sync {a.arm} s{s}] acc={r['final_acc']:.3f} | obj_ami-null={m.get('obj_ami_minus_null',float('nan')):.3f} "
              f"deltaR={m.get('deltaR',float('nan')):.3f} R_global={m.get('R_global',float('nan')):.3f}  {r.get('measure_error','')}", flush=True)
    print("STEP40_DONE")
