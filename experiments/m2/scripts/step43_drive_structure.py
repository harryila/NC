"""STEP 43 — IS THE COUPLING DRIVE OBJECT-STRUCTURED? (resolving the step42 paradox). step42 found the coupling drive
DOMINATES the tangent update (||g_J||/||g_c|| ~ 20x) and zeroing J changes the state ~147% -- YET severing coupling
(R5d, retrained) leaves accuracy + object-phase UNCHANGED, and the frozen-weights J-zero obj_ami barely drops. The
paradox: the coupling does MOST of the moving, so why is NONE of it object-discriminative?
HYPOTHESIS: g_J = Proj_x(Jx) is large but OBJECT-AGNOSTIC (a smooth common-mode field that moves all oscillators
together), while the object-cluster structure the readout uses is carried by the small stimulus drive g_c = Proj_x(c)
+ the sphere geometry. TEST: treat each DRIVE as a state and measure its object-aligned-phase clustering
(obj_ami_minus_null, the same metric step40 uses on x). PREDICTION if hypothesis right:
    obj_ami(g_c)  > 0   (the small stimulus drive points in object-clustered directions = it carries the binding info)
    obj_ami(g_J) ~= 0   (the large coupling drive is object-agnostic churn)
Also measure the COMMON-MODE fraction of each drive: ||mean-over-neighborhood|| / ||drive|| -- a high common-mode
fraction for g_J would confirm 'smooth field that moves everything together'. And cos(g_J, g_c): are the two drives
even pointed the same way? Runs on the TRAINED model, no retrain.
Usage: python step43_drive_structure.py --arm R6 --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2/scripts"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, types
import step27_conjunction_binding as s27  # noqa
import step9_fully_online as s9
from step40_synchrony_measure import object_labels_16, _measures
from source.layers.kutils import reshape, reshape_back


def _proj(v_g, x_g):
    return v_g - (x_g * v_g).sum(2, keepdim=True) * x_g


def drive_capture(arm, cg, n, te_imgs, device):
    net = cg.net
    Xk = te_imgs[0][0][:32].to(device)
    labels = np.stack([object_labels_16(Xk[b].cpu().numpy()) for b in range(len(Xk))])
    grabbed = {}  # layer -> list over steps of (gJ_field, gc_field) as (B,C,H,W) numpy

    def make_patched(k, l):
        store = grabbed.setdefault(l, [])
        def patched(self, x, c):
            _y = self.connectivity(x)
            xg = reshape(x, self.n)
            gJ = reshape_back(_proj(reshape(_y, self.n), xg))   # back to (B,C,H,W)
            gc = reshape_back(_proj(reshape(c,  self.n), xg))
            store.append((gJ.detach().cpu().numpy(), gc.detach().cpu().numpy()))
            # exact original update (forward unchanged)
            y = _y + c
            omg_x = self.omg(x) if hasattr(self, "omg") else torch.zeros_like(x)
            yg = reshape(y, self.n)
            if self.apply_proj:
                y_yxx, sim = self.project(yg, xg)
            else:
                y_yxx, sim = yg, yg * xg
            return omg_x + reshape_back(y_yxx), reshape_back(sim)
        return types.MethodType(patched, k)

    orig = {}
    for l in range(net.L):
        kk = net.layers[l][2]; orig[l] = kk.kupdate; kk.kupdate = make_patched(kk, l)
    with torch.no_grad():
        net.feature(Xk)
    for l in range(net.L):
        net.layers[l][2].kupdate = orig[l]

    def common_mode_frac(field):
        # fraction of the drive that is a local common mode: ||3x3 avg|| / ||field||, per-group-dir agnostic (use raw)
        t = torch.tensor(field)
        avg = torch.nn.functional.avg_pool2d(t, 3, 1, 1)
        return float(avg.norm() / max(t.norm().item(), 1e-9))

    out = {}
    L = 1  # the layer step40/the context uses (xs[1])
    steps = grabbed.get(L, [])
    if steps:
        per = []
        for (gJ, gc) in steps:
            mJ = _measures(gJ, labels, n=n)["obj_ami_minus_null"]
            mC = _measures(gc, labels, n=n)["obj_ami_minus_null"]
            # cos between the two drive fields (per-element, mean)
            a, b = torch.tensor(gJ), torch.tensor(gc)
            cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-9))
            per.append(dict(ami_gJ=round(mJ, 4), ami_gc=round(mC, 4), cos_Jc=round(cos, 4),
                            cmf_gJ=round(common_mode_frac(gJ), 4), cmf_gc=round(common_mode_frac(gc), 4)))
        out[f"L{L}"] = dict(
            ami_gJ=[p["ami_gJ"] for p in per], ami_gc=[p["ami_gc"] for p in per],
            cos_Jc=[p["cos_Jc"] for p in per],
            cmf_gJ=[p["cmf_gJ"] for p in per], cmf_gc=[p["cmf_gc"] for p in per])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--out", default="step43_drive.json")
    a = ap.parse_args(); path = "experiments/m2/results/" + a.out
    recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=a.n_tasks, device=a.device, capture_fn=drive_capture)
        recs.append(r); json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        m = (r.get("measure", {}) or {}).get("L1", {})
        print(f"[drive {a.arm} s{s}] ami_gJ={m.get('ami_gJ')} ami_gc={m.get('ami_gc')} "
              f"cos_Jc={m.get('cos_Jc')} cmf_gJ={m.get('cmf_gJ')} cmf_gc={m.get('cmf_gc')}  {r.get('measure_error','')}", flush=True)
    print("STEP43_DONE")
