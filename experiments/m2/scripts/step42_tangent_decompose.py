"""STEP 42 — TANGENT-SPACE DECOMPOSITION (WHY is the coupling inert?). Not content with "R5d==R6, coupling inert":
this OPENS the black box. The AKOrN Kuramoto step (klayer.py:152-165) is Riemannian gradient flow on the product of
unit spheres:  x <- normalize(x + gamma*[ Omega x + Proj_x(Jx + c) ]).  Proj is LINEAR, so the tangent update splits
EXACTLY into two drives:
    g_J = Proj_x(Jx)   (lateral COUPLING drive -- the 'synchrony' term)
    g_c = Proj_x(c)    (feed-forward STIMULUS drive -- the conditioning, constant over the T steps)
MECHANISTIC HYPOTHESIS: severing J is inert because ||g_J|| << ||g_c|| -- the coupling never steered the trajectory;
the oscillators just align to their stimulus c on the sphere. This script MEASURES it on the TRAINED model (no retrain):
  - per-step, per-group ||g_J||, ||g_c||, ||Omega x||  and ratio ||g_J||/||g_c||
  - cos(g_J, g_c): is the coupling even pushing in a useful direction or orthogonal noise?
  - align(x_t, c): does x converge toward the stimulus over T? (the 'alignment' picture)
  - FROZEN-WEIGHTS J-ZERO COUNTERFACTUAL: rerun the SAME trained net with connectivity output forced to 0; compare
    final-layer phase obj_ami(J-zero) vs obj_ami(full). Stronger than R5d (which retrains a different arch): if the
    object-phase structure is unchanged with the EXACT trained weights minus the coupling output, coupling is proven
    a bystander analytically, not just by parameter-matched retrain.
Usage: python step42_tangent_decompose.py --arm R6 --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2/scripts"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, types
import step27_conjunction_binding as s27  # noqa: patches shp -> conjunction data
import step9_fully_online as s9
from step40_synchrony_measure import object_labels_16, _measures
from source.layers.kutils import reshape, reshape_back


def _proj(v_g, x_g):
    """tangent projection per group: v - <v,x>x, summing over the within-group channel dim (dim 2)."""
    sim = (x_g * v_g).sum(2, keepdim=True)
    return v_g - sim * x_g


def _gnorm(v_g):
    return v_g.norm(dim=2)  # (B,G,H,W) per-group vector norm


def decompose_capture(arm, cg, n, te_imgs, device):
    net = cg.net
    Xk = te_imgs[0][0][:32].to(device)
    rec = {l: [] for l in range(net.L)}

    def make_patched(k, l):
        def patched(self, x, c):
            _y = self.connectivity(x)               # Jx (coupling output)
            y = _y + c
            omg_x = self.omg(x) if hasattr(self, "omg") else torch.zeros_like(x)
            xg = reshape(x, self.n)
            gJ = _proj(reshape(_y, self.n), xg)     # coupling tangent drive
            gc = _proj(reshape(c,  self.n), xg)     # stimulus tangent drive
            og = reshape(omg_x, self.n)
            nJ, nc, no = _gnorm(gJ), _gnorm(gc), _gnorm(og)
            cosJc = (gJ * gc).sum(2) / (nJ * nc).clamp_min(1e-9)
            align = (xg * reshape(c, self.n)).sum(2) / (_gnorm(xg) * _gnorm(reshape(c, self.n))).clamp_min(1e-9)
            rec[l].append(dict(nJ=float(nJ.mean()), nc=float(nc.mean()), no=float(no.mean()),
                               ratio_JC=float((nJ / nc.clamp_min(1e-9)).mean()),
                               cos_JC=float(cosJc.mean()), align_xc=float(align.mean())))
            # exact original update (so the forward is unchanged)
            yg = reshape(y, self.n)
            if self.apply_proj:
                y_yxx, sim = self.project(yg, xg)
            else:
                y_yxx, sim = yg, yg * xg
            return omg_x + reshape_back(y_yxx), reshape_back(sim)
        return types.MethodType(patched, k)

    orig = {}
    for l in range(net.L):
        k = net.layers[l][2]; orig[l] = k.kupdate; k.kupdate = make_patched(k, l)
    with torch.no_grad():
        _c, _x, xs_full, _e = net.feature(Xk)
    for l in range(net.L):
        net.layers[l][2].kupdate = orig[l]

    labels = np.stack([object_labels_16(Xk[b].cpu().numpy()) for b in range(len(Xk))])
    full_state = xs_full[1][-1].detach().cpu().numpy()
    m_full = _measures(full_state, labels, n=n)

    # FROZEN-WEIGHTS J-ZERO COUNTERFACTUAL: force connectivity output -> 0 (Jx=0) with the SAME trained weights.
    # A forward-hook that returns a value REPLACES the module output, so connectivity(x) -> 0 (coupling severed),
    # while every other trained weight (omega, c_norm, readout, project) is byte-identical.
    hooks = [net.layers[l][2].connectivity.register_forward_hook(
                 lambda mod, inp, out: torch.zeros_like(out)) for l in range(net.L)]
    with torch.no_grad():
        _cz, _xz, xs_z, _ez = net.feature(Xk)
    for h in hooks:
        h.remove()
    zero_state = xs_z[1][-1].detach().cpu().numpy()
    m_zero = _measures(zero_state, labels, n=n)
    rel_change = float(np.linalg.norm(zero_state - full_state) / max(np.linalg.norm(full_state), 1e-9))

    # summarize per-layer step trajectories
    summ = {}
    for l in range(net.L):
        steps = rec[l]
        summ[f"L{l}"] = dict(
            ratio_JC=[round(s["ratio_JC"], 4) for s in steps],
            nJ=[round(s["nJ"], 4) for s in steps], nc=[round(s["nc"], 4) for s in steps],
            no=[round(s["no"], 4) for s in steps],
            cos_JC=[round(s["cos_JC"], 4) for s in steps], align_xc=[round(s["align_xc"], 4) for s in steps])
    return dict(decomp=summ,
                obj_ami_full=round(m_full["obj_ami_minus_null"], 4),
                obj_ami_Jzero=round(m_zero["obj_ami_minus_null"], 4),
                Jzero_rel_state_change=round(rel_change, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--out", default="step42_decomp.json")
    a = ap.parse_args(); path = "experiments/m2/results/" + a.out
    recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=a.n_tasks, device=a.device, capture_fn=decompose_capture)
        recs.append(r); json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        m = r.get("measure", {}) or {}; d = m.get("decomp", {}).get("L1", {})
        print(f"[decomp {a.arm} s{s}] ratio_JC(L1)={d.get('ratio_JC')} align_xc={d.get('align_xc')} "
              f"| obj_ami full={m.get('obj_ami_full')} Jzero={m.get('obj_ami_Jzero')} "
              f"relΔstate={m.get('Jzero_rel_state_change')}  {r.get('measure_error','')}", flush=True)
    print("STEP42_DONE")
