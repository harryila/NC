"""STEP 44 — COUNTERFACTUAL LADDER (airtight attribution of WHERE object-binding comes from). On the TRAINED R6, at
layer 1, measure object-aligned-phase structure (obj_ami_minus_null, the step40 metric) of the oscillator state under
four counterfactuals, all with FROZEN trained weights, no retrain:
    x_init  : layer-1 input BEFORE any Kuramoto step (normalize(transition(x_layer0)))  -> is structure INHERITED?
    full    : both drives on  (Jx + c)                                                    -> the real model
    J-zero  : coupling output forced to 0 (c only)                                        -> does binding survive w/o coupling?
    c-zero  : stimulus c forced to 0 (Jx only)                                            -> does binding survive w/o stimulus?
DECISIVE PREDICTION (the 'object-agnostic coupling' account from step42/43):
    obj_ami(full) ~= obj_ami(J-zero)  >>  obj_ami(c-zero)
    i.e. binding rides on the STIMULUS drive + inherited structure; the dominant COUPLING drive is object-agnostic and
    removing it is harmless, while removing the (small) stimulus collapses binding.
If instead obj_ami(c-zero) ~= full -> coupling alone suffices (would contradict the inert-coupling story).
If obj_ami(x_init) ~= full -> the structure is INHERITED feedforward, the layer-1 Kuramoto step (either drive) is not
where binding is formed (a sharper, honest refinement). Reports all four + the per-step trajectory.
Usage: python step44_counterfactual_ladder.py --arm R6 --seeds 0 1 2 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2/scripts"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, types
import step27_conjunction_binding as s27  # noqa
import step9_fully_online as s9
from step40_synchrony_measure import object_labels_16, _measures
from source.layers.kutils import normalize as sphere_normalize


def _ami(state, labels, n):
    return _measures(state, labels, n=n)["obj_ami_minus_null"]


def ladder_capture(arm, cg, n, te_imgs, device):
    net = cg.net
    Xk = te_imgs[0][0][:32].to(device)
    labels = np.stack([object_labels_16(Xk[b].cpu().numpy()) for b in range(len(Xk))])
    L = 1

    def run_variant(mode):
        """mode in {'full','jzero','czero'}: hook the connectivity / c to zero the chosen drive. Returns xs[L][-1]
        and (for 'full') the layer-L pre-Kuramoto x_init."""
        hooks = []
        if mode == "jzero":
            hooks = [net.layers[l][2].connectivity.register_forward_hook(
                        lambda m, i, o: torch.zeros_like(o)) for l in range(net.L)]
        cap = {}
        if mode == "czero":
            # patch each KLayer.forward to zero c (the stimulus) just for this pass
            orig_fwd = {}
            for l in range(net.L):
                k = net.layers[l][2]; orig_fwd[l] = k.forward
                def make_f(self, of):
                    def f(x, c, T, gamma):
                        return of(x, torch.zeros_like(c), T, gamma)
                    return f
                k.forward = make_f(k, orig_fwd[l])
        # also grab layer-L pre-Kuramoto x_init via a forward_pre_hook on the KLayer
        def pre_hook(mod, args):
            x_in = args[0]
            cap["x_init"] = sphere_normalize(x_in, mod.n).detach().cpu().numpy()
        ph = net.layers[L][2].register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            _c, _x, xs, _e = net.feature(Xk)
        ph.remove()
        for h in hooks: h.remove()
        if mode == "czero":
            for l in range(net.L): net.layers[l][2].forward = orig_fwd[l]
        return xs[L][-1].detach().cpu().numpy(), cap.get("x_init")

    full_state, x_init = run_variant("full")
    jzero_state, _ = run_variant("jzero")
    czero_state, _ = run_variant("czero")
    return dict(
        ami_x_init=round(_ami(x_init, labels, n), 4),
        ami_full=round(_ami(full_state, labels, n), 4),
        ami_Jzero=round(_ami(jzero_state, labels, n), 4),
        ami_czero=round(_ami(czero_state, labels, n), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--out", default="step44_ladder.json")
    a = ap.parse_args(); path = "experiments/m2/results/" + a.out
    recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        r = s9.run(a.arm, s, n_tasks=a.n_tasks, device=a.device, capture_fn=ladder_capture)
        recs.append(r); json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
        m = r.get("measure", {}) or {}
        print(f"[ladder {a.arm} s{s}] x_init={m.get('ami_x_init')} full={m.get('ami_full')} "
              f"Jzero={m.get('ami_Jzero')} czero={m.get('ami_czero')}  {r.get('measure_error','')}", flush=True)
    print("STEP44_DONE")
