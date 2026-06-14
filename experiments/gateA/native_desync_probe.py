"""
GATE A -- DESYNCHRONIZATION PROBE (the sync-necessity test the norm/projection ablations could not isolate).

The norm/projection arms keep the coupling on, and the coupling produces global phase synchronization in EVERY arm
(R_global stays high even with the sphere ablated). So they cannot answer: is the phase synchronization itself
causally necessary for binding, or an epiphenomenon of the attention routing? This probe attacks it directly.

METHOD (eval-time, FROZEN trained model -- the phase analog of the frozen J-zero coupling counterfactual):
  During the Kuramoto recurrence we inject per-step, per-token directional Gaussian NOISE of strength sigma into the
  oscillator state (then renormalize), while leaving the attention coupling / routing fully intact. Increasing sigma
  scrambles relative phases -> drives the global order parameter R_global down. We sweep sigma and, at each level,
  measure R_global AND FG-ARI/MBO (via the exact readout+clustering from native_decompose).
INTERPRETATION:
  FG-ARI stays ~flat while R_global collapses  => phase synchronization is INERT for the readout -> binding does not
                                                  ride on phase coherence (strong synchrony-skeptic result, native).
  FG-ARI falls in lockstep with R_global       => phase synchronization IS load-bearing -> honest entangled result.
CAVEAT: frozen intervention (model did NOT train with noise), so it tests whether the trained model's output DEPENDS
  on phase coherence -- the upper-bound analog of frozen J-zero; a trained-with-noise arm is the confirmatory version.

Runs on the FULL AKOrN checkpoint (ema_499). CPU-friendly (small probe) to avoid GPU contention / OOM.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_GA = "/root/NC/experiments/gateA"
if _GA not in sys.path:
    sys.path.insert(0, _GA)
_SRC = os.environ.get("AKORN_SRC", "/root/NC/external/akorn")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import native_decompose as nd  # noqa: E402  (build_net, load_checkpoint, load_probe_batch, eval_fgari_mbo, _order_param)
from source.layers.kutils import reshape, normalize as sphere_normalize  # noqa: E402


def make_noisy_forward(sigma):
    """KLayer.forward with per-step directional noise sigma injected into the state (routing untouched).
    Byte-identical to klayer.py:152-165 except the post-update noise + renormalize."""
    def fwd(self, x, c, T, gamma):
        xs, es = [], []
        c = self.c_norm(c)
        x = sphere_normalize(x, self.n)
        es.append(torch.zeros(x.shape[0]).to(x.device))
        for t in range(T):
            dxdt, _sim = self.kupdate(x, c)
            x = sphere_normalize(x + gamma * dxdt, self.n)
            if sigma > 0:
                x = sphere_normalize(x + sigma * torch.randn_like(x), self.n)   # DESYNC: scramble phase
            xs.append(x)
            es.append((-_sim).reshape(x.shape[0], -1).sum(-1))
        return xs, es
    return fwd


def _patch_all(model, sigma):
    import types
    originals = {}
    for l in range(model.L):
        k = model.layers[l][0]
        originals[l] = k.forward
        k.forward = types.MethodType(make_noisy_forward(sigma), k)
    return originals


def _restore_all(model, originals):
    for l, f in originals.items():
        model.layers[l][0].forward = f


def _rglobal(model, imgs, device):
    with torch.no_grad():
        c, x, xs, es = model.feature(imgs.to(device))
    return float(np.mean([nd._order_param(reshape(xs[l + 1][-1], model.n)) for l in range(model.L)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="clevrtex_full")
    ap.add_argument("--data_root", default="/root/data/clevrtex/clevrtex_full")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--n_images", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_clusters", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/NC/experiments/gateA/native_desync_probe.json")
    # model config (full AKOrN defaults)
    for name, typ, dflt in [("N", int, 4), ("ch", int, 256), ("L", int, 1), ("T", int, 8),
                            ("gamma", float, 1.0), ("psize", int, 8), ("ksize", int, 1), ("heads", int, 8),
                            ("init_omg", float, 0.01)]:
        ap.add_argument(f"--{name}", type=typ, default=dflt)
    ap.add_argument("--J", type=str, default="attn")
    ap.add_argument("--c_norm", type=str, default="gn")
    ap.add_argument("--norm_ablate", type=str, default="none")
    for name in ["use_omega", "global_omg", "learn_omg", "use_ro_x", "no_ro", "autorescale"]:
        ap.add_argument(f"--{name}", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--maxpool", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--project", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--gta", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--model_imsize", type=int, default=None)
    ap.add_argument("--data_imsize", type=int, default=None)
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    imgs, gt, imsize, meta = nd.load_probe_batch(args)
    net = nd.build_net(args, imsize)
    model, load_info = nd.load_checkpoint(net, args.ckpt, device)
    print("load:", load_info.get("load_kind"), "missing:", load_info.get("n_missing"))

    rows = []
    for sigma in args.sigmas:
        torch.manual_seed(args.seed)
        orig = _patch_all(model, sigma)
        rg = _rglobal(model, imgs, device)
        torch.manual_seed(args.seed)
        metrics, _ = nd.eval_fgari_mbo(model, imgs, gt, args.n_clusters, device)
        _restore_all(model, orig)
        row = dict(sigma=sigma, R_global=round(rg, 4),
                   fgari=round(metrics["fgari"], 4), mbo=round(metrics["mbo"], 4))
        rows.append(row)
        print(f"  sigma={sigma:<5} R_global={row['R_global']:<7} FG-ARI={row['fgari']:<7} MBO={row['mbo']}", flush=True)

    json.dump({"rows": rows, "ckpt": args.ckpt, "n_images": len(imgs)}, open(args.out, "w"), indent=2)
    print("DESYNC_PROBE_DONE ->", args.out)
