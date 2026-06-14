"""
GATE A -- ARM A3: native spherical-NORMALIZATION ablation (the decisive synchrony-vs-stability dissociation).

WHY (from the design-verification workflow w7c0mv62k):
  --project False only removes the parameter-free tangent projection; the attention coupling AND the per-step spherical
  normalize() both stay on, so phase-locking still happens -- it is NOT a synchronization-off knob. The op that
  manufactures the unit-sphere "phase" variable and drives the global order parameter R_global is the hardwired
  normalize() at klayer.py:156/161 (there is NO CLI flag for it). To dissociate "normalization = stability" from
  "phase-synchrony", we replace that normalize() with a bounded but NON-spherical norm (clamp: cap per-group radius at
  rmax, dropping the unit-length constraint) while keeping the attention coupling + stimulus ON. We also set
  apply_proj=False because the tangent projection y-<y,x>x assumes ||x||=1, which a non-spherical norm breaks (this is
  exactly the R7clampNP control from the CL work, experiments/m1_wk0/ladder.py:129-141). Parameter-IDENTICAL to full
  AKOrN (the norm replacement is parameter-free).

THE DISSOCIATION:
  A1 project-off  : apply_proj=False, sphere normalize ON   (control)
  A3 norm-ablate  : apply_proj=False, sphere normalize -> clamp (this arm)
  A1 vs A3 isolates the SPHERE NORMALIZATION (both have projection off). Measure FG-ARI AND R_global:
    FG-ARI(A3) ~= FG-ARI(A1) while R_global collapses  => magnitude-bounding (stability) is the workhorse, the
                                                           unit-sphere phase-synchrony is a bystander. (thesis)
    FG-ARI(A3) falls toward ItrSA                        => the unit-sphere / phase-synchrony matters. (thesis falsified)

This ports experiments/m1_wk0/ladder.py {_make_norm_fn,_patched_klayer_forward,build_R7(no_proj)} to the native objs
KLayer (same source.layers.klayer.KLayer class). Wired into train_obj.py / eval_obj.py via --norm_ablate {clamp,soft,layernorm}.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

_AKORN_SRC = os.environ.get("AKORN_SRC", "/root/NC/external/akorn")
if _AKORN_SRC not in sys.path:
    sys.path.insert(0, _AKORN_SRC)

from source.layers.klayer import KLayer  # noqa: E402
from source.layers.kutils import reshape, reshape_back, normalize as sphere_normalize  # noqa: E402


def _make_norm_fn(kind, n, rmax=3.0, lam=1.0):
    """Per-step state-norm replacements (group/within-group sphere dim = 2). Parameter-free."""
    if kind == "spherical":                                   # == full AKOrN (bit-exact control)
        return lambda x: sphere_normalize(x, n)
    if kind == "clamp":                                       # cap per-group radius at rmax; DROP the unit-sphere constraint
        def f(x):
            xr = reshape(x, n); r = xr.norm(dim=2, keepdim=True).clamp_min(1e-8)
            return reshape_back(xr * (r.clamp(max=rmax) / r))
        return f
    if kind == "soft":                                        # dose-response x/(1+lam*(r-1)); lam=1==sphere, lam=0==none
        def f(x):
            xr = reshape(x, n); r = xr.norm(dim=2, keepdim=True).clamp_min(1e-8)
            return reshape_back(xr / (1.0 + lam * (r - 1.0)))
        return f
    if kind == "layernorm":                                   # channel mean/var norm, rescaled to RMS 1/sqrt(n)
        def f(x):
            C = x.shape[1]
            z = F.layer_norm(x.permute(0, 2, 3, 1), (C,)) * (1.0 / (n ** 0.5))
            return z.permute(0, 3, 1, 2).contiguous()
        return f
    raise ValueError(f"unknown norm variant {kind!r}")


def _patched_klayer_forward(self, x, c, T, gamma):
    # BYTE-IDENTICAL to KLayer.forward (klayer.py:152-165) except normalize() -> self._norm_fn
    xs, es = [], []
    c = self.c_norm(c)
    x = self._norm_fn(x)
    es.append(torch.zeros(x.shape[0]).to(x.device))
    for t in range(T):
        dxdt, _sim = self.kupdate(x, c)
        x = self._norm_fn(x + gamma * dxdt)
        xs.append(x)
        es.append((-_sim).reshape(x.shape[0], -1).sum(-1))
    return xs, es


def norm_ablate_akorn(net, variant="clamp", rmax=3.0, lam=1.0, no_proj=True, verbose=True):
    """Replace each KLayer's spherical normalize with `variant`; set apply_proj=False if no_proj.
    Parameter-free (param-identical to full AKOrN). Returns the mutated net."""
    n_patched = 0
    for m in list(net.modules()):  # materialize before mutating
        if isinstance(m, KLayer):
            m._norm_fn = _make_norm_fn(variant, m.n, rmax=rmax, lam=lam)
            m.forward = types.MethodType(_patched_klayer_forward, m)
            if no_proj:
                m.apply_proj = False
            n_patched += 1
    if verbose:
        print(f"  [norm_ablate] patched {n_patched} KLayer(s): variant={variant} "
              f"no_proj={no_proj} rmax={rmax} lam={lam}")
    if n_patched == 0:
        raise RuntimeError("norm_ablate_akorn: no KLayer found to ablate.")
    return net


# ---- self-test / param-identity audit (run on the GPU box) ----
if __name__ == "__main__":
    from source.models.objs.knet import AKOrN

    def build():
        return AKOrN(n=4, ch=256, L=1, T=8, psize=8, gta=True, J="attn", ksize=1, c_norm="gn",
                     gamma=1.0, imsize=128, use_omega=False, init_omg=0.01, global_omg=False,
                     maxpool=True, project=True, heads=8, use_ro_x=False, learn_omg=False,
                     no_ro=False, autorescale=False)

    dev = "cpu" if not torch.cuda.is_available() else "cuda"
    full = build().to(dev)
    abl = build().to(dev)
    norm_ablate_akorn(abl, variant="clamp", no_proj=True)
    nf = sum(p.numel() for p in full.parameters() if p.requires_grad)
    na = sum(p.numel() for p in abl.parameters() if p.requires_grad)
    print(f"full params={nf:,}  norm-ablate params={na:,}  delta={na-nf} (param-free ablation, expect 0)")
    assert na == nf, "norm-ablate must be parameter-identical!"
    k = abl.layers[0][0]
    print(f"apply_proj after ablate (expect False): {k.apply_proj}")
    assert k.apply_proj is False
    with torch.no_grad():
        img = torch.rand(2, 3, 128, 128, device=dev)
        of, ao = full(img), abl(img)
        print(f"forward parity: full {tuple(of.shape)} == ablate {tuple(ao.shape)}: {of.shape == ao.shape}")
        assert of.shape == ao.shape
    print("A3 norm-ablate ready: parameter-identical, clamp+no_proj, forward OK.")
