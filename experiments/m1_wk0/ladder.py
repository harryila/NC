"""
M1 control-ladder construction, built on the REAL AKOrN classification model.

Synchrony's causal effect = R6 - R5 (a single `apply_proj` flip inside identical KLayer machinery).
  R1 dense scalar   R2 +structured sparsity   R3 +vector coding
  R4 +spherical norm   R5 +recurrence, NO synchrony coupling   R6 full AKOrN

R5/R6 are wired directly from the repo's KLayer (it already exposes `apply_proj`,
which IS the tangent-space synchrony projection). R1-R4 use a non-oscillator
LadderCore -- scaffolded here with explicit TODOs for the Wk1-2 build.

!!! UNTESTED SCAFFOLDING. Validate every instantiation in the Wk-0 spike. !!!
Requires the akorn repo on PYTHONPATH (e.g. `export PYTHONPATH=/path/to/akorn`).
See anatomy.md for the line-level code map this is built against.
"""
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401  -- makes the pinned external/akorn importable; MUST precede `source.*`
from source.models.classification.knet import AKOrN
from source.layers.klayer import KLayer
from source.layers.kutils import normalize as sphere_normalize, reshape, reshape_back

# Keep IDENTICAL across all rungs; only the per-layer core (index 2) changes.
# NOTE norm="gn" (not the repo default "bn") -- BN running-stat drift is a CL confound.
BACKBONE = dict(
    n=4, ch=64, L=3, T=3, J="conv", ksizes=[9, 7, 5], ro_ksize=3, ro_N=2,
    norm="gn", c_norm="gn",
    use_omega=True, init_omg=1.0, global_omg=True, learn_omg=True, ensemble=1,
)


def _base_model(out_classes, **overrides):
    cfg = {**BACKBONE, **overrides}
    return AKOrN(out_classes=out_classes, **cfg)


def _layers(model):
    # knet.py: each layer = ModuleList([transition, Identity, k_layer, readout, Identity]); core is index 2.
    for l in range(model.L):
        yield l, model.layers[l]


# ----------------------------- R6: full AKOrN (synchrony ON) -----------------------------
def build_R6(out_classes, **ov):
    return _base_model(out_classes, **ov)


# ------------- R5: recurrence + normalization + Omega ON, SYNCHRONY OFF (bracketed) -------------
def build_R5(out_classes, variant="depthwise", **ov):
    """
    'no_proj'   -> KLayer.apply_proj=False: drop the tangent-space projection (the explicit
                   Kuramoto phase-coupling). Surgical single-flag change vs R6.
    'depthwise' -> connectivity := 1x1 depthwise conv: removes neuron-neuron coupling entirely
                   -> pure per-oscillator normalized recurrent dynamics. (RECOMMENDED primary R5.)
    'frozen_J'  -> freeze connectivity at random init: no LEARNED relational coupling.
    Run all three; report R6 minus each (brackets "what counts as synchrony").
    """
    m = _base_model(out_classes, **ov)
    for _, layer in _layers(m):
        k = layer[2]  # KLayer
        if variant == "no_proj":
            k.apply_proj = False                         # klayer.py:141
        elif variant == "depthwise":
            ch = k.ch                                    # klayer.py:83
            layer[2].connectivity = nn.Conv2d(ch, ch, kernel_size=1, groups=ch, bias=True)
        elif variant == "frozen_J":
            for p in k.connectivity.parameters():
                p.requires_grad_(False)
        else:
            raise ValueError(f"unknown R5 variant {variant!r}")
    return m


# R6 internal negative control: synchrony machinery present but coupling scrambled+frozen.
def build_R6_scrambled(out_classes, **ov):
    m = _base_model(out_classes, **ov)
    for _, layer in _layers(m):
        for p in layer[2].connectivity.parameters():
            p.requires_grad_(False)  # random fixed J; apply_proj stays True
    return m


# ---- R7: NORMALIZATION ABLATION (E2). Coupling + projection + omega + recurrence ALL ON; ONLY the per-step
# spherical normalize() is swapped for a bounded-but-non-spherical variant, to test whether the SPHERICAL
# per-oscillator competition (not mere boundedness, not coupling) is the binding ingredient. We monkeypatch each
# KLayer.forward to route both normalize() calls through k._norm_fn -- byte-identical to klayer.py:152-165 otherwise
# (verified bit-exact when _norm_fn == spherical). Zero added params. RMS of the spherical state = 1/sqrt(n).
import types as _types
import torch.nn.functional as _F


def _make_norm_fn(kind, n, rmax=3.0, lam=1.0):
    if kind == "spherical":                                  # == R6 control (bit-exact)
        return lambda x: sphere_normalize(x, n)
    if kind == "clamp":                                      # V1: cap per-group radius at rmax, drop the sphere constraint
        def f(x):
            xr = reshape(x, n); r = xr.norm(dim=2, keepdim=True).clamp_min(1e-8)
            return reshape_back(xr * (r.clamp(max=rmax) / r))
        return f
    if kind == "layernorm":                                  # V2: normalize mean/var over channels, rescale to RMS 1/sqrt(n)
        def f(x):
            C = x.shape[1]
            z = _F.layer_norm(x.permute(0, 2, 3, 1), (C,)) * (1.0 / (n ** 0.5))
            return z.permute(0, 3, 1, 2).contiguous()
        return f
    if kind == "soft":                                       # V3 dose-response: x/(1+lam*(r-1)); lam=1 == sphere, lam=0 == none
        def f(x):
            xr = reshape(x, n); r = xr.norm(dim=2, keepdim=True).clamp_min(1e-8)
            return reshape_back(xr / (1.0 + lam * (r - 1.0)))
        return f
    raise ValueError(f"unknown R7 norm variant {kind!r}")


def _patched_klayer_forward(self, x, c, T, gamma):
    # IDENTICAL to KLayer.forward (klayer.py:152-165) except normalize() -> self._norm_fn
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


def build_R7(out_classes, variant="clamp", lam=1.0, rmax=3.0, no_proj=False, **ov):
    """E2 normalization-ablation arm: full R6 (coupling+proj+omega+T=3) but the per-step spherical normalize is
    replaced by `variant` in {spherical(=R6 control), clamp(V1), layernorm(V2), soft(V3,lam)}. Zero added params.
    no_proj=True ALSO sets apply_proj=False -> separates 'lost normalization-competition' from 'corrupted tangent
    projection' (the projection y-<y,x>x assumes ||x||=1, which the non-spherical variants break)."""
    m = _base_model(out_classes, **ov)
    for _, layer in _layers(m):
        k = layer[2]
        k._norm_fn = _make_norm_fn(variant, k.n, rmax=rmax, lam=lam)
        k.forward = _types.MethodType(_patched_klayer_forward, k)
        if no_proj:
            k.apply_proj = False
    return m


# ----------------------------- R1-R4: non-oscillator core -----------------------------
class LadderCore(nn.Module):
    """Drop-in for KLayer with the SAME forward signature: forward(x, c, T, gamma) -> (xs, es).
    Toggles ladder ingredients below recurrence/synchrony. Recurrence is added at R5 (T>1)."""

    def __init__(self, ch, n=1, ksize=9, sparsity=None, spherical=False, c_norm="gn"):
        super().__init__()
        self.ch, self.n, self.spherical, self.sparsity = ch, n, spherical, sparsity
        self.connectivity = nn.Conv2d(ch, ch, ksize, 1, ksize // 2)
        self.c_norm = nn.GroupNorm(ch // max(n, 1), ch, affine=True) if c_norm == "gn" else nn.Identity()
        self.act = nn.ReLU()

    def _kwta(self, x):
        # grouped k-WTA. TODO(Wk1-2): match the STRUCTURE of AKOrN's emergent sparsity
        # (synchronized assemblies fire together), not just the marginal fraction-active.
        if not self.sparsity:
            return x
        B, C = x.shape[:2]
        k = max(1, int(self.sparsity * C))
        flat = x.flatten(2)                              # B, C, HW
        thresh = flat.abs().topk(k, dim=1).values[:, -1:, :]
        mask = (flat.abs() >= thresh).float().view_as(x)
        return x * mask

    def forward(self, x, c, T, gamma):
        c = self.c_norm(c)
        if self.spherical:
            x = sphere_normalize(x, self.n)
        xs, es = [], [torch.zeros(x.shape[0], device=x.device)]
        steps = T if (T and T > 1) else 1                # R1-R4 set model.T=1 (recurrence is R5's ingredient)
        for _ in range(steps):
            y = self._kwta(self.act(self.connectivity(x) + c))
            x = x + gamma * y
            if self.spherical:
                x = sphere_normalize(x, self.n)
            xs.append(x)
            es.append(torch.zeros(x.shape[0], device=x.device))
        return xs, es


def build_R1_to_R4(rung, out_classes, akorn_sparsity=None, **ov):
    """akorn_sparsity = per-layer target fraction-active, MEASURED from a trained R6 fixed point
    (see budget.py / Wk-1). Pass it so R2-R4 match AKOrN's sparsity level."""
    assert rung in ("R1", "R2", "R3", "R4")
    spec = {
        "R1": dict(n=1, sparsity=None, spherical=False),
        "R2": dict(n=1, sparsity=akorn_sparsity, spherical=False),
        "R3": dict(n=BACKBONE["n"], sparsity=akorn_sparsity, spherical=False),
        "R4": dict(n=BACKBONE["n"], sparsity=akorn_sparsity, spherical=True),
    }[rung]
    m = _base_model(out_classes, **ov)
    ksz = {0: 9, 1: 7, 2: 5}
    for l, layer in _layers(m):
        layer[2] = LadderCore(ch=layer[2].ch if hasattr(layer[2], "ch") else BACKBONE["ch"] * (2 ** l),
                              ksize=ksz.get(l, 5), c_norm="gn", **spec)
    m.T = [1] * m.L  # no recurrence below R5
    return m


# ----------------------------- helpers -----------------------------
def build(rung, out_classes, **kw):
    if rung == "R6":
        return build_R6(out_classes, **kw)
    if rung == "R5":
        return build_R5(out_classes, **kw)
    if rung == "R7":
        return build_R7(out_classes, **kw)
    if rung == "R6s":
        return build_R6_scrambled(out_classes, **kw)
    return build_R1_to_R4(rung, out_classes, **kw)


def param_report(models: dict):
    """Enforce the ~2% param-match gate (red-team blocker #3). FLOPs: add fvcore/ptflops in Wk-1."""
    counts = {k: sum(p.numel() for p in m.parameters() if p.requires_grad) for k, m in models.items()}
    ref = max(counts.values())
    for k, v in counts.items():
        flag = "" if abs(v - ref) / ref <= 0.02 else "  <-- EXCEEDS 2% (compensate width/MLP or report as capacity-bounded)"
        print(f"{k:14s} params={v:>12,}  Δvs_max={100*(v-ref)/ref:+5.1f}%{flag}")
    return counts


if __name__ == "__main__":
    # Smoke check (CPU ok): build each rung at CIFAR-100 width and compare params.
    C = 100
    models = {
        "R1": build("R1", C), "R2": build("R2", C, akorn_sparsity=0.1),
        "R3": build("R3", C, akorn_sparsity=0.1), "R4": build("R4", C, akorn_sparsity=0.1),
        "R5_depthwise": build("R5", C, variant="depthwise"),
        "R5_no_proj": build("R5", C, variant="no_proj"),
        "R6": build("R6", C), "R6s": build("R6s", C),
    }
    param_report(models)
    x = torch.randn(2, 3, 32, 32)
    for name, m in models.items():
        m.eval()
        with torch.no_grad():
            print(name, "logits", tuple(m(x).shape))
