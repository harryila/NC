"""
GATE A -- TRAINED DESYNCHRONIZATION arm (the NON-CIRCULAR synchrony-necessity test).

The frozen desync probe was circular: it perturbed the state x that the readout then reads, so an FG-ARI drop only
says "I corrupted the readout's input", not "synchrony matters". The fix (per the synthesis adversary): TRAIN AKOrN
from scratch with per-step phase noise injected into the oscillator recurrence, so the MODEL ADAPTS to a low-coherence
regime. Then:
  - eval-with-matched-noise: if FG-ARI recovers toward ~75 while R_global stays LOW -> the model BINDS WITHOUT global
    phase coherence -> synchronization is NOT required at training time (clean causal evidence, not a frozen artifact).
  - if FG-ARI fails to recover -> synchronization IS needed; we honestly retreat to the dispensability story.
The noise is per-step, per-token Gaussian on the state direction, then renormalize -- identical mechanism to
native_desync_probe.make_noisy_forward, but active during TRAINING (and matched at eval) so it is a learned regime,
not a frozen corruption. Parameter-identical to full AKOrN.

Wired via --phase_noise SIGMA in train_obj.py / eval_obj.py (0 = off). Use the same SIGMA at train and eval.
"""
import os
import sys
import types

import torch

_SRC = os.environ.get("AKORN_SRC", "/root/NC/external/akorn")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from source.layers.klayer import KLayer  # noqa: E402
from source.layers.kutils import normalize as sphere_normalize  # noqa: E402


def _make_noisy_forward(sigma):
    def fwd(self, x, c, T, gamma):
        # BYTE-IDENTICAL to klayer.py:152-165 except per-step phase noise injected into the state.
        xs, es = [], []
        c = self.c_norm(c)
        x = sphere_normalize(x, self.n)
        es.append(torch.zeros(x.shape[0]).to(x.device))
        for t in range(T):
            dxdt, _sim = self.kupdate(x, c)
            x = sphere_normalize(x + gamma * dxdt, self.n)
            if sigma > 0:
                x = sphere_normalize(x + sigma * torch.randn_like(x), self.n)  # DESYNC during training
            xs.append(x)
            es.append((-_sim).reshape(x.shape[0], -1).sum(-1))
        return xs, es
    return fwd


def phase_noise_akorn(net, sigma, verbose=True):
    """Inject per-step phase noise into every KLayer recurrence (train + eval). Parameter-identical."""
    n = 0
    for m in list(net.modules()):
        if isinstance(m, KLayer):
            m.forward = types.MethodType(_make_noisy_forward(float(sigma)), m)
            n += 1
    if verbose:
        print(f"  [phase_noise] patched {n} KLayer(s) with sigma={sigma}")
    if n == 0:
        raise RuntimeError("phase_noise_akorn: no KLayer found.")
    return net


if __name__ == "__main__":
    from source.models.objs.knet import AKOrN
    dev = "cpu" if not torch.cuda.is_available() else "cuda"
    net = AKOrN(n=4, ch=256, L=1, T=8, psize=8, gta=True, J="attn", ksize=1, c_norm="gn", gamma=1.0,
                imsize=128, use_omega=False, init_omg=0.01, global_omg=False, maxpool=True, project=True,
                heads=8, use_ro_x=False, learn_omg=False, no_ro=False, autorescale=False).to(dev)
    n0 = sum(p.numel() for p in net.parameters() if p.requires_grad)
    phase_noise_akorn(net, 0.5)
    n1 = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"params full={n0:,} noisy={n1:,} delta={n1-n0} (param-free, expect 0)")
    assert n0 == n1
    with torch.no_grad():
        y = net(torch.rand(2, 3, 128, 128, device=dev))
    print("forward OK:", tuple(y.shape), "-> trained-desync arm ready")
