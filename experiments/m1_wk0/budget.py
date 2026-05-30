"""
Wk-0 compute-budget probe + the (corrected) sparsity-target measurement.

GPU-hour extrapolation is unchanged. The sparsity probe is FIXED: v1 thresholded the
oscillator state |x|>eps, but x is L2-normalized onto the unit sphere every step, so
~everything is "active" (~0.999) -- a meaningless k-WTA target. We instead use the
PARTICIPATION RATIO of the readout features (the activations that actually feed forward
and the head): PR = (Σλ)^2 / Σλ^2 over the feature-covariance eigenvalues ≈ effective #
active dims. Reported as PR/num_features in (0,1] -> the target R2-R4 k-WTA should match.
Persists results/sparsity_target.json (the one cross-job dependency; see RUNNING.md DAG).

Run on the GPU box:  python budget.py
"""
import json
import os
import time
import torch

from ladder import build, param_report

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def measure(rung="R6", num_classes=100, bs=128, steps=20, device="cuda", **kw):
    m = build(rung, num_classes, **kw).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (bs,), device=device)
    m.train()
    for _ in range(3):
        opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    s_per_step = (time.time() - t0) / steps
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")
    return s_per_step, peak_gb


def extrapolate(s_per_step, *, arms=8, seeds=10, scenarios=1, tasks=10, epochs=400,
                imgs_per_task=5000, bs=128):
    steps_per_epoch = imgs_per_task / bs
    total_steps = arms * seeds * scenarios * tasks * epochs * steps_per_epoch
    gpu_hours = total_steps * s_per_step / 3600
    print(f"  steps/epoch≈{steps_per_epoch:.0f}  total_train_steps≈{total_steps:,.0f}")
    print(f"  => ~{gpu_hours:,.0f} GPU-hours ({arms} arms × {seeds} seeds × {scenarios} scen "
          f"× {tasks} tasks × {epochs} ep). CALIBRATE epochs (calibrate_epochs.py) — 400 is overkill.")
    return gpu_hours


@torch.no_grad()
def effective_sparsity(rung="R6", num_classes=100, device="cuda", **kw):
    """Per-layer effective fraction-active at the R6 fixed point via the PARTICIPATION RATIO
    of the readout features (post-block), NOT |oscillator state|>eps. Returns PR/dim in (0,1]
    per layer -> the k-WTA target for R2-R4."""
    m = build(rung, num_classes, **kw).to(device).eval()
    feats = []
    hooks = [m.layers[l][3].register_forward_hook(lambda mod, i, o: feats.append(o.detach()))
             for l in range(m.L)]                      # knet.py: index 3 = readout block
    m(torch.randn(256, 3, 32, 32, device=device))
    for h in hooks:
        h.remove()
    fracs = []
    for f in feats:
        z = f.flatten(2).mean(-1) if f.ndim == 4 else f       # B, C
        z = z - z.mean(0, keepdim=True)
        cov = (z.T @ z) / max(1, z.shape[0] - 1)
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        pr = (ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12)
        fracs.append(float(pr / z.shape[1]))
    print("  per-layer effective sparsity (PR/dim):", [round(x, 3) for x in fracs])
    return fracs


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    param_report({"R6": build("R6", 100),
                  "R5_no_proj": build("R5", 100, variant="no_proj"),
                  "R5_depthwise": build("R5", 100, variant="depthwise")})
    sps, gb = measure("R6", device=dev)
    print(f"R6: {sps*1000:.1f} ms/step, peak {gb:.1f} GB")
    extrapolate(sps)
    if dev == "cuda":
        fr = effective_sparsity("R6", device=dev)
        # Persist the ONE cross-job dependency (R2-R4 k-WTA target). See RUNNING.md DAG.
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "sparsity_target.json"), "w") as f:
            json.dump(fr, f)
        print(f"  wrote {os.path.join(RESULTS, 'sparsity_target.json')}: {[round(x,3) for x in fr]}")
