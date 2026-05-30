"""
Wk-0 compute-budget probe (red-team: "no GPU-hour estimate exists").

Measures wall-clock/step and peak VRAM for R6 at the chosen config, then extrapolates
to the full ladder matrix so you commit to a REAL budget before the 5-7 week run.

Also dumps the per-layer fraction-active at the R6 fixed point -> `akorn_sparsity`,
the target the R2-R4 k-WTA controls must match (ladder.build_R1_to_R4).

Run on the GPU box:  PYTHONPATH=/path/to/akorn python budget.py
"""
import time
import torch

from ladder import build, param_report


def measure(rung="R6", num_classes=100, bs=128, steps=20, device="cuda", **kw):
    m = build(rung, num_classes, **kw).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (bs,), device=device)

    m.train()
    for _ in range(3):  # warmup
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
    print(f"  => ~{gpu_hours:,.0f} GPU-hours for the matrix "
          f"({arms} arms × {seeds} seeds × {scenarios} scen × {tasks} tasks × {epochs} ep)")
    print("  (add eval cost: eval_inits× forwards per experience; de-scope epochs/seeds if infeasible)")
    return gpu_hours


@torch.no_grad()
def fraction_active(rung="R6", num_classes=100, device="cuda", thresh=1e-3, **kw):
    """Per-layer mean fraction of |activation|>thresh at the core output -> match target for R2-R4."""
    m = build(rung, num_classes, **kw).to(device).eval()
    x = torch.randn(64, 3, 32, 32, device=device)
    _, _, xs, _ = m.feature(x)                 # xs[l] is the list of per-step states for layer l
    fr = [float((s[-1].abs() > thresh).float().mean()) for s in xs]
    print("  per-layer fraction-active (R6 fixed point):", [round(f, 3) for f in fr])
    return fr


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    param_report({"R6": build("R6", 100), "R5d": build("R5", 100, variant="depthwise")})
    sps, gb = measure("R6", device=dev)
    print(f"R6: {sps*1000:.1f} ms/step, peak {gb:.1f} GB")
    extrapolate(sps)
    if dev == "cuda":
        fraction_active("R6", device=dev)
