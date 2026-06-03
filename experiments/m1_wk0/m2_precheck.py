"""M2 VIABILITY PRE-CHECK: is the oscillatory phase-state a NON-EMPTY task channel?

M1 proved synchrony makes task representations OVERLAP LESS (reorganize). That is NECESSARY but NOT
SUFFICIENT for M2: M2's premise is that phase-state carries TASK INFORMATION (I(phase; task) > 0). If
the phases reorganize but carry no decodable task identity, the M2 "channel" is EMPTY -- the S_N /
"zero task bits" trap that sank DND / HSPC-T. This script answers, cheaply and BEFORE any M2 build:

    Can a FROZEN LINEAR PROBE read task-id off the trained R6 oscillator phase-state?

  cv_accuracy >> chance  -> the channel exists; M2 is genuinely viable -> proceed.
  cv_accuracy ~  chance  -> empty channel; synchrony reorganizes but does NOT encode task -> M2 needs
                            synchrony made an EXPLICIT task-symmetry-breaker (contrastive phase
                            objective) BEFORE measurement. A real finding, found cheaply.

MEMORY-FRUGAL BY CONSTRUCTION (the box's prior stalls were OOM-class: 34GB CKA, 3M-site expansion):
  * captures phase-state on a CAPPED probe set (default per_class small) and POOLS each sample to a
    fixed-length descriptor IMMEDIATELY (m2_primitives._pool_phase_state) -- never stacks full
    (B,C,H,W) tensors or millions of group_directions sites across samples;
  * runs the probe on CPU (numpy ridge / sklearn), tiny matrices (n_samples x ~2n descriptor);
  * single short training run (R6 on the existing CIFAR class-IL stream, few epochs) -- no big matmul.

Reuses the SAME ladder + capture path as the scored runs so the phase-state is identical in kind.

Usage (GPU box):
    python m2_precheck.py --epochs 30 --n-tasks 5 --per-class 8 --device cuda
    python m2_precheck.py --demo        # CPU: exercises the decodability call on synthetic phases
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _capture_phase_per_sample(model, loader, layer, device, eval_inits, base_seed, max_samples):
    """Capture per-SAMPLE oscillator phase-state at one layer, POOLED immediately to a small
    descriptor. Returns a list of pooled vectors (len <= max_samples). Memory-frugal: never holds
    more than one batch of full (B,C,H,W) phase tensors at a time; pools each sample on the spot."""
    import torch
    from h3 import _seeded as h3_seeded, group_directions
    from m2_primitives import _pool_phase_state
    was_training = model.training
    model.eval()
    pooled, n_seen = [], 0
    try:
        # average the oscillator state over eval_inits fixed-seed forwards (matches the scored path),
        # but accumulate the SMALL pooled descriptor, not the full tensor.
        with torch.no_grad():
            for batch in loader:
                if n_seen >= max_samples:
                    break
                x = batch[0].to(device)
                acc = None
                for j in range(eval_inits):
                    with h3_seeded(base_seed + j):
                        _c, _x, xs, _es = model.net.feature(x)
                    st = xs[layer][-1].detach().float().cpu()          # (B,C,H,W) for THIS batch only
                    acc = st if acc is None else acc + st
                acc = (acc / float(eval_inits)).numpy()                 # (B,C,H,W)
                n = int(getattr(model.net, "n", 4))
                for b in range(acc.shape[0]):
                    if n_seen >= max_samples:
                        break
                    # group_directions on ONE sample -> (sites, n); _pool_phase_state -> small fixed vec
                    dirs = group_directions(acc[b:b + 1], n=n)          # one sample's sites only
                    pooled.append(_pool_phase_state(dirs))
                    n_seen += 1
                del acc
        return pooled
    finally:
        if was_training:
            model.train()


def run_precheck(epochs=30, n_tasks=5, per_class=8, layer=0, device="cuda",
                 eval_inits=4, seed=0, max_samples_per_task=240):
    """Train R6 on the CIFAR class-IL stream, capture pooled phase-state per task on a small probe,
    and test linear task-decodability. Writes results/m2_precheck.json."""
    import numpy as np
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa: F401
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import InteractiveLogger
    from avalanche_backbone import LadderClassifier, _build_probe_loader
    import m2_primitives as m2

    bench = SplitCIFAR100(n_experiences=n_tasks, return_task_id=False, seed=seed)
    model = LadderClassifier("R6", num_classes=100, eval_inits=eval_inits, base_seed=seed).to(device)
    evalp = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[InteractiveLogger()])
    cl = Naive(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
               criterion=nn.CrossEntropyLoss(), train_mb_size=128, eval_mb_size=100,
               train_epochs=epochs, evaluator=evalp, device=device)

    # build ONE fixed probe loader per task (small, class-balanced) from each task's TEST set
    phase_by_task = {}
    for i, exp in enumerate(bench.train_stream):
        cl.train(exp)
    # after training the full stream, capture phase-state per task on each task's test experience
    for i, exp in enumerate(bench.test_stream):
        from torch.utils.data import DataLoader, Subset
        ds = exp.dataset
        idx = list(range(min(max_samples_per_task, len(ds))))
        loader = DataLoader(Subset(ds, idx), batch_size=100, shuffle=False)
        phase_by_task[i] = _capture_phase_per_sample(
            model, loader, layer=layer, device=device, eval_inits=eval_inits,
            base_seed=seed, max_samples=max_samples_per_task)
        print(f"[task {i}] captured {len(phase_by_task[i])} pooled phase descriptors")

    cv_acc, chance = m2.linear_task_decodability(phase_by_task, seed=seed)
    verdict = ("CHANNEL EXISTS (M2 viable)" if cv_acc > chance + 0.15
               else "EMPTY CHANNEL (zero task bits — M2 needs explicit symmetry-breaker)"
               if cv_acc < chance + 0.05 else "AMBIGUOUS (weak task signal — more probing)")
    out = {"layer": layer, "n_tasks": n_tasks, "epochs": epochs, "per_task_samples": max_samples_per_task,
           "cv_accuracy": float(cv_acc), "chance": float(chance),
           "margin_over_chance": float(cv_acc - chance), "verdict": verdict}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "m2_precheck.json"), "w"), indent=2, default=str)
    print("\n=== M2 PRE-CHECK ===")
    print(f"  linear task-decodability cv_acc = {cv_acc:.3f}  vs chance = {chance:.3f}  "
          f"(margin {cv_acc - chance:+.3f})")
    print(f"  VERDICT: {verdict}")
    print("  wrote results/m2_precheck.json")
    return out


def _demo():
    """CPU: exercise the decodability call on synthetic separable vs random phases (no GPU/torch)."""
    import numpy as np
    import m2_primitives as m2
    rng = np.random.default_rng(0)
    n = 4
    # separable: each task's phases cluster around a distinct mean -> should decode well above chance
    sep = {t: [np.clip(rng.normal((t + 1) / 6.0, 0.05, size=(50, n)), -1, 1) for _ in range(40)]
           for t in range(5)}
    cv, ch = m2.linear_task_decodability(sep, seed=0)
    print(f"[separable] cv={cv:.3f} chance={ch:.3f} (expect cv >> chance)")
    # random: identical distribution across tasks -> should be ~chance
    rnd = {t: [rng.normal(0.0, 1.0, size=(50, n)) for _ in range(40)] for t in range(5)}
    cv2, ch2 = m2.linear_task_decodability(rnd, seed=0)
    print(f"[random]   cv={cv2:.3f} chance={ch2:.3f} (expect cv ~ chance)")
    assert cv > ch + 0.15, "separable phases must be decodable"
    assert cv2 < ch2 + 0.15, "random phases must be ~chance"
    print("=== M2 PRE-CHECK DEMO OK (decodability logic validated) ===")


def main():
    ap = argparse.ArgumentParser(description="M2 viability pre-check: task-decodability from phase-state")
    ap.add_argument("--demo", action="store_true", help="CPU: decodability logic on synthetic phases")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=8, help="(reserved) probe class balance knob")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval-inits", type=int, default=4)
    ap.add_argument("--max-samples-per-task", type=int, default=240, help="cap probe size (OOM guard)")
    args = ap.parse_args()
    if args.demo:
        _demo(); return
    run_precheck(epochs=args.epochs, n_tasks=args.n_tasks, layer=args.layer, device=args.device,
                 eval_inits=args.eval_inits, max_samples_per_task=args.max_samples_per_task)


if __name__ == "__main__":
    main()
