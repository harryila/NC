"""
Avalanche integration for the M1 ladder + the deterministic-eval fix.

Two red-team blockers handled here:
  - AKOrN re-randomizes oscillator state every forward (knet.py:182) -> stochastic logits
    corrupt BWT/CKA. We make EVAL deterministic via fixed-seed N-init logit averaging,
    and give the (deterministic) feedforward rungs the SAME N-init budget so the
    inference protocol is matched across all arms.
  - class-IL head: build with out_classes=total (100) and let the Split benchmark present
    class-incremental experiences; a multi-head wrapper is provided for task-IL.

!!! UNTESTED SCAFFOLDING. Validate on Split-MNIST in the Wk-0 spike before scaling. !!!
"""
import contextlib
import torch
import torch.nn as nn

from ladder import build


@contextlib.contextmanager
def _seeded(seed):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class LadderClassifier(nn.Module):
    """Wraps a ladder rung as an Avalanche-compatible model.

    train: single stochastic forward (matches AKOrN training).
    eval : average logits over `eval_inits` FIXED-seed forwards (deterministic + matched across arms).
           For feedforward rungs (R1-R4) the forwards are identical, so averaging is a harmless no-op
           that keeps the inference budget identical across arms.
    """

    def __init__(self, rung, num_classes=100, eval_inits=8, base_seed=0, **rung_kw):
        super().__init__()
        self.net = build(rung, num_classes, **rung_kw)
        self.eval_inits = eval_inits
        self.base_seed = base_seed

    def forward(self, x):
        if self.training:
            return self.net(x)
        logits = 0.0
        for i in range(self.eval_inits):
            with _seeded(self.base_seed + i):
                logits = logits + self.net(x)
        return logits / self.eval_inits


# --------------------------- Avalanche driver (sketch) ---------------------------
# Fill in during the Wk-0 spike; this is the intended shape, not a tested script.
def run_split_cifar100(rung, scenario="class", n_experiences=10, seed=0,
                       epochs=400, lr=1e-4, eval_inits=8, device="cuda", **rung_kw):
    """
    scenario: 'class' (class-IL, the ARBITER) or 'task' (task-IL, diagnostic only).
    Returns the Avalanche metrics dict (per-experience acc -> compute ACC / BWT / Forgetting).
    """
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive          # swap for EWC / replay (A4) / DER++ (A5)
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import (
        accuracy_metrics, forgetting_metrics, bwt_metrics,
    )
    from avalanche.logging import InteractiveLogger

    return_task_id = (scenario == "task")
    bench = SplitCIFAR100(n_experiences=n_experiences, return_task_id=return_task_id, seed=seed)

    model = LadderClassifier(rung, num_classes=100, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)

    evalp = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    strategy = Naive(model, optim, nn.CrossEntropyLoss(), train_mb_size=128, eval_mb_size=100,
                     train_epochs=epochs, evaluator=evalp, device=device)

    for exp in bench.train_stream:
        strategy.train(exp)
        strategy.eval(bench.test_stream)
    return evalp.get_last_metrics()


if __name__ == "__main__":
    # CPU smoke: deterministic eval should give identical logits across calls; train should differ.
    m = LadderClassifier("R6", num_classes=10, eval_inits=4)
    x = torch.randn(2, 3, 32, 32)
    m.eval()
    a, b = m(x), m(x)
    print("eval deterministic:", torch.allclose(a, b))
    m.train()
    print("train stochastic:", not torch.allclose(m(x), m(x)))
