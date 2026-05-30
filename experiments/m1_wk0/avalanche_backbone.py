"""
Avalanche integration for the M1 ladder + the deterministic-eval fix, with a CL-STRATEGY
DISPATCH so the A4/A5 reference baselines run on the SAME R6 backbone (strategy-level contrast).

Verified against avalanche-lib 0.6.0 (the env per requirements.txt / setup_run.log). Key facts:
  * Strategy constructors are KEYWORD-ONLY after `self` -> we pass everything by keyword
    (the old positional `Naive(model, optim, criterion, ...)` raises TypeError on 0.4.0+).
  * Naive / EWC / Replay / DER are in avalanche.training.supervised.
      DER(*, mem_size, batch_size_mem=None, alpha=0.1, beta=0.5, ...) -> beta>0 IS DER++ (no plugin needed).
      EWC(*, ewc_lambda, mode='separate', ...)  (ewc_lambda required); Replay(*, mem_size=200, ...).
  * FOSTER does NOT exist in avalanche-lib -> we do NOT accept 'foster' (raise with a clear message)
    rather than silently aliasing it to DER++ in a results table.

!!! UNTESTED SCAFFOLDING beyond import/smoke. Validate on Split-MNIST in the Wk-0 spike. !!!
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
    eval : average logits over `eval_inits` FIXED-seed forwards (deterministic + matched across
           arms). Feedforward rungs are deterministic, so the average is a harmless no-op that
           keeps the inference budget identical across arms.
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


# ------------------------------- Avalanche driver -------------------------------
def run_split_cifar100(rung, scenario="class", n_experiences=10, seed=0,
                       epochs=400, lr=1e-4, eval_inits=8, device="cuda",
                       strategy="naive", mem_size=2000, ewc_lambda=1.0, ewc_mode="separate",
                       der_alpha=0.1, der_beta=0.5, batch_size_mem=128, **rung_kw):
    """
    Split-CIFAR100 continual learning on a ladder rung with a selectable CL strategy.

    rung      : ladder rung (R1..R6/R6s). For the A4/A5 reference baselines use rung="R6" so every
                arm shares the SAME backbone and differs only in the CL STRATEGY.
    scenario  : 'class' (class-IL, the ARBITER) or 'task' (task-IL, diagnostic only).
    strategy  : 'naive'  -> plain finetuning (lower-bound).
                'ewc'    -> EWC regularization (A4 reference).
                'replay' -> experience replay, reservoir buffer (A4 reference).
                'derpp'  -> built-in DER with beta>0 == DER++ (A5 strong anchor).
    Returns the Avalanche metrics dict; primary endpoint key 'StreamForgetting/eval_phase/test_stream'.

    DARK-KNOWLEDGE CONTRACT (derpp): the built-in DER captures replay logits under model.eval(),
    i.e. the eval_inits-AVERAGED deterministic LadderClassifier logit (a stable target), while the
    train-time replay forward is the single stochastic forward. On the oscillatory arms this adds
    gradient variance to the MSE term. For the A5 anchor this is acceptable and documented; if it
    matters, run derpp with eval_inits=1 so target and prediction share one noise model. (Flagged
    for Wk-2; the anchor is reference-only, not the gate.)
    """
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive, EWC, Replay, DER
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
    from avalanche.logging import InteractiveLogger

    bench = SplitCIFAR100(n_experiences=n_experiences, return_task_id=(scenario == "task"), seed=seed)
    model = LadderClassifier(rung, num_classes=100, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)

    evalp = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    # ALL strategy constructors are keyword-only; criterion is required for EWC/Replay.
    common = dict(model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
                  train_mb_size=128, eval_mb_size=100, train_epochs=epochs,
                  evaluator=evalp, device=device)

    s = strategy.lower()
    if s == "naive":
        cl = Naive(**common)
    elif s == "ewc":
        cl = EWC(ewc_lambda=ewc_lambda, mode=ewc_mode, **common)
    elif s == "replay":
        cl = Replay(mem_size=mem_size, **common)
    elif s == "derpp":
        cl = DER(mem_size=mem_size, batch_size_mem=batch_size_mem, alpha=der_alpha, beta=der_beta, **common)
    elif s == "foster":
        raise ValueError("FOSTER is not shipped by avalanche-lib (confirmed absent in 0.6.0). "
                         "Use strategy='derpp' (DER++) as the A5 anchor, or vendor PyCIL's FOSTER.")
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected naive|ewc|replay|derpp")

    for exp in bench.train_stream:
        cl.train(exp)
        cl.eval(bench.test_stream)
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
