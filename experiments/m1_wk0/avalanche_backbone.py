"""
Avalanche integration for the M1 ladder: deterministic-eval wrapper, CL-strategy dispatch
(A4/A5 on the R6 backbone), and a trajectory-recording driver that also snapshots H3 features.

Verified vs avalanche-lib 0.6.0: strategy ctors are KEYWORD-ONLY; DER(beta>0)==DER++; FOSTER absent.
Per-experience accuracy key: 'Top1_Acc_Exp/eval_phase/test_stream/Task{NNN}/Exp{MMM}' (class-IL=>Task000).

!!! Validate on Split-MNIST in the Wk-0 spike before scaling. !!!
"""
import contextlib
import numpy as np
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
    """train: single stochastic forward. eval: mean logits over `eval_inits` FIXED-seed forwards
    (deterministic + matched across arms; a no-op for the deterministic feedforward rungs)."""

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


# ----------------------- fixed class-balanced probe loader (built ONCE) -----------------------
def _build_probe_loader(bench, per_class=2, batch_size=100, probe_seed=12345):
    """ONE fixed, class-balanced probe DataLoader from the test stream, with a DEDICATED probe_seed
    (independent of the run seed) and shuffle=False -> byte-identical probe inputs across arms/seeds
    (an h3 CKA validity requirement). NOTE: probe is a read-only subset of the scored test set; for
    strict independence reserve a held-out slice (TODO) — the overlap only weakens the descriptive
    overlap<->forgetting coupling, not the apply_proj-flip / phase claims."""
    from torch.utils.data import ConcatDataset, Subset, DataLoader
    test_sets = [exp.dataset for exp in bench.test_stream]
    full = ConcatDataset(test_sets)
    labels = []
    for ds in test_sets:
        t = getattr(ds, "targets", None)
        labels.extend(list(t) if t is not None else [int(ds[i][1]) for i in range(len(ds))])
    labels = np.asarray(labels)
    rng = np.random.default_rng(probe_seed)
    idx = []
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        take = min(per_class, len(pool))
        idx.extend(sorted(rng.choice(pool, size=take, replace=False).tolist()))
    return DataLoader(Subset(full, sorted(idx)), batch_size=batch_size, shuffle=False)


def _build_aug_probe_loader(probe_loader, batch_size=100, aug_seed=12345):
    """A fixed AUGMENTED view of the SAME probe inputs, materialized ONCE into an in-memory
    TensorDataset so every arm/seed/snapshot sees BYTE-IDENTICAL augmented inputs (the same
    CKA-validity requirement as the clean probe). The augment is a fixed-seed light random
    crop (pad 4, reflect) + horizontal flip -- the standard CIFAR train-time augment -- applied
    once here, NOT re-sampled per forward, so O_intra (clean-vs-augmented self-CKA) is matched
    across arms and is purely a property of each snapshot's representation, not of augment noise.

    Returns a shuffle=False DataLoader yielding (x_aug, y, task_id) in the SAME order as
    probe_loader, so extract_features over it produces a per-task augmented feature dict aligned
    row-for-row with the clean features."""
    from torch.utils.data import TensorDataset, DataLoader
    g = torch.Generator().manual_seed(aug_seed)
    xs, ys, tids = [], [], []
    for batch in probe_loader:                       # shuffle=False -> deterministic order
        x, y = batch[0], batch[1]
        tid = batch[2] if len(batch) > 2 else torch.zeros(len(y), dtype=torch.long)
        B, _C, H, W = x.shape
        xp = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        for b in range(B):
            top = int(torch.randint(0, 2 * 4 + 1, (1,), generator=g).item())
            left = int(torch.randint(0, 2 * 4 + 1, (1,), generator=g).item())
            crop = xp[b:b + 1, :, top:top + H, left:left + W]
            if torch.rand((1,), generator=g).item() < 0.5:
                crop = torch.flip(crop, dims=[3])    # horizontal flip
            xs.append(crop)
        ys.append(y)
        tids.append(tid if torch.is_tensor(tid) else torch.as_tensor(tid))
    x_aug = torch.cat(xs, 0)
    y_aug = torch.cat(ys, 0)
    t_aug = torch.cat([t.reshape(-1) for t in tids], 0)
    return DataLoader(TensorDataset(x_aug, y_aug, t_aug), batch_size=batch_size, shuffle=False)


def _capture_osc(model, probe_loader, layers, device, eval_inits, base_seed):
    """Capture xs[l][-1] (oscillator state [B,C,H,W]) per layer on the probe set, averaged over
    eval_inits fixed-seed forwards (synchrony arms only)."""
    from h3 import _seeded as h3_seeded
    was_training = model.training
    model.eval()
    accum = {l: None for l in layers}
    try:
        for j in range(eval_inits):
            per = {l: [] for l in layers}
            with h3_seeded(base_seed + j), torch.no_grad():
                for batch in probe_loader:
                    _c, _x, xs, _es = model.net.feature(batch[0].to(device))
                    for l in layers:
                        per[l].append(xs[l][-1].detach().float().cpu())
            for l in layers:
                cat = torch.cat(per[l], 0)
                accum[l] = cat if accum[l] is None else accum[l] + cat
        return {l: (accum[l] / float(eval_inits)).numpy() for l in layers}
    finally:
        if was_training:
            model.train()


# ----------------------------------- the driver -----------------------------------
def run_split_cifar100(rung, scenario="class", n_experiences=10, seed=0,
                       epochs=400, lr=1e-4, eval_inits=8, device="cuda",
                       strategy="naive", mem_size=2000, ewc_lambda=1.0, ewc_mode="separate",
                       der_alpha=0.1, der_beta=0.5, batch_size_mem=128,
                       snapshot_h3=False, h3_layers=(0, 1, 2), h3_n=None, h3_k=8,
                       probe_per_class=2, probe_seed=12345, h3_augment_intra=True, **rung_kw):
    """Split-CIFAR100 CL on a ladder rung with a selectable strategy. Records the per-experience
    accuracy MATRIX (-> learning_acc / forgetting / bwt) and, if snapshot_h3, H3 features per task.

    For A4/A5 baselines use rung="R6" + strategy in {ewc,replay,derpp}. derpp == built-in DER (beta>0)
    == DER++. DARK-KNOWLEDGE contract: DER stores replay logits under model.eval() (the eval-averaged
    target); the train replay forward is stochastic -> adds gradient variance on oscillatory arms.
    Acceptable for the reference anchor; set eval_inits=1 to match noise models if needed.

    Returns {final_metrics, acc_matrix, learning_acc, avg_forgetting(pts), bwt(pts), selfcheck_ok, h3}.
    """
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.training.supervised import Naive, EWC, Replay, DER
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
    from avalanche.logging import InteractiveLogger
    import h3

    is_task = (scenario == "task")
    bench = SplitCIFAR100(n_experiences=n_experiences, return_task_id=is_task, seed=seed)
    model = LadderClassifier(rung, num_classes=100, eval_inits=eval_inits, base_seed=seed, **rung_kw).to(device)

    evalp = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
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
        raise ValueError("FOSTER is not in avalanche-lib 0.6.0; use strategy='derpp' (DER++) for A5.")
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected naive|ewc|replay|derpp")

    T = n_experiences

    def _acc_key(j):
        return f"Top1_Acc_Exp/eval_phase/test_stream/Task{(j if is_task else 0):03d}/Exp{j:03d}"

    is_synchrony_arm = str(rung).upper().startswith("R6")
    h3_layers = list(h3_layers)
    h3_n_eff = h3_n if h3_n is not None else int(getattr(model.net, "n", 4))   # derive n from the model
    probe_loader, aug_probe_loader = None, None
    feats_by_task, aug_feats_by_task, osc_by_task = {}, {}, {}
    if snapshot_h3:
        probe_loader = _build_probe_loader(bench, per_class=probe_per_class,
                                           batch_size=common["eval_mb_size"], probe_seed=probe_seed)
        if h3_augment_intra:
            # Augmented view of the SAME probe (fixed seed) -> real O_intra (honest DiD).
            aug_probe_loader = _build_aug_probe_loader(probe_loader,
                                                       batch_size=common["eval_mb_size"], aug_seed=probe_seed)

    A = np.full((T, T), np.nan, dtype=float)
    for i, exp in enumerate(bench.train_stream):
        cl.train(exp)
        if snapshot_h3:
            feats_by_task[i] = h3.extract_features(model, probe_loader, layers=h3_layers, device=device,
                                                   eval_inits=model.eval_inits, base_seed=model.base_seed)
            if aug_probe_loader is not None:
                aug_feats_by_task[i] = h3.extract_features(model, aug_probe_loader, layers=h3_layers,
                                                           device=device, eval_inits=model.eval_inits,
                                                           base_seed=model.base_seed)
            if is_synchrony_arm:
                osc_by_task[i] = _capture_osc(model, probe_loader, layers=h3_layers, device=device,
                                              eval_inits=model.eval_inits, base_seed=model.base_seed)
        cl.eval(bench.test_stream)
        last = evalp.get_last_metrics()
        for j in range(T):
            A[i, j] = float(last[_acc_key(j)])

    final_metrics = evalp.get_last_metrics()
    learning_acc = [float(A[k, k]) for k in range(T)]
    fwd = [A[j, j] - A[T - 1, j] for j in range(T - 1)]
    avg_forgetting = 100.0 * float(np.mean(fwd)) if fwd else 0.0
    bwt = -avg_forgetting

    # Self-check (equal-size 10x10 => matrix-derived == Avalanche stream metric). WARN, never abort a
    # completed multi-GPU-hour run (review fix: a tripped assert must not discard saved output).
    selfcheck_ok = True
    sf = final_metrics.get("StreamForgetting/eval_phase/test_stream")
    if isinstance(sf, (int, float)) and abs(avg_forgetting - 100.0 * float(sf)) >= 1e-3:
        selfcheck_ok = False
        print(f"WARNING selfcheck: matrix forgetting {avg_forgetting:.4f} != "
              f"StreamForgetting*100 {100.0 * float(sf):.4f} (check the accuracy-key task-id).")

    h3_out = {"overlap_summary": None, "phase_stability": "N/A"}
    if snapshot_h3 and len(feats_by_task) >= 2:
        # Honest DiD: pass the augmented-probe features so O_intra is the real within-snapshot
        # augmentation self-CKA. Only feed it when we captured the same task indices clean+augmented.
        aug = aug_feats_by_task if (aug_feats_by_task and set(aug_feats_by_task) == set(feats_by_task)) else None
        h3_out["overlap_summary"] = h3.overlap_summaries(feats_by_task, layers=h3_layers,
                                                         aug_features_by_task=aug)
        if is_synchrony_arm and len(osc_by_task) >= 2:
            h3_out["phase_stability"] = {l: h3.phase_cluster_stability(osc_by_task, layer=l, n=h3_n_eff, k=h3_k)
                                         for l in h3_layers}
        else:
            h3_out["phase_stability"] = h3.PHASE_NA["phase_cluster_stability"]

    return {"final_metrics": final_metrics, "acc_matrix": A.tolist(), "learning_acc": learning_acc,
            "avg_forgetting": avg_forgetting, "bwt": bwt, "selfcheck_ok": selfcheck_ok, "h3": h3_out}


if __name__ == "__main__":
    m = LadderClassifier("R6", num_classes=10, eval_inits=4)
    x = torch.randn(2, 3, 32, 32)
    m.eval()
    print("eval deterministic:", torch.allclose(m(x), m(x)))
    m.train()
    print("train stochastic:", not torch.allclose(m(x), m(x)))
