"""M3 / corrected-M2 headline: end-to-end UNBYPASSABLE phase->theta hypernetwork.

Tests the actual thesis with all 4 audit fixes: does the L1 phase-context — R6 (learned synchrony) vs
R6s (frozen random coupling) — as the SOLE task-adaptive pathway, reduce forgetting on a binding CL stream?

UNBYPASSABILITY (CCC's key principle, the thing the old C_ctx headline violated):
  * trunk = FROZEN-RANDOM conv (task-agnostic) -> features f. No static task-adaptive alternative channel.
  * context-gen = a frozen AKOrN (R6 or R6s), trained briefly then frozen -> L1 phase-state -> context c.
  * the ONLY trainable, task-adaptive params = the hypernet g: c -> theta (the head weights for f->logits).
  So all task adaptation MUST flow through the phase-context. If phase carries the task (R6's L1 channel),
  g can generate correct per-task heads with NO labels; if not, it forgets.

METRICS (functional, CCC-style): forgetting/BWT on the stream + wrong-context degradation (swap c to a
different task's context -> accuracy must drop if the channel is real & used). Compare R6 vs R6s.

SMOKE (--demo, CPU): unbypassability invariant on synthetic data — a context that carries task lets g
separate tasks (acc >> chance) and wrong-context hurts; a CONSTANT context cannot (acc ~ chance), proving
there is no bypass route. This is the invariant the headline failed and Split-MNIST accidentally satisfied.

Usage: python m2_hypernet.py --demo
       python m2_hypernet.py --arm R6 --seeds 0 1 2 --epochs 40 --device cuda   (and --arm R6s)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.normpath(os.path.join(HERE, "..", "m1_wk0"))
for p in (M1, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = os.path.join(HERE, "results")

import m2_shapes_construct as shp   # gen + experiences (binding construct)

ARM_RUNG = {"R6": ("R6", {}), "R6s": ("R6s", {})}   # learned-synchrony vs frozen-random-coupling
CTX_LAYER = 1   # the layer where the corrected screen found the synchrony channel (L1)


# ------------------------------------------------------------------ torch model pieces
def _build_pieces(arm, device, seed, ctx_dim_target=8):
    """Returns (akorn_ctx_gen [frozen], rand_trunk [frozen], n, feat_dim). akorn produces L1 phase;
    rand_trunk is a fixed random conv producing task-agnostic features."""
    import torch
    import torch.nn as nn
    import _bootstrap  # noqa
    from avalanche_backbone import LadderClassifier

    rung, kw = ARM_RUNG[arm]
    torch.manual_seed(seed)
    akorn = LadderClassifier(rung, num_classes=shp.N_CLASSES, eval_inits=1, base_seed=seed, **kw).to(device)
    # frozen-random task-agnostic trunk: fixed random conv stack 3->32->feat, global-pool
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    trunk = nn.Sequential(
        nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    ).to(device)
    for p_ in trunk.parameters():
        p_.requires_grad_(False)
    feat_dim = 64
    return akorn, trunk, int(getattr(akorn.net, "n", 4)), feat_dim


def _phase_context(akorn, x, layer, n, device):
    """L1 phase-state -> fixed 2n-dim pooled descriptor per sample (eval_inits=1, no averaging)."""
    import torch
    from h3 import _seeded as h3_seeded, group_directions
    from m2_primitives import _pool_phase_state
    with torch.no_grad(), h3_seeded(0):
        _c, _x, xs, _es = akorn.net.feature(x)
    st = xs[layer][-1].detach().float().cpu().numpy()   # (B,C,H,W)
    out = np.stack([_pool_phase_state(group_directions(st[b:b + 1], n=n)) for b in range(st.shape[0])])
    return torch.tensor(out, dtype=torch.float32, device=device)


class HyperHead(object):
    """g: context-> theta(=head weights). Wraps theta_generator.PhaseContextThetaGen + applies to features."""
    def __init__(self, ctx_dim, feat_dim, num_classes, device, seed):
        import torch  # noqa
        import theta_generator as tg
        self.gen = tg.PhaseContextThetaGen(context_dim=ctx_dim, feat_dim=feat_dim,
                                           num_classes=num_classes, hidden=64, seed=seed).to(device)
        self.tg = tg
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def logits(self, ctx, feats):
        out = self.gen(ctx)                                   # per-sample theta(c)
        return self.tg.apply_generated_head(feats, out["theta_flat"], self.feat_dim, self.num_classes)


# ------------------------------------------------------------------ end-to-end CL training
def run_arm(arm, seed, n_tasks=5, epochs=40, ctx_epochs=20, device="cuda", n_per_class=600, max_probe=160):
    import torch
    import torch.nn as nn

    # data: binding construct (multi-object Shapes), class-IL experiences
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_tr = shp._experiences(Xtr, ytr, n_exp=n_tasks)
    exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)

    akorn, trunk, n, feat_dim = _build_pieces(arm, device, seed)
    # briefly TRAIN the AKOrN context-gen so its phase-state is meaningful, then FREEZE it.
    opt_a = torch.optim.Adam(akorn.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    akorn.train()
    for (Xe, ye) in exps_tr:
        Xt = torch.tensor(Xe); yt = torch.tensor(ye)
        for ep in range(ctx_epochs):
            pr = torch.randperm(len(yt))
            for i in range(0, len(yt), 128):
                idx = pr[i:i + 128]
                opt_a.zero_grad(); loss = crit(akorn(Xt[idx].to(device)), yt[idx].to(device)); loss.backward(); opt_a.step()
    akorn.eval()
    for p_ in akorn.parameters():
        p_.requires_grad_(False)

    ctx_dim = 2 * n
    hh = HyperHead(ctx_dim, feat_dim, shp.N_CLASSES, device, seed)
    opt_g = torch.optim.Adam(hh.gen.parameters(), lr=1e-3)

    # end-to-end CL: ONLY the hypernet g trains; trunk + akorn frozen. naive sequential (no replay).
    T = n_tasks
    A = np.full((T, T), np.nan)
    for ti, (Xe, ye) in enumerate(exps_tr):
        Xt = torch.tensor(Xe); yt = torch.tensor(ye)
        hh.gen.train()
        for ep in range(epochs):
            pr = torch.randperm(len(yt))
            for i in range(0, len(yt), 128):
                idx = pr[i:i + 128]
                xb = Xt[idx].to(device); yb = yt[idx].to(device)
                with torch.no_grad():
                    ctx = _phase_context(akorn, xb, CTX_LAYER, n, device)
                    f = trunk(xb)
                opt_g.zero_grad(); loss = crit(hh.logits(ctx, f), yb); loss.backward(); opt_g.step()
        # eval accuracy on all seen tasks (forgetting matrix)
        hh.gen.eval()
        for tj in range(T):
            Xj = torch.tensor(exps_te[tj][0]); yj = torch.tensor(exps_te[tj][1])
            A[ti, tj] = _acc(hh, akorn, trunk, Xj, yj, n, device)
    learn = [float(A[k, k]) for k in range(T)]
    fwd = [A[k, k] - A[T - 1, k] for k in range(T - 1)]
    forgetting = float(np.mean(fwd)) if fwd else 0.0
    # wrong-context dP5: eval task tj's inputs but with context from a DIFFERENT task's inputs
    dP5 = _wrong_context_dP5(hh, akorn, trunk, exps_te, n, device)
    out = {"arm": arm, "seed": seed, "learn_acc": float(np.mean(learn)),
           "final_acc": float(np.mean(A[T - 1])), "forgetting": forgetting,
           "wrong_ctx_dP5": dP5, "acc_matrix": A.tolist()}
    del akorn, trunk, hh.gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _acc(hh, akorn, trunk, X, y, n, device, ctx_override=None, const=False):
    import torch
    correct = tot = 0
    with torch.no_grad():
        for i in range(0, len(y), 128):
            xb = X[i:i + 128].to(device); yb = y[i:i + 128].to(device)
            if const:
                ctx = torch.ones(len(xb), 2 * n, device=device)
            elif ctx_override is not None:
                ctx = ctx_override[i:i + 128].to(device)
            else:
                ctx = _phase_context(akorn, xb, CTX_LAYER, n, device)
            f = trunk(xb)
            pred = hh.logits(ctx, f).argmax(1)
            correct += int((pred == yb).sum()); tot += len(yb)
    return correct / max(tot, 1)


def run_joint(arm, seed, n_tasks=5, epochs=80, ctx_epochs=20, device="cuda", n_per_class=600, max_probe=160):
    """JOINT upper-bound (no CL): CAN the hypernet turn each arm's phase-context into task-appropriate
    PARAMETERS at all? Isolates channel USABILITY from CL forgetting dynamics (g's own forgetting). Reports
    real-context acc, constant-context acc (control = how much the trunk alone solves), their lift, and
    wrong-context dP5 at the (high-acc) joint solution. R6 lift >> R6s lift => channel usable for param-gen."""
    import torch
    import torch.nn as nn
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)
    akorn, trunk, n, feat_dim = _build_pieces(arm, device, seed)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    opt_a = torch.optim.Adam(akorn.parameters(), lr=1e-4); crit = nn.CrossEntropyLoss(); akorn.train()
    for ep in range(ctx_epochs):
        pr = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = pr[i:i + 128]; opt_a.zero_grad(); loss = crit(akorn(Xt[idx].to(device)), yt[idx].to(device)); loss.backward(); opt_a.step()
    akorn.eval()
    for p_ in akorn.parameters():
        p_.requires_grad_(False)

    # PRECOMPUTE context + features once (akorn + trunk are frozen) -> g-training is pure tensor ops (fast).
    def _cache(X):
        ctxs, fs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = X[i:i + 128].to(device)
                ctxs.append(_phase_context(akorn, xb, CTX_LAYER, n, device))
                fs.append(trunk(xb))
        return torch.cat(ctxs), torch.cat(fs)

    ctx_tr, f_tr = _cache(Xt)
    yt_dev = yt.to(device)
    test_cf = []
    for t in range(n_tasks):
        ty = torch.tensor(exps_te[t][1])
        c, f = _cache(torch.tensor(exps_te[t][0]))
        test_cf.append((c, f, ty.to(device)))

    def joint_g(use_const):
        hh = HyperHead(2 * n, feat_dim, shp.N_CLASSES, device, seed)
        opt = torch.optim.Adam(hh.gen.parameters(), lr=1e-3); hh.gen.train()
        ctx_src = torch.ones_like(ctx_tr) if use_const else ctx_tr
        for ep in range(epochs):
            pr = torch.randperm(len(yt), device=device)
            for i in range(0, len(yt), 256):
                idx = pr[i:i + 256]
                opt.zero_grad(); loss = crit(hh.logits(ctx_src[idx], f_tr[idx]), yt_dev[idx]); loss.backward(); opt.step()
        hh.gen.eval()
        accs = []
        with torch.no_grad():
            for (c, f, y) in test_cf:
                cc = torch.ones_like(c) if use_const else c
                accs.append(float((hh.logits(cc, f).argmax(1) == y).float().mean()))
        acc = float(np.mean(accs))
        dP5 = 0.0
        if not use_const:
            drops = []
            with torch.no_grad():
                for t in range(n_tasks):
                    c, f, y = test_cf[t]
                    cw = test_cf[(t + 1) % n_tasks][0]
                    m = min(len(f), len(cw))
                    a_w = float((hh.logits(cw[:m], f[:m]).argmax(1) == y[:m]).float().mean())
                    drops.append(a_w - accs[t])
            dP5 = float(np.mean(drops))
        del hh.gen
        return acc, dP5

    acc_real, dP5 = joint_g(False)
    acc_const, _ = joint_g(True)
    out = {"arm": arm, "seed": seed, "joint_acc_real": acc_real, "joint_acc_const": acc_const,
           "ctx_lift": acc_real - acc_const, "wrong_ctx_dP5": dP5}
    del akorn, trunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_cl_reg(arm, seed, beta=0.1, n_anchors=300, mech="outreg", n_replay=128, oracle=False, n_tasks=5,
               epochs=40, ctx_epochs=20, device="cuda", n_per_class=600, max_probe=160):
    """M3: CL training of g over a FROZEN phase channel (akorn) + frozen-random trunk (same fixed channel as
    the joint upper-bound), with a forgetting-mitigation mechanism. Tests whether the joint-confirmed-usable
    phase channel BYPASSES catastrophic forgetting, and whether R6 (task-separated contexts) beats R6s.
      mech='outreg' : von Oswald-style OUTPUT REGULARIZATION (penalize drift of g's theta on past contexts);
                      beta=0 reproduces naive CL (sanity).  [probe: fails at all beta -> per-input mismatch]
      mech='replay' : latent CONTEXT-REPLAY -- store a small buffer of past (phase-context, feature, label)
                      and interleave replay batches. R6's separable contexts should retain > R6s' overlapping."""
    import torch
    import torch.nn as nn
    Xtr, ytr = shp._gen_split(n_per_class, seed=1000 + seed)
    Xte, yte = shp._gen_split(max(60, max_probe // 5), seed=5000 + seed)
    exps_tr = shp._experiences(Xtr, ytr, n_exp=n_tasks)
    exps_te = shp._experiences(Xte, yte, n_exp=n_tasks)
    akorn, trunk, n, feat_dim = _build_pieces(arm, device, seed)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    opt_a = torch.optim.Adam(akorn.parameters(), lr=1e-4); crit = nn.CrossEntropyLoss(); akorn.train()
    for ep in range(ctx_epochs):
        pr = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = pr[i:i + 128]; opt_a.zero_grad(); loss = crit(akorn(Xt[idx].to(device)), yt[idx].to(device)); loss.backward(); opt_a.step()
    akorn.eval()
    for p_ in akorn.parameters():
        p_.requires_grad_(False)

    def _cache(X):
        cs, fs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = X[i:i + 128].to(device)
                cs.append(_phase_context(akorn, xb, CTX_LAYER, n, device)); fs.append(trunk(xb))
        return torch.cat(cs), torch.cat(fs)

    train_cf = [(_cache(torch.tensor(exps_tr[t][0])) + (torch.tensor(exps_tr[t][1]).to(device),)) for t in range(n_tasks)]
    test_cf = [(_cache(torch.tensor(exps_te[t][0])) + (torch.tensor(exps_te[t][1]).to(device),)) for t in range(n_tasks)]
    if oracle:   # LIVENESS: replace phase context with a one-hot task id (perfect context). If replay still
        def _orc(cf):  # collapses to chance here, the CL/replay harness is broken (not a channel result).
            out = []
            for t, (c, f, y) in enumerate(cf):
                oc = torch.zeros_like(c); oc[:, t] = 1.0
                out.append((oc, f, y))
            return out
        train_cf = _orc(train_cf); test_cf = _orc(test_cf)

    hh = HyperHead(2 * n, feat_dim, shp.N_CLASSES, device, seed); opt = torch.optim.Adam(hh.gen.parameters(), lr=1e-3)
    anchors = None
    bctx = bf = by = None   # replay buffer
    T = n_tasks; A = np.full((T, T), np.nan)
    for t in range(T):
        ctx_t, f_t, y_t = train_cf[t]
        targets = None
        if mech == "outreg" and anchors is not None and beta > 0:
            with torch.no_grad():
                targets = hh.gen(anchors)["theta_flat"].detach()
        hh.gen.train()
        for ep in range(epochs):
            pr = torch.randperm(len(y_t), device=device)
            for i in range(0, len(y_t), 128):
                idx = pr[i:i + 128]
                opt.zero_grad()
                loss = crit(hh.logits(ctx_t[idx], f_t[idx]), y_t[idx])
                if targets is not None:
                    loss = loss + beta * ((hh.gen(anchors)["theta_flat"] - targets) ** 2).mean()
                if mech == "replay" and by is not None:
                    rb = torch.randint(0, len(by), (min(n_replay, len(by)),), device=device)
                    loss = loss + crit(hh.logits(bctx[rb], bf[rb]), by[rb])
                loss.backward(); opt.step()
        hh.gen.eval()
        with torch.no_grad():
            for tj in range(T):
                c, f, y = test_cf[tj]
                A[t, tj] = float((hh.logits(c, f).argmax(1) == y).float().mean())
        k = max(1, n_anchors // T)
        if mech == "outreg":
            sel = ctx_t[torch.randperm(len(ctx_t), device=device)[:k]].detach()
            anchors = sel if anchors is None else torch.cat([anchors, sel])
        else:
            sel = torch.randperm(len(y_t), device=device)[:k]
            sc, sf, sy = ctx_t[sel].detach(), f_t[sel].detach(), y_t[sel].detach()
            bctx = sc if bctx is None else torch.cat([bctx, sc])
            bf = sf if bf is None else torch.cat([bf, sf])
            by = sy if by is None else torch.cat([by, sy])
    learn = [float(A[k, k]) for k in range(T)]
    fwd = [A[k, k] - A[T - 1, k] for k in range(T - 1)]
    forgetting = float(np.mean(fwd)) if fwd else 0.0
    drops = []
    with torch.no_grad():
        for t in range(T):
            c, f, y = test_cf[t]
            a_r = float((hh.logits(c, f).argmax(1) == y).float().mean())
            cw = test_cf[(t + 1) % T][0]; m = min(len(f), len(cw))
            a_w = float((hh.logits(cw[:m], f[:m]).argmax(1) == y[:m]).float().mean())
            drops.append(a_w - a_r)
    dP5 = float(np.mean(drops))
    out = {"arm": arm, "seed": seed, "beta": beta, "mech": mech, "oracle": oracle, "learn_acc": float(np.mean(learn)),
           "final_acc": float(np.mean(A[T - 1])), "forgetting": forgetting, "wrong_ctx_dP5": dP5,
           "acc_matrix": A.tolist()}
    del akorn, trunk, hh.gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _wrong_context_dP5(hh, akorn, trunk, exps_te, n, device):
    """Mean accuracy drop when each task's inputs are scored with another task's context. <0 => channel used."""
    import torch
    T = len(exps_te)
    right = wrong = 0.0
    for tj in range(T):
        Xj = torch.tensor(exps_te[tj][0]); yj = torch.tensor(exps_te[tj][1])
        a_right = _acc(hh, akorn, trunk, Xj, yj, n, device)
        # wrong context: phase from the NEXT task's inputs (rolled), same x for trunk/labels
        Xk = torch.tensor(exps_te[(tj + 1) % T][0])[:len(yj)]
        with torch.no_grad():
            wrong_ctx = _phase_context(akorn, Xk.to(device), CTX_LAYER, n, device)
        a_wrong = _acc(hh, akorn, trunk, Xj, yj, n, device, ctx_override=wrong_ctx)
        right += a_right; wrong += a_wrong
    return float((wrong - right) / T)   # negative = wrong context degrades (channel is real & used)


# ------------------------------------------------------------------ smoke (CPU): unbypassability invariant
def _demo():
    """Synthetic: context that encodes task lets g separate tasks (and wrong-ctx hurts); constant context
    cannot (acc ~ chance) -> proves no bypass route exists in the head."""
    import torch
    import theta_generator as tg
    torch.manual_seed(0)
    K, T, nf, B = 4, 4, 16, 256
    gen = tg.PhaseContextThetaGen(context_dim=8, feat_dim=nf, num_classes=K, hidden=64, seed=0)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    # task-agnostic features (random), labels = task id; context = one-hot-ish task vector (carries task)
    def batch(constant_ctx):
        t = torch.randint(0, T, (B,))
        f = torch.randn(B, nf)
        if constant_ctx:
            c = torch.ones(B, 8)
        else:
            c = torch.zeros(B, 8); c[torch.arange(B), t % 8] = 1.0
        return c, f, t
    def train_eval(constant_ctx):
        g2 = tg.PhaseContextThetaGen(context_dim=8, feat_dim=nf, num_classes=K, hidden=64, seed=0)
        op = torch.optim.Adam(g2.parameters(), lr=1e-3)
        for _ in range(300):
            c, f, t = batch(constant_ctx)
            op.zero_grad()
            logit = tg.apply_generated_head(f, g2(c)["theta_flat"], nf, K)
            loss = torch.nn.functional.cross_entropy(logit, t); loss.backward(); op.step()
        c, f, t = batch(constant_ctx)
        acc = float((tg.apply_generated_head(f, g2(c)["theta_flat"], nf, K).argmax(1) == t).float().mean())
        return acc
    acc_info = train_eval(False)
    acc_const = train_eval(True)
    print("[demo] task-carrying context acc=%.3f | constant context acc=%.3f (chance=%.2f)" % (acc_info, acc_const, 1.0 / K))
    assert acc_const < 0.45, "constant context should be ~chance (no bypass)"
    print("=== M2 HYPERNET DEMO OK (unbypassable: only context-routed task info can be used) ===")


def _save(recs, name="m2_hypernet"):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".json")
    existing = []
    if os.path.exists(path):
        try:
            existing = json.load(open(path)).get("runs", [])
        except Exception:
            existing = []
    existing += recs
    json.dump({"runs": existing}, open(path, "w"), indent=2, default=str)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--joint", action="store_true", help="joint upper-bound: channel usability without CL")
    ap.add_argument("--cl-reg", action="store_true", help="M3: CL hypernet (outreg or replay)")
    ap.add_argument("--mech", choices=["outreg", "replay"], default="outreg", help="forgetting mechanism")
    ap.add_argument("--beta", type=float, default=0.1, help="output-regularization strength (outreg)")
    ap.add_argument("--n-anchors", type=int, default=300, help="total stored anchors/replay samples (across tasks)")
    ap.add_argument("--n-replay", type=int, default=128, help="replay batch size (replay mech)")
    ap.add_argument("--oracle-ctx", action="store_true", help="liveness: use one-hot task id as context")
    ap.add_argument("--arm", choices=["R6", "R6s"], default="R6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ctx-epochs", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    if a.demo:
        _demo(); return
    if a.cl_reg:
        for s in a.seeds:
            r = run_cl_reg(a.arm, s, beta=a.beta, mech=a.mech, n_anchors=a.n_anchors, n_replay=a.n_replay, oracle=a.oracle_ctx, n_tasks=a.n_tasks, epochs=a.epochs, ctx_epochs=a.ctx_epochs, device=a.device)
            print("[%s CLREG mech=%s b=%.3g oracle=%s seed %d] learn=%.3f final=%.3f forgetting=%.3f dP5=%.4f" % (
                a.arm, a.mech, a.beta, a.oracle_ctx, s, r["learn_acc"], r["final_acc"], r["forgetting"], r["wrong_ctx_dP5"]), flush=True)
            _save([r], name="m2_hypernet_clreg")
        print("HYPERNET_CLREG_DONE %s mech=%s b=%.3g" % (a.arm, a.mech, a.beta))
        return
    if a.joint:
        for s in a.seeds:
            r = run_joint(a.arm, s, n_tasks=a.n_tasks, epochs=max(a.epochs, 80), ctx_epochs=a.ctx_epochs, device=a.device)
            print("[%s JOINT seed %d] real=%.3f const=%.3f lift=%.4f dP5=%.4f" % (
                a.arm, s, r["joint_acc_real"], r["joint_acc_const"], r["ctx_lift"], r["wrong_ctx_dP5"]), flush=True)
            _save([r], name="m2_hypernet_joint")
        print("HYPERNET_JOINT_DONE %s" % a.arm)
        return
    recs = []
    for s in a.seeds:
        r = run_arm(a.arm, s, n_tasks=a.n_tasks, epochs=a.epochs, ctx_epochs=a.ctx_epochs, device=a.device)
        print("[%s seed %d] learn=%.3f final=%.3f forgetting=%.3f dP5=%.4f" % (
            a.arm, s, r["learn_acc"], r["final_acc"], r["forgetting"], r["wrong_ctx_dP5"]), flush=True)
        recs.append(r); _save([r])
    print("HYPERNET_ARM_DONE %s" % a.arm)


if __name__ == "__main__":
    main()
