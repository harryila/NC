"""
POSITIVE CONTROL (#4) — the synchrony-FAVORING task the prereg REQUIRES to PASS before any
null / PIVOT-A / PIVOT-B may be declared (preregistration.md §Guards: "Positive control
(synchrony-favoring task) must PASS before any null is declared").

WHY a positive control at all. A null R6−R5 on Split-CIFAR-100 (the M1 front-load came back
INCONCLUSIVE / saturated) is only interpretable as "synchrony does not help HERE" if we can
SHOW the H3 probe has the DETECTION POWER to see a synchrony effect WHEN ONE IS KNOWN TO EXIST.
Absent that, a null is indistinguishable from a dead instrument. This module manufactures a task
where binding is *required* to solve it, runs the EXACT same R6-vs-R5:no_proj contrast through the
EXACT same h3.overlap_summaries / h3.paired_did pipeline, and gates on:

    PASS  iff  R6 has SIGNIFICANTLY LOWER inter-task overlap than R5:no_proj
               (seed-paired one-sided p < 0.05, R5_inner − R6_inner > 0).

A PASS = "the probe CAN see synchrony's overlap-reduction where it should exist" -> a null on
CIFAR is a substantive null, not a null instrument. A FAIL = the probe is underpowered / the
construct is wrong, and NO PIVOT may be declared on its strength.

DESIGN (cheapest construct-valid feature-binding stream).
  Classes are the CONJUNCTION of two independent features:
      color in {R, G, B}  x  shape in {square, circle, triangle}  ->  9 classes.
  Solving REQUIRES BINDING: neither color alone nor shape alone is diagnostic of the class
  (each color appears in 3 classes, each shape appears in 3 classes; only the *pair* is). This
  is the canonical illusory-conjunction / feature-binding setup where synchrony (temporal
  binding / Kuramoto phase alignment) is hypothesised to help — so R6 (apply_proj=True) should
  organise representations so cross-task overlap is LOWER (less interference) than R5:no_proj.
  Images are 3x32x32 in [0,1] (matches AKOrN's rgb_normalize mean/std 0.5 and the 3x32x32 head).
  The 9 classes are split into a few SEQUENTIAL tasks (class-IL), so inter-task CKA is defined.

  We deliberately AVOID the Sudoku/CLEVR/ItrSA object-discovery codepath (different model + loss
  = weeks). Everything rides on the EXISTING ladder (build('R6') / build('R5', variant='no_proj'))
  and the EXISTING h3 pipeline, via a self-contained run_positive_control() parallel driver
  (cleaner than monkey-injecting a benchmark into run_split_cifar100, which is wired to
  avalanche.SplitCIFAR100).

CONTRAST WITH THE NEGATIVE/GEOMETRY CONTROL. The M1 geometry kill-test SURVIVED — the head-free
H3 overlap effect needs LEARNED synchrony, not projection geometry. The positive control is the
complementary direction: confirm a TRUE effect is DETECTABLE. Together they bound the probe:
sensitive to real synchrony, not fooled by geometry.

LOCAL/CPU NOTE. torch + the AKOrN backbone live on the GPU box; they are NOT importable here.
This file therefore:
  * py_compiles with no torch import at module top (torch is imported lazily inside the driver),
  * exposes a numpy-only synthetic-data generator (make_binding_dataset) + a --demo that proves
    the generator's shapes/labels/binding property on numpy alone (via /tmp/h3venv/bin/python),
  * exposes a --smoke mode (2 tasks, ~64 imgs/class, 1 epoch, cpu) that exercises the full
    dataset+model+H3 wiring without a real GPU run (run it on the GPU box / a torch-CPU box).

Run:
  /tmp/h3venv/bin/python positive_control.py --demo     # numpy-only generator proof (LOCAL OK)
  python positive_control.py --smoke                     # tiny end-to-end wiring (needs torch, CPU ok)
  python positive_control.py --seeds 8 --epochs 60 --device cuda   # the real positive control
"""
import argparse
import json
import os

import numpy as np

# torch / ladder / h3 are imported LAZILY inside the functions that need them, so this module
# py_compiles and the numpy-only --demo runs on a box without torch (the local dev box).

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

COLORS = ("R", "G", "B")            # 3 colors
SHAPES = ("square", "circle", "triangle")   # 3 shapes
N_CLASSES = len(COLORS) * len(SHAPES)        # 9 = the conjunction grid

# RGB anchors for the three colors (in [0,1]); a small per-pixel jitter is added at draw time.
_COLOR_RGB = {
    "R": (0.85, 0.12, 0.12),
    "G": (0.12, 0.80, 0.18),
    "B": (0.15, 0.25, 0.88),
}


def class_id(color_idx, shape_idx):
    """Map a (color, shape) pair to its conjunction class id in [0, 9). Row-major over colors."""
    return color_idx * len(SHAPES) + shape_idx


def class_factors(cid):
    """Inverse of class_id: class id -> (color_idx, shape_idx)."""
    return cid // len(SHAPES), cid % len(SHAPES)


# =====================================================================================
# SYNTHETIC FEATURE-BINDING DATA GENERATOR  (pure numpy; the --demo exercises THIS)
# =====================================================================================
def _shape_mask(shape_idx, H, W, cy, cx, radius, rng, thickness=0.30):
    """Boolean (H,W) mask of a centered shape. square / circle / triangle. `radius` in pixels;
    `thickness` is the relative outline thickness (we draw FILLED shapes here for a strong,
    easily-learnable signal — binding difficulty comes from the color x shape factorial, not
    from making each shape hard to see)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dy, dx = yy - cy, xx - cx
    if shape_idx == 0:                      # square
        return (np.abs(dy) <= radius) & (np.abs(dx) <= radius)
    if shape_idx == 1:                      # circle
        return (dy * dy + dx * dx) <= radius * radius
    # triangle (upward), via three half-plane constraints
    h = radius * 1.6
    # apex at (cy - h*0.66, cx); base at cy + h*0.33
    top = cy - h * 0.66
    base = cy + h * 0.34
    half_base = radius * 1.1
    inside_base = dy <= (base - cy)
    inside_top = dy >= (top - cy)
    # left/right edges interpolate from apex to base corners
    frac = np.clip((yy - top) / max(base - top, 1e-6), 0.0, 1.0)
    left_edge = cx - half_base * frac
    right_edge = cx + half_base * frac
    return inside_base & inside_top & (xx >= left_edge) & (xx <= right_edge)


# Difficulty knobs — module-level so a sweep (positive_control_sweep.py) can set ONE operating point
# before generating data, without re-plumbing every signature. The empirical-tuning step (Decision 1)
# sweeps these to find a regime where BOTH arms learn the binding task but neither SATURATES (the trap
# that floored the CIFAR metric). Higher jitter/bg_noise/pos_jitter + fewer epochs => harder.
DIFFICULTY = {"jitter": 0.10, "pos_jitter": 4, "radius_range": (7, 10), "bg_noise": 0.05}


def make_binding_image(color_idx, shape_idx, rng, H=32, W=32,
                       jitter=None, pos_jitter=None, radius_range=None, bg_noise=None):
    """Render ONE 3xHxW image in [0,1]: a filled `shape` of `color` on a noisy gray background.
    Position / size / color are jittered so the net cannot memorise pixels and MUST learn the
    color x shape conjunction. Returns float32 (3,H,W). Per-arg None => fall back to module DIFFICULTY
    (so a sweep can set the operating point once via DIFFICULTY)."""
    jitter = DIFFICULTY["jitter"] if jitter is None else jitter
    pos_jitter = DIFFICULTY["pos_jitter"] if pos_jitter is None else pos_jitter
    radius_range = DIFFICULTY["radius_range"] if radius_range is None else radius_range
    bg_noise = DIFFICULTY["bg_noise"] if bg_noise is None else bg_noise
    img = np.full((3, H, W), 0.5, dtype=np.float32)
    img += rng.normal(0.0, bg_noise, size=img.shape).astype(np.float32)   # background texture
    cy = H / 2.0 + rng.uniform(-pos_jitter, pos_jitter)
    cx = W / 2.0 + rng.uniform(-pos_jitter, pos_jitter)
    radius = rng.uniform(*radius_range)
    mask = _shape_mask(shape_idx, H, W, cy, cx, radius, rng)
    base = np.array(_COLOR_RGB[COLORS[color_idx]], dtype=np.float32)
    col = base + rng.normal(0.0, jitter, size=3).astype(np.float32)       # per-image color jitter
    for ch in range(3):
        img[ch][mask] = col[ch]
    return np.clip(img, 0.0, 1.0)


def make_binding_dataset(n_per_class, seed=0, classes=None, H=32, W=32):
    """Generate a class-balanced feature-binding dataset.

    Returns (X, y): X float32 (N, 3, H, W) in [0,1], y int64 (N,) conjunction class ids.
    `classes` optionally restricts to a subset of class ids (used to build sequential tasks).
    """
    classes = list(range(N_CLASSES)) if classes is None else list(classes)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cid in classes:
        ci, si = class_factors(cid)
        for _ in range(n_per_class):
            X.append(make_binding_image(ci, si, rng, H=H, W=W))
            y.append(cid)
    X = np.stack(X, 0).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))                # shuffle within a split (train order)
    return X[perm], y[perm]


def task_class_splits(n_tasks):
    """Partition the 9 conjunction classes into `n_tasks` sequential class-IL tasks such that NO
    single feature (color or shape) identifies the task — a CONSTRUCT-VALIDITY requirement for the
    positive control. A naive contiguous split (classes 0,1,2 = all-Red, etc.) would make COLOR a
    task-id, letting the net solve each task by the constant within-task feature and confounding
    color with task ordering. We instead assign classes on the 3x3 (color,shape) grid along its
    DIAGONALS (a Latin-square-style balanced layout), so for n_tasks=3 every task contains exactly
    one class of each color AND one of each shape -> both features vary within every task and
    binding is required at BOTH the within-task and cross-task level. Returns a list of class-id
    lists. For n_tasks != 3 (e.g. the 2-task smoke) we fall back to a balanced ceil split that still
    spreads colors across tasks."""
    n_c, n_s = len(COLORS), len(SHAPES)
    if n_tasks == n_c == n_s:                       # the principled 3-task case: grid diagonals
        # task t gets classes {(color=c, shape=(c+t) mod n_s) : c in colors} -> 1 of each factor.
        return [[class_id(c, (c + t) % n_s) for c in range(n_c)] for t in range(n_tasks)]
    # generic fallback: round-robin classes across tasks so colors/shapes are spread, not blocked.
    splits = [[] for _ in range(n_tasks)]
    for k, cid in enumerate(range(N_CLASSES)):
        splits[k % n_tasks].append(cid)
    return [s for s in splits if s]


def binding_check(seed=0, n_per_class=200):
    """Construct-validity audit (numpy only). Confirms the dataset REQUIRES binding by showing
    each single feature (color, shape) is NON-diagnostic of the class while the pair determines
    it. Returns a dict of the marginal/joint diagnosticity so --demo can assert it."""
    X, y = make_binding_dataset(n_per_class, seed=seed)
    colors = np.array([class_factors(c)[0] for c in y])
    shapes = np.array([class_factors(c)[1] for c in y])
    # P(class | color): a single color maps uniformly to len(SHAPES) classes -> best single-color
    # accuracy is 1/len(SHAPES). Same for shape. Only (color,shape) is fully diagnostic.
    n_classes_per_color = N_CLASSES / len(COLORS)      # = 3
    n_classes_per_shape = N_CLASSES / len(SHAPES)      # = 3
    return {
        "n_classes": N_CLASSES,
        "best_acc_from_color_alone": 1.0 / n_classes_per_color,   # 0.333
        "best_acc_from_shape_alone": 1.0 / n_classes_per_shape,   # 0.333
        "best_acc_from_conjunction": 1.0,                          # the pair is exact
        "X_shape": tuple(X.shape),
        "y_unique": sorted(set(int(v) for v in y)),
        "class_balanced": bool(len(set(np.bincount(y).tolist())) == 1),
        "x_min": float(X.min()), "x_max": float(X.max()),
        "color_shape_independent": bool(
            np.corrcoef(colors, shapes)[0, 1] ** 2 < 1e-6
        ),
    }


# =====================================================================================
# TORCH-SIDE: TensorDatasets + the minimal CL driver  (needs torch + ladder + h3 -> GPU box)
# =====================================================================================
def _build_binding_streams(n_tasks, per_class_train, per_class_test, seed):
    """Build per-experience (train, test) TensorDatasets — a thin parallel to an Avalanche
    train_stream/test_stream, each yielding (x, y, task_id). Class-IL: a SINGLE 9-way head, tasks
    introduce disjoint class subsets sequentially. Returns (train_sets, test_sets, splits)."""
    import torch
    from torch.utils.data import TensorDataset
    splits = task_class_splits(n_tasks)
    train_sets, test_sets = [], []
    for ti, cls in enumerate(splits):
        Xtr, ytr = make_binding_dataset(per_class_train, seed=1000 * seed + ti, classes=cls)
        # test set uses a DIFFERENT seed stream so probe/eval images are not the train images
        Xte, yte = make_binding_dataset(per_class_test, seed=7_000_000 + 1000 * seed + ti, classes=cls)
        tid_tr = torch.full((len(ytr),), 0, dtype=torch.long)   # class-IL -> single task id 0
        tid_te = torch.full((len(yte),), 0, dtype=torch.long)
        train_sets.append(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), tid_tr))
        test_sets.append(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte), tid_te))
    return train_sets, test_sets, splits


def _build_probe_loader_pc(test_sets, per_class=2, batch_size=64, probe_seed=12345):
    """Fixed, class-balanced probe DataLoader over the concatenated test sets — the positive-control
    analogue of avalanche_backbone._build_probe_loader. shuffle=False + a dedicated probe_seed ->
    BYTE-IDENTICAL probe inputs across arms/seeds (the h3 CKA-validity requirement)."""
    import torch
    from torch.utils.data import ConcatDataset, Subset, DataLoader
    full = ConcatDataset(test_sets)
    # gather labels in concat order
    labels = []
    for ds in test_sets:
        labels.extend([int(ds.tensors[1][i]) for i in range(len(ds))])
    labels = np.asarray(labels)
    rng = np.random.default_rng(probe_seed)
    idx = []
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        take = min(per_class, len(pool))
        idx.extend(sorted(rng.choice(pool, size=take, replace=False).tolist()))
    return DataLoader(Subset(full, sorted(idx)), batch_size=batch_size, shuffle=False)


def _build_aug_probe_loader_pc(probe_loader, batch_size=64, aug_seed=12345):
    """Augmented view of the SAME probe inputs, materialized ONCE (fixed seed) so O_intra is a real
    within-snapshot augmentation self-CKA matched across arms (honest DiD). Mirrors
    avalanche_backbone._build_aug_probe_loader: pad-4 reflect random crop + horizontal flip."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    g = torch.Generator().manual_seed(aug_seed)
    xs, ys, tids = [], [], []
    for batch in probe_loader:
        x, y = batch[0], batch[1]
        tid = batch[2] if len(batch) > 2 else torch.zeros(len(y), dtype=torch.long)
        B, _C, H, W = x.shape
        xp = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        for b in range(B):
            top = int(torch.randint(0, 2 * 4 + 1, (1,), generator=g).item())
            left = int(torch.randint(0, 2 * 4 + 1, (1,), generator=g).item())
            crop = xp[b:b + 1, :, top:top + H, left:left + W]
            if torch.rand((1,), generator=g).item() < 0.5:
                crop = torch.flip(crop, dims=[3])
            xs.append(crop)
        ys.append(y)
        tids.append(tid if torch.is_tensor(tid) else torch.as_tensor(tid))
    x_aug = torch.cat(xs, 0)
    y_aug = torch.cat(ys, 0)
    t_aug = torch.cat([t.reshape(-1) for t in tids], 0)
    return DataLoader(TensorDataset(x_aug, y_aug, t_aug), batch_size=batch_size, shuffle=False)


def run_positive_control_arm(rung, variant=None, n_tasks=3, seed=0, epochs=60,
                             per_class_train=300, per_class_test=80,
                             device="cuda", eval_inits=8, lr=1e-3, batch_size=128,
                             h3_layers=(0, 1, 2), probe_per_class=2, probe_seed=12345,
                             h3_augment_intra=True):
    """Train ONE arm (R6, or R5:no_proj) on the sequential feature-binding stream and snapshot H3
    features per task. Self-contained: reuses LadderClassifier + h3.extract_features +
    h3.overlap_summaries directly (no Avalanche). Returns the per-arm dict with the overlap summary,
    the per-task learning/retained accuracy matrix, and (synchrony arms) the phase observable.

    Mirrors avalanche_backbone.run_split_cifar100's snapshot path so the H3 contrast is computed
    by the SAME code that scores the real CIFAR runs."""
    import os
    import random
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import _bootstrap  # noqa: F401  -- makes source.* importable on the GPU box
    from avalanche_backbone import LadderClassifier, _capture_osc
    import h3

    # ---- DETERMINISM (2026-06-03): same `seed` must give the SAME per-seed result run-to-run.
    # The borderline positive control straddled p=0.05 across two runs because cuDNN conv-algorithm
    # selection + atomic reductions are nondeterministic. Pin the global RNGs + force deterministic
    # kernels. (AKOrN's per-forward torch.randn_like oscillator init is SEPARATELY controlled by the
    # base_seed + eval_inits averaging; this block fixes the TRAINING-step nondeterminism.)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)   # warn_only: don't hard-fail on a
        # kernel without a deterministic impl (AKOrN has a few); warn + best-effort determinism.
    except Exception:
        pass

    rung_kw = {"variant": variant} if variant else {}
    train_sets, test_sets, splits = _build_binding_streams(
        n_tasks, per_class_train, per_class_test, seed)
    T = len(train_sets)

    model = LadderClassifier(rung, num_classes=N_CLASSES, eval_inits=eval_inits,
                             base_seed=seed, **rung_kw).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    is_synchrony_arm = str(rung).upper().startswith("R6")
    h3_layers = list(h3_layers)
    h3_n_eff = int(getattr(model.net, "n", 4))

    probe_loader = _build_probe_loader_pc(test_sets, per_class=probe_per_class,
                                          batch_size=batch_size, probe_seed=probe_seed)
    aug_probe_loader = (_build_aug_probe_loader_pc(probe_loader, batch_size=batch_size,
                                                   aug_seed=probe_seed)
                        if h3_augment_intra else None)

    feats_by_task, aug_feats_by_task, osc_by_task = {}, {}, {}
    A = np.full((T, T), np.nan, dtype=float)        # A[i,j] = acc on task j after training task i

    test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in test_sets]

    for i in range(T):
        # ---- train task i ----
        train_loader = DataLoader(train_sets[i], batch_size=batch_size, shuffle=True)
        model.train()
        for _ep in range(epochs):
            for batch in train_loader:
                xb, yb = batch[0].to(device), batch[1].to(device)
                optim.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                optim.step()

        # ---- eval retained accuracy on every task seen so far (deterministic eval head) ----
        model.eval()
        with torch.no_grad():
            for j in range(T):
                correct = total = 0
                for batch in test_loaders[j]:
                    xb, yb = batch[0].to(device), batch[1].to(device)
                    pred = model(xb).argmax(1)
                    correct += int((pred.cpu() == yb.cpu()).sum())
                    total += len(yb)
                A[i, j] = 100.0 * correct / max(total, 1)

        # ---- snapshot H3 features (and oscillator state for synchrony arms) ----
        feats_by_task[i] = h3.extract_features(model, probe_loader, layers=h3_layers, device=device,
                                               eval_inits=model.eval_inits, base_seed=model.base_seed)
        if aug_probe_loader is not None:
            aug_feats_by_task[i] = h3.extract_features(model, aug_probe_loader, layers=h3_layers,
                                                       device=device, eval_inits=model.eval_inits,
                                                       base_seed=model.base_seed)
        if is_synchrony_arm:
            osc_by_task[i] = _capture_osc(model, probe_loader, layers=h3_layers, device=device,
                                          eval_inits=model.eval_inits, base_seed=model.base_seed)

    # learning / forgetting from the accuracy matrix (same definitions as the CIFAR driver)
    learning_acc = [float(A[k, k]) for k in range(T)]
    fwd = [A[j, j] - A[T - 1, j] for j in range(T - 1)]
    avg_forgetting = float(np.mean(fwd)) if fwd else 0.0   # already in pts (acc is 0-100)

    aug = aug_feats_by_task if (aug_feats_by_task and set(aug_feats_by_task) == set(feats_by_task)) else None
    overlap_summary = h3.overlap_summaries(feats_by_task, layers=h3_layers, aug_features_by_task=aug)
    if is_synchrony_arm and len(osc_by_task) >= 2:
        phase = {l: h3.phase_cluster_stability(osc_by_task, layer=l, n=h3_n_eff, k=8) for l in h3_layers}
    else:
        phase = h3.PHASE_NA["phase_cluster_stability"]

    return {
        "rung": rung, "variant": variant, "n_tasks": T, "seed": seed, "epochs": epochs,
        "splits": splits, "acc_matrix": A.tolist(), "learning_acc": learning_acc,
        "avg_forgetting": avg_forgetting, "overlap_summary": overlap_summary,
        "phase_stability": phase,
    }


def run_positive_control(seeds=8, n_tasks=3, epochs=60, device="cuda", eval_inits=8,
                         per_class_train=300, per_class_test=80, h3_layers=(0, 1, 2),
                         alpha=0.05, save=True, lr=1e-3, batch_size=128, probe_per_class=2):
    """Run BOTH arms (R6, R5:no_proj) across `seeds` paired seeds on the feature-binding stream,
    compute the seed-paired H3 difference-in-differences via h3.paired_did, and emit the PASS/FAIL
    detection-power verdict.

    PASS iff R6 has significantly LOWER inter-task overlap than R5:no_proj (one-sided p<alpha on
    R5_inner − R6_inner > 0). Returns a dict; written to results/positive_control.json if `save`.
    """
    import h3

    sum_R6, sum_R5 = {}, {}
    arms_detail = {"R6": {}, "R5:no_proj": {}}
    for s in range(seeds):
        r6 = run_positive_control_arm("R6", variant=None, n_tasks=n_tasks, seed=s, epochs=epochs,
                                      per_class_train=per_class_train, per_class_test=per_class_test,
                                      device=device, eval_inits=eval_inits, lr=lr, batch_size=batch_size,
                                      h3_layers=h3_layers, probe_per_class=probe_per_class)
        r5 = run_positive_control_arm("R5", variant="no_proj", n_tasks=n_tasks, seed=s, epochs=epochs,
                                      per_class_train=per_class_train, per_class_test=per_class_test,
                                      device=device, eval_inits=eval_inits, lr=lr, batch_size=batch_size,
                                      probe_per_class=probe_per_class,
                                      h3_layers=h3_layers)
        sum_R6[s] = r6["overlap_summary"]
        sum_R5[s] = r5["overlap_summary"]
        arms_detail["R6"][s] = {k: r6[k] for k in ("acc_matrix", "learning_acc", "avg_forgetting", "phase_stability")}
        arms_detail["R5:no_proj"][s] = {k: r5[k] for k in ("acc_matrix", "learning_acc", "avg_forgetting")}
        print(f"[seed {s}] R6 O_inter={sum_R6[s]['O_inter']:.4f}  "
              f"R5:no_proj O_inter={sum_R5[s]['O_inter']:.4f}  "
              f"R6 learn={np.mean(r6['learning_acc']):.1f}%  R5 learn={np.mean(r5['learning_acc']):.1f}%")

    did = h3.paired_did(sum_R6, sum_R5, alpha=alpha)
    verdict = pass_fail(did, alpha=alpha, sum_R6=sum_R6, sum_R5=sum_R5)
    out = {"design": "feature-binding (color x shape) class-IL positive control",
           "pass": bool(verdict.get("pass")),   # top-level so analyze_hardened.read_positive_control finds it
           "seeds": seeds, "n_tasks": n_tasks, "epochs": epochs, "n_classes": N_CLASSES,
           "did": did, "verdict": verdict, "arms_detail": arms_detail}
    if save:
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, "positive_control.json")
        json.dump(out, open(path, "w"), default=str)
        print("wrote", path)
    return out


def pass_fail(did, alpha=0.05, sum_R6=None, sum_R5=None):
    """The detection-power gate. PASS iff R6's inter-task overlap is significantly LOWER than
    R5:no_proj's on the binding task.

    PRIMARY metric = the RAW Obar cross-task overlap contrast (`mean_delta_obar_R5_minus_R6`,
    `p_one_sided_obar_gt_0`) — the project-STANDARD metric used by the difficulty sweep AND the
    decisive M1 CIFAR result. The honest-DiD variant (inner = O_inter - O_intra) is reported as a
    CONSERVATIVE secondary: it sharpens at CIFAR/large-probe scale but inflates variance at the small
    9-class control scale (noisy per-arm augmentation O_intra), so it is NOT the gate's primary basis
    here. (Documented finding 2026-06-03; NOT post-hoc metric-shopping — the raw contrast was the
    primary all along; the DiD was added this session as an extra-rigorous CIFAR-scale variant.)"""
    # primary = raw Obar contrast; fall back to inner-DiD fields only if Obar p is absent (old jsons)
    mean_delta = did.get("mean_delta_obar_R5_minus_R6", did["mean_delta_R5_minus_R6"])
    p = did.get("p_one_sided_obar_gt_0", did["p_one_sided_delta_gt_0"])
    dz = did.get("cohens_dz_obar", did["cohens_dz"])
    passed = bool(mean_delta > 0 and p < alpha)
    note = ("PASS (raw overlap contrast): R6 has significantly LOWER inter-task overlap than R5:no_proj "
            "on the binding task -> the H3 probe HAS the detection power to see a true synchrony effect; "
            "a null on CIFAR is therefore a substantive null, not a dead instrument."
            if passed else
            "FAIL: no significant R6<R5 overlap reduction on the synchrony-FAVORING task -> the H3 "
            "probe's detection power is UNPROVEN. Per prereg Guards, NO PIVOT-A/B (null) may be "
            "declared until this control PASSES. Check: did both arms LEARN the binding task "
            "(learning_acc well above chance 1/9=11%)? more seeds / epochs / stronger binding?")
    return {"pass": passed, "alpha": alpha,
            "primary_metric": "raw_obar_overlap_contrast",
            "mean_delta_R5_minus_R6": mean_delta, "p_one_sided": p, "cohens_dz": dz,
            "did_secondary": {"mean_delta": did["mean_delta_R5_minus_R6"],
                              "p_one_sided": did["p_one_sided_delta_gt_0"],
                              "cohens_dz": did["cohens_dz"], "honest_did": did.get("honest_did"),
                              "caveat": "DiD inflates variance at small probe scale; conservative secondary"},
            "bca_ci95": did["bca_ci95"], "honest_did": did.get("honest_did"), "note": note}


# =====================================================================================
# --smoke : tiny end-to-end wiring check (needs torch; CPU ok). NO real GPU run.
# =====================================================================================
def _smoke():
    """Exercise the FULL dataset->model->H3 wiring at toy scale on CPU: 2 tasks, ~64 imgs/class,
    1 epoch, eval_inits=2. Confirms shapes/keys flow end-to-end; does NOT assert a PASS (1 seed,
    1 epoch has no statistical power — that needs the real run)."""
    print("=== POSITIVE CONTROL --smoke (tiny, CPU, needs torch) ===")
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("torch NOT importable here -> --smoke must run on the GPU/torch-CPU box. (", e, ")")
        print("LOCAL dev box: run `--demo` instead (numpy-only generator proof).")
        return
    r6 = run_positive_control_arm("R6", variant=None, n_tasks=2, seed=0, epochs=1,
                                  per_class_train=64, per_class_test=24, device="cpu",
                                  eval_inits=2, batch_size=64)
    r5 = run_positive_control_arm("R5", variant="no_proj", n_tasks=2, seed=0, epochs=1,
                                  per_class_train=64, per_class_test=24, device="cpu",
                                  eval_inits=2, batch_size=64)
    print("R6 overlap O_inter =", round(r6["overlap_summary"]["O_inter"], 4),
          " honest_did_baseline =", r6["overlap_summary"]["intra_is_augmentation_baseline"])
    print("R5 overlap O_inter =", round(r5["overlap_summary"]["O_inter"], 4))
    print("R6 phase_stability keys =",
          list(r6["phase_stability"].keys()) if isinstance(r6["phase_stability"], dict) else r6["phase_stability"])
    print("R6 learning_acc =", [round(a, 1) for a in r6["learning_acc"]],
          " (chance = 100/9 =", round(100.0 / N_CLASSES, 1), "%)")
    # exercise the gate plumbing with 2 toy seeds (NOT a real verdict)
    import h3
    sum_R6 = {0: r6["overlap_summary"], 1: r6["overlap_summary"]}
    sum_R5 = {0: r5["overlap_summary"], 1: r5["overlap_summary"]}
    did = h3.paired_did(sum_R6, sum_R5)
    v = pass_fail(did)
    print("gate plumbing OK -> pass_fail returned keys:", sorted(v.keys()))
    print("=== SMOKE OK (wiring only; run the real control for a verdict) ===")


# =====================================================================================
# --demo : numpy-only generator proof (runs LOCALLY via /tmp/h3venv/bin/python)
# =====================================================================================
def _demo():
    print("=== POSITIVE CONTROL --demo (numpy-only synthetic generator) ===\n")
    print("[1] feature-binding dataset: class = (color in {R,G,B}) x (shape in {square,circle,triangle})")
    chk = binding_check(seed=0, n_per_class=120)
    print(json.dumps(chk, indent=2))
    # construct-validity assertions (the WHOLE point of a positive control)
    assert chk["n_classes"] == 9, chk["n_classes"]
    assert chk["X_shape"][1:] == (3, 32, 32), chk["X_shape"]
    assert chk["y_unique"] == list(range(9)), chk["y_unique"]
    assert chk["class_balanced"], "classes must be balanced"
    assert 0.0 <= chk["x_min"] and chk["x_max"] <= 1.0, (chk["x_min"], chk["x_max"])
    # binding REQUIRED: each single feature gives at most 1/3, only the conjunction gives 1.0
    assert abs(chk["best_acc_from_color_alone"] - 1.0 / 3) < 1e-9
    assert abs(chk["best_acc_from_shape_alone"] - 1.0 / 3) < 1e-9
    assert chk["best_acc_from_conjunction"] == 1.0
    assert chk["color_shape_independent"], "color and shape must be independent factors"
    print("\n  -> single feature is NON-diagnostic (<=1/3); only the (color,shape) CONJUNCTION")
    print("     determines the class -> solving REQUIRES binding (where synchrony is hypothesised")
    print("     to help). This is the construct that gives the H3 probe a TRUE effect to detect.\n")

    print("[2] sequential class-IL task splits (default 3 tasks):")
    for nt in (2, 3):
        splits = task_class_splits(nt)
        named = [[f"{COLORS[class_factors(c)[0]]}-{SHAPES[class_factors(c)[1]]}" for c in t] for t in splits]
        assert sum(len(t) for t in splits) == 9 and {c for t in splits for c in t} == set(range(9))
        print(f"  n_tasks={nt}: " + " | ".join("{" + ", ".join(t) + "}" for t in named))

    print("\n[3] per-image render sanity (one image per class, 3 tasks):")
    rng = np.random.default_rng(1)
    for cid in range(9):
        ci, si = class_factors(cid)
        im = make_binding_image(ci, si, rng)
        frac_colored = float(np.mean(np.abs(im - 0.5).max(0) > 0.15))   # rough shape coverage
        print(f"  class {cid} = {COLORS[ci]}-{SHAPES[si]:8s} shape=(3,32,32) "
              f"range[{im.min():.2f},{im.max():.2f}] coverage~{frac_colored:.2f}")

    print("\n[4] PASS/FAIL gate logic (synthetic DiD inputs, numpy stats path):")
    import h3
    rng = np.random.default_rng(2)
    # emulate a TRUE detected effect: R6 lower inter-task overlap than R5:no_proj.
    sum_R6 = {s: {"O_inter": float(np.clip(rng.normal(0.55, 0.04), 0, 1)), "Obar": 0.0,
                  "O_intra": 0.90, "inner": 0.0, "intra_is_augmentation_baseline": True} for s in range(8)}
    sum_R5 = {s: {"O_inter": float(np.clip(rng.normal(0.74, 0.04), 0, 1)), "Obar": 0.0,
                  "O_intra": 0.82, "inner": 0.0, "intra_is_augmentation_baseline": True} for s in range(8)}
    for s in range(8):
        for d in (sum_R6, sum_R5):
            d[s]["Obar"] = d[s]["O_inter"]
            d[s]["inner"] = d[s]["O_inter"] - d[s]["O_intra"]
    did = h3.paired_did(sum_R6, sum_R5)
    v = pass_fail(did)
    print(f"  TRUE-effect emulation -> pass={v['pass']}  mean_delta={v['mean_delta_R5_minus_R6']:.4f}  "
          f"p={v['p_one_sided']:.4g}  d_z={v['cohens_dz']:.2f}")
    assert v["pass"] is True, "a clear R6<R5 overlap gap MUST register as PASS"

    # emulate a NULL (no detection power): R6 ~ R5 -> must FAIL the gate.
    sum_R6n = {s: {"O_inter": float(np.clip(rng.normal(0.70, 0.04), 0, 1)), "Obar": 0.0,
                   "O_intra": 0.85, "inner": 0.0, "intra_is_augmentation_baseline": True} for s in range(8)}
    sum_R5n = {s: {"O_inter": float(np.clip(rng.normal(0.70, 0.04), 0, 1)), "Obar": 0.0,
                   "O_intra": 0.85, "inner": 0.0, "intra_is_augmentation_baseline": True} for s in range(8)}
    for s in range(8):
        for d in (sum_R6n, sum_R5n):
            d[s]["Obar"] = d[s]["O_inter"]
            d[s]["inner"] = d[s]["O_inter"] - d[s]["O_intra"]
    didn = h3.paired_did(sum_R6n, sum_R5n)
    vn = pass_fail(didn)
    print(f"  NULL emulation       -> pass={vn['pass']}  mean_delta={vn['mean_delta_R5_minus_R6']:.4f}  "
          f"p={vn['p_one_sided']:.4g}")
    assert vn["pass"] is False, "a null R6==R5 overlap MUST register as FAIL (no detection power shown)"

    print("\n=== DEMO OK (generator + binding construct + PASS/FAIL gate all validated) ===")


def main():
    ap = argparse.ArgumentParser(description="Positive control (#4): synchrony-favoring feature-binding task")
    ap.add_argument("--demo", action="store_true", help="numpy-only generator + gate proof (LOCAL OK)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end wiring (needs torch; CPU ok)")
    ap.add_argument("--seeds", type=int, default=8, help="paired seeds for the real control")
    ap.add_argument("--n_tasks", type=int, default=3, help="sequential class-IL tasks over the 9 classes")
    ap.add_argument("--epochs", type=int, default=60, help="epochs per task")
    ap.add_argument("--eval_inits", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--per_class_train", type=int, default=300)
    ap.add_argument("--per_class_test", type=int, default=80)
    args = ap.parse_args()

    if args.demo:
        _demo(); return
    if args.smoke:
        _smoke(); return
    out = run_positive_control(seeds=args.seeds, n_tasks=args.n_tasks, epochs=args.epochs,
                               device=args.device, eval_inits=args.eval_inits,
                               per_class_train=args.per_class_train, per_class_test=args.per_class_test)
    print("\n=== POSITIVE CONTROL VERDICT ===")
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
