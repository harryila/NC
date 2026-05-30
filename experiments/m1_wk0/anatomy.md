# AKOrN anatomy — verified code map for the M1 ladder

*Read from `github.com/autonomousvision/akorn` @ HEAD (2026-05-30). Line numbers are that checkout; re-confirm against yours.* This turns the abstract ladder (R1→R6) into concrete intervention points.

## The classification model
`source/models/classification/knet.py` → `class AKOrN(nn.Module)`
- **Backbone (keep bit-identical across all rungs):** `conv0` (3×3) → 3 layers, each `[strided_conv(transition), Identity, KLayer, readout_block, Identity]` → `AdaptiveAvgPool2d` → `out`.
- **The core to swap per rung = index 2 of each layer** (`knet.py:168-174`); `feature()` unpacks it as `k_layer` (`knet.py:186-192`). **Injection point:** `model.layers[l][2] = <rung core>`.
- **Head (knet.py:85):** `self.out = nn.Linear(channels[-1], out_classes)` — fixed. For class-IL on CIFAR-100, build with `out_classes=100` and let Avalanche present class-incremental experiences (no head-growth needed); use a multi-head wrapper only for task-IL.
- **⚠ Stochastic init (knet.py:182):** `x = torch.randn_like(c)` — oscillator state re-randomized **every forward** → stochastic logits. Must be made deterministic for eval/BWT/CKA (see `avalanche_backbone.py`).
- **Ensemble (knet.py:221-229):** `ensemble>1` averages logits over N random-init forwards — the *only* test-time knob on the classification path (NB: **energy-voting is Sudoku-only**, not here).
- **Default norm = `"bn"` (knet.py:51)** → switch to `"gn"` for all rungs (BN running-stats drift across CL tasks is a confound).

## The Kuramoto core (the rung ingredients live here)
`source/layers/klayer.py` → `class KLayer`, `forward(x, c, T, gamma)` (klayer.py:152-165):
```
c = self.c_norm(c)
x = normalize(x, n)                       # (4) SPHERICAL NORMALIZATION (per n-dim group, onto unit sphere)
for t in range(T):                        # (5) RECURRENCE (T steps)
    dxdt, sim = self.kupdate(x, c)
    x = normalize(x + gamma*dxdt, n)      # (4) re-project to sphere each step
```
`kupdate` (klayer.py:126-150):
```
_y = self.connectivity(x)                 # relational coupling (conv or attn)  -> J
y  = _y + c                               # conditional stimulus / bias
omg_x = self.omg(x)                       # Ω natural-frequency term (OmegaLayer)
if self.apply_proj:                       # (6) SYNCHRONY: tangent-space projection
    y_yxx, sim = self.project(y, x)       #     project() = y - (x·y)x  (klayer.py:121-124)
dxdt = omg_x + reshape_back(y_yxx)
```
- **`normalize(x, n)`** (`kutils.py:31-35`) = per-group L2 normalize onto the unit sphere → ingredient (4).
- **`project(y, x) = y - (x·y)x`** (`klayer.py:121-124`) = the tangent-space Kuramoto coupling that rotates oscillators toward alignment → **this IS "synchrony"**, gated by **`apply_proj`** (`klayer.py:86, 141`).
- **`OmegaLayer`** (`klayer.py:21-58`) = the Ω term; `use_omega`, `learn_omg` flags (the paper notes energy-voting needs Ω).
- **`n`** = oscillator (vector) dimension → ingredient (3) "vector coding"; `n=1` ⇒ scalar.

## Ladder ↔ code mapping
| Rung | Ingredient added | How (against real code) |
|---|---|---|
| R1 | dense scalar | `LadderCore(n=1, sparsity=None, spherical=False)`, T=1 |
| R2 | +structured sparsity | `LadderCore(n=1, sparsity=p*)` (grouped k-WTA), T=1 |
| R3 | +vector coding | `LadderCore(n=N, sparsity=p*)`, T=1 |
| R4 | +spherical norm | `LadderCore(n=N, sparsity=p*, spherical=True)`, T=1 |
| **R5** | +recurrence, **no synchrony** | `KLayer(..., apply_proj=False)`, T=3 — variants: `no_proj` / `depthwise` (1×1 groups=ch connectivity) / `frozen_J` |
| **R6** | **+synchrony** | `KLayer(..., apply_proj=True)`, T=3 — **stock AKOrN** |

## The decisive contrast is clean by construction
**R6 − R5(`no_proj`) is a single boolean flip (`apply_proj`) inside bit-identical KLayer machinery** — identical connectivity, Ω, normalization, recurrence, bias, readout. That is the strongest possible isolation of synchrony. R1–R4 (LadderCore) contextualize as lower bounds; the **R4→R5 seam** bundles recurrence+Ω+KLayer-mechanics and is *not* single-ingredient — so the gate rides on R6−R5, not on the seam. **Bracket the R5 definition:** run `no_proj` (surgical), `depthwise` (no neuron-neuron coupling — recommended primary), and `frozen_J` (no learned coupling); report R6 minus each.

## Training recipe (mirror it; `train_classification.py`)
Adam, **lr 1e-4**, wd 0, **400 epochs**, batch 128, CE, EMA(β=0.99), strong aug (`augmentation_strong(imsize=32)`), **CIFAR-10 only** (`train_classification.py:188` raises otherwise → CIFAR-100/Split benchmarks are your build). Model defaults: `n=2(CLI)/4(class)`, ch=64, L=3, T=3, J=conv.
