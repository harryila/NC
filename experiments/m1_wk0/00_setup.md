# Wk-0 setup

> **The fast path is `make setup` from the repo root** (runs `setup.sh`: pins AKOrN into `external/akorn`, installs `requirements.txt`). The manual steps below are the equivalent if you prefer to do it by hand. See [../../RUNNING.md](../../RUNNING.md) for the full GPU-box guide + the run DAG.

## 1. Environment (GPU box) — manual equivalent of `make setup`
```bash
bash setup.sh                             # from repo root: clone+pin AKOrN, install lean deps
# (manual:)
# git clone https://github.com/autonomousvision/akorn.git external/akorn
# pip install torch torchvision avalanche-lib einops ema_pytorch scipy scikit-learn tqdm tensorboard
```

## 2. Native repro (validate the ACTUAL codepath you'll use)
The CL study runs on `source/models/classification/knet.py` — so reproduce **CIFAR-10 classification**, NOT Sudoku/CLEVRTex (different model/loss/harness):
```bash
python train_classification.py wk0_smoke --data cifar10 --epochs 1   # smoke (1 epoch)
# then a short real run (e.g. --epochs 50) and check accuracy climbs sensibly.
```
Gate: AKOrN classification trains and `ensemble>1` inference behaves as documented before any wrapping.

## 3. Drop in the scaffolding
Put `ladder.py`, `avalanche_backbone.py`, `budget.py` (this folder) where the repo is importable:
```bash
export PYTHONPATH=/path/to/akorn
python ladder.py            # smoke: builds R1-R6, prints param-match table + logit shapes (CPU ok)
python avalanche_backbone.py   # smoke: eval deterministic == True, train stochastic == True
python budget.py            # GPU: ms/step, peak VRAM, GPU-hour extrapolation, R6 fraction-active
```

## 4. Known build work (not yet done — Wk-1/2)
- **R1–R4 `LadderCore`**: validate it trains; tune k-WTA to the `fraction_active` target from `budget.py`; confirm param/FLOP match (±2%) via `param_report` + add `fvcore`/`ptflops` for FLOPs.
- **CIFAR-100 / Split benchmarks**: `out_classes=100`; wire `SplitCIFAR100` (class-IL arbiter) + a longer stream (20×5) and Split-MNIST sanity.
- **Class-IL head**: fixed 100-way is fine for class-IL; add a multi-head wrapper only for the task-IL diagnostic.
- **A5 anchor**: add DER++/FOSTER on the same backbone (with/without replay).
- **Metrics/stats**: CKA + activation-support-overlap probes; exact paired permutation test; TOST.

## 5. Reconfirm against your checkout
Line numbers in `anatomy.md` are from HEAD on 2026-05-30. Re-grep `apply_proj`, `torch.randn_like`, `self.out =`, and `normalize(` before trusting the injection points.
