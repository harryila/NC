# GATE A — EXECUTION RUNBOOK (AKOrN native object discovery, 1× RTX 4090)

**Goal:** Test the "coupling is causally inert" finding on AKOrN's *own* object-discovery
benchmark, where the ICLR-2025 Oral credits the Kuramoto coupling. Reproduce
**AKOrN^attn L=1 on CLEVRTex-full** (target ≈ **75.79 FG-ARI / 54.94 MBO**, the ~9.7pt FG-ARI gap
over ItrSA that the paper attributes to coupling), then run the **param-matched coupling-severed**
model under an identical budget and compare.

**Authoritative source read for this runbook:** `/tmp/akorn_src` (synced from the box).
Every nontrivial claim below cites `file:line` in that tree. Read it yourself before running.

---

## 0. CRITICAL PRE-FLIGHT FACTS (do not skip)

### 0.1 Tetrominoes is a TRAP as a headline (smoke-test ONLY)
AKOrN's own appendix shows AKOrN^attn does *not* beat ItrSA on Tetrominoes; a null there is a
known reproduction and reviewer-fatal as a headline. We use Tetrominoes **only** as a cheap
end-to-end smoke test (loaders, augs, loss, eval, severance plumbing all wired). The decisive
venue is **CLEVRTex-full**. The synced perf table (`scripts/synths.md:50-55`) lists Tetrominoes
AKOrN^attn 88.6 FG-ARI / 56.4 MBO at the model's native 32px (no up-tiling); the task's
"~86 / ~55" target is the L=1 no-uptile operating point — treat **≥85 FG-ARI / ≥54 MBO** as
smoke-pass, not as a result.

### 0.2 TWO FILES ARE MISSING FROM THE SYNCED TREE — recover before you start
The sync `/tmp/akorn_src` is **incomplete vs the upstream AKOrN repo**:

1. **`eval_obj.py` is NOT in `/tmp/akorn_src`.** Confirmed: the only eval scripts present are
   `source/evals/sudoku/evals.py`, plus the metric *helpers* `source/evals/objs/fgari.py`
   (`calc_fgari_score`) and `source/evals/objs/mbo.py` (`calc_mean_best_overlap`). The
   driver `eval_obj.py` that `scripts/synths.md:46` calls lives at the **upstream repo root**.
   → It IS present in the box's pinned checkout at `$AKORN_HOME` (`external/akorn`, set by
   `env.sh` / `setup.sh`). **Run eval from `$AKORN_HOME`, not from `/tmp/akorn_src`.** If it is
   somehow absent there too, `git checkout` it from the upstream `takerum/akorn` repo at the
   pinned commit. **Verify it exists before training anything** (Step 1.4).

2. **`native_severance.py` does NOT exist yet** anywhere (`/tmp/akorn_src` or `NeuralCombs`).
   It must be authored as a separate deliverable (the param-match spec is fully pinned in §6
   below so the severed run is a one-line `--J` swap). This runbook gives the exact commands
   assuming `native_severance.py` registers a `--J=severed` connectivity; if your
   implementation exposes it differently, adjust only the `--J` token.

### 0.3 Hardware note
Task says **1× RTX 4090 (24GB)**. The repo's `gpu_info.txt` shows an **A100-80GB** from an
earlier session — do **not** trust it; size everything for **24GB** (§5.2). If you are in fact
on the A100, the batch sizes below are safe (smaller than necessary) and you can raise
`--batchsize` toward the default 256 with `--num_processes` accordingly.

---

## 1. ENVIRONMENT SETUP

### 1.1 What's already on the box (per task)
Present: `torch 2.4.1+cu124`, `torchvision`, `ema_pytorch`, `einops`, `sklearn`
(scikit-learn), `scipy`, `accelerate`, `timm`, `fastcluster`, `h5py`. Being installed:
`accelerate`, `timm`, `fastcluster`, `h5py`. `external/akorn` pinned via `setup.sh`. **NO gsutil.**

### 1.2 What the code actually imports (cross-checked against source)
- `train_obj.py:15-16` → `accelerate`, `Accelerator` ✓
- `train_obj.py:9-13` → `source.*`, `torch.utils.tensorboard.SummaryWriter` (needs **tensorboard**)
- `train_obj.py:20` → `tqdm` ✓
- `train_obj.py:346` → `from ema_pytorch import EMA` ✓
- `source/layers/kutils.py:1` → **`from sympy import prod`** — `sympy` is imported at module top
  even though `prod` is unused. **`sympy` must be installed or every model import fails.**
  This is the easiest thing to miss. Install it.
- `source/layers/klayer.py:3` → `torch.nn.utils.parametrizations.weight_norm` (stdlib torch) ✓
- `source/data/datasets/objs/clevr_tex.py:10-12` → `scipy.optimize.linear_sum_assignment`,
  `sklearn.metrics.adjusted_rand_score`, `PIL` ✓
- Tetrominoes **TFRecord→npz conversion only** (`data/convert_tfrecord_to_np.py:12-16`,
  `data/tfloaders/tetrominoes.py:17`) needs **`tensorflow`** (`import tensorflow.compat.v1`).
  TF is needed **only on the machine doing the one-time conversion**, NOT for training/eval.
  If you use a pre-converted `.npz` mirror (§2.1 option B), you never install TF.

### 1.3 Install (idempotent)
```bash
source ~/Desktop/.../NeuralCombs/env.sh   # sets AKORN_HOME + PYTHONPATH=external/akorn
pip install sympy tensorboard            # sympy is REQUIRED (kutils import); tensorboard for SummaryWriter
# already present per task: accelerate timm fastcluster h5py ema_pytorch einops scikit-learn scipy tqdm pillow
# Tetrominoes conversion box only (skip if using a pre-converted .npz):
#   pip install 'tensorflow-cpu==2.13.*'   # GZIP TFRecord reader; CPU build is enough
```
`requirements.txt` in NeuralCombs (root) intentionally does **not** list AKOrN's deps
(`requirements.txt:2` — "AKOrN itself is pinned into external/akorn by setup.sh"); there is **no
`requirements.txt` inside `/tmp/akorn_src`** (confirmed). So the lines above are the source of truth.

### 1.4 PRE-FLIGHT VERIFY (run all, must pass before any training)
```bash
cd "$AKORN_HOME"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import sympy, tensorboard, accelerate, ema_pytorch, einops, scipy, sklearn; print('deps ok')"
python -c "from source.models.objs.knet import AKOrN; print('AKOrN import ok')"   # fails loudly if sympy missing
python -c "from source.layers.klayer import KLayer; print('KLayer ok')"
ls -la eval_obj.py train_obj.py        # BOTH must exist here; eval_obj.py is the missing-from-/tmp one
accelerate config default || true       # single-process default is fine for 1 GPU
```

---

## 2. DATA

### 2.1 Tetrominoes WITHOUT gsutil (SMOKE)

The loader wants a **single `.npz`**:
`./data/tetrominoes/tetrominoes_{train,test}.npz` (`tetrominoes.py:14-21`,
`load_data.py:97-111`, default `data_root="./data/tetrominoes/"`). The npz must contain keys
`images` (float, shape `[N,3,32,32]`, range [0,1]) and `labels` (uint8 `[N,32,32]` pixelwise
instance ids) — see `npdataset.py:20-24` and the writer `convert_tfrecord_to_np.py:159-184`.
Raw TFRecords are at `gs://multi-object-datasets` (`data/download_synths.sh:1-4`). With **no
gsutil**, pick ONE:

**Option A — install a gsutil-free downloader, then convert (canonical path).**
The bucket `gs://multi-object-datasets` is **public**, so you can pull the single object over
plain HTTPS without gsutil/auth:
```bash
cd "$AKORN_HOME/data"
mkdir -p tetrominoes
# public-bucket HTTPS endpoint (no auth, no gsutil):
wget -O tetrominoes/tetrominoes_train.tfrecords \
  "https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords"
# convert TFRecord -> the npz splits the loader wants (needs tensorflow-cpu from §1.3):
python convert_tfrecord_to_np.py --dataset_name=tetrominoes
# writes (convert_tfrecord_to_np.py:159-184, output_path "./tetrominoes/"):
#   tetrominoes/tetrominoes_eval.npz  (64)
#   tetrominoes/tetrominoes_test.npz  (320)   <-- eval_obj.py reads this (split=test)
#   tetrominoes/tetrominoes_val.npz   (10000)
#   tetrominoes/tetrominoes_train.npz (~49616) <-- train_obj.py reads this (split=train)
```
Note: the converter writes into `./tetrominoes/` relative to CWD, i.e. `data/tetrominoes/` —
exactly `data_root`. Center-crop to 32px happens inside the converter
(`convert_tfrecord_to_np.py:26-29,137-139`); the loader's nominal imsize is 32
(`load_data.py:99`).

**Option B — pip gsutil (fallback if the HTTPS object name 404s).**
`gsutil` *is* pip-installable (no Google Cloud SDK needed):
```bash
pip install gsutil
gsutil -o "GSUtil:state_dir=$AKORN_HOME/data/.gsutil" cp \
  gs://multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords "$AKORN_HOME/data/tetrominoes/"
python "$AKORN_HOME/data/convert_tfrecord_to_np.py" --dataset_name=tetrominoes
```
(This mirrors `download_synths.sh:2-5` but for tetrominoes only and with pip-gsutil.)

**Option C — pre-converted .npz mirror (skips TF entirely).** If a colleague/box already has
`tetrominoes_{train,test}.npz` in the exact `npdataset.py` schema, copy them straight into
`$AKORN_HOME/data/tetrominoes/`. **Validate before trusting:**
```bash
python - <<'PY'
import numpy as np
d=np.load("data/tetrominoes/tetrominoes_train.npz")
print("keys",list(d.keys()),"images",d["images"].shape,d["images"].dtype,
      "labels",d["labels"].shape, "img range",float(d["images"].min()),float(d["images"].max()))
# expect: images (N,3,32,32) float in [0,1]; labels (N,32,32) small ints
PY
```

### 2.2 CLEVRTex-full via Oxford (DECISIVE) — no gsutil needed

CLEVRTex ships as Oxford tarballs over plain HTTPS (`data/download_clevrtex.sh:4-12`). The loader
expects a **directory tree of PNGs+JSON**, not npz. Exact expectations from `clevr_tex.py`:
- default root `./data/clevr_tex/clevrtex_full` (`load_data.py:18`)
- it appends the variant subfolder `clevrtex_full` (`clevr_tex.py:94-95,130-132`) → final
  basepath `./data/clevr_tex/clevrtex_full/clevrtex_full`
- inside, files named `CLEVRTEX_full_NNNNNN.png`, `CLEVRTEX_full_NNNNNN_flat.png` (mask),
  `CLEVRTEX_full_NNNNNN.json` (`clevr_tex.py:52-62`). Globbed recursively (`**/`,
  `clevr_tex.py:59`), so the extracted nested layout is fine.
- splits are computed by index fraction at load time: test=[0,0.1), val=[0.1,0.2),
  train=[0.2,1.0) (`clevr_tex.py:30`). **No separate test download** — one `clevrtex_full`
  tree serves train+eval.

```bash
cd "$AKORN_HOME/data"
mkdir -p clevr_tex && cd clevr_tex
# clevrtex_full is 5 parts (~38GB total). For GATE A you need ONLY clevrtex_full
# (train+test come from index slicing). The outd/camo tarballs are extra eval variants — SKIP for Gate A.
for p in 1 2 3 4 5; do
  wget -c --show-progress \
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part${p}.tar.gz"
done
for p in 1 2 3 4 5; do tar -xzf "clevrtex_full_part${p}.tar.gz"; done
# (Tarballs extract to a clevrtex_full/ dir; the loader's recursive glob finds the PNGs.)
# Sanity (the loader will also print "Indexing ..." / "Sourced full ..."):
find . -name 'CLEVRTEX_full_*.png' ! -name '*_flat.png' | head -3
find . -name 'CLEVRTEX_full_*_flat.png' | wc -l    # ~50k masks expected
```
**Disk:** 5 tarballs (~38GB) + extracted PNGs. You have 98GB free — fine, but `rm` each
`*.tar.gz` after a successful extract if you want headroom (`download_clevrtex.sh:28` does this;
omit `-c`/keep tarballs only if your link is flaky). The eval transform is plain `ToTensor`
(`clevr_tex.py:588`); train uses simclr crop+colorjitter pairs (`clevr_tex.py:565-575`,
`augs.py:27-35`); both resize to 128 with 0.8 center-crop (`clevr_tex.py:29,196-198`).

### 2.3 Data-ready gate
Do not launch training until:
- `data/tetrominoes/tetrominoes_train.npz` + `tetrominoes_test.npz` validate (§2.1 Option C check).
- `data/clevr_tex/clevrtex_full/...` contains both `CLEVRTEX_full_*.png` and `*_flat.png`
  (mask count ≈ image count); a 1-batch dry load prints `Sourced full (train) from ...`
  without `DatasetReadError` (`clevr_tex.py:86-90,139`).

---

## 3. TRAIN_OBJ.PY ARGUMENT REFERENCE (defaults read from source)

All from `train_obj.py` argparse. **Defaults in brackets.**
- `--exp_name` (str) → writes to `runs/<exp_name>/` (`train_obj.py:201`).
- `--model` [`akorn`] — `akorn` or `vit` (vit T>1 = ItrSA baseline) (`:118,285,311`).
- `--data` [`clevrtex`] — **must be `clevrtex_full`** for the decisive run; `tetrominoes` for
  smoke. (Note: bare `clevrtex` is **not** a valid key in `load_data.py`; use `clevrtex_full`.)
- `--data_root` [None→dataset default] (`:98-103`, defaults in `load_data.py`).
- `--data_imsize` [None→dataset default: 32 tetr / 128 clevrtex] (`:106-111`, `load_data.py:25,99`).
- `--batchsize` [**256**] (`:104`). This is the **GLOBAL** batch; per-process =
  `batchsize // num_processes` (`train_obj.py:224`). **Effective images/step = batchsize** (and
  the loader view doubles it to 2× for the positive pair, `train_obj.py:241`,249).
- `--num_workers` [8] (`:105`).
- `--epochs` [**500**] (`:77`). `--checkpoint_every` [50] (`:79`). `--lr` [1e-3] (`:84`,
  Adam wd=0, `:337`). `--warmup_iters` [0] (`:85`). `--beta` [0.998] EMA (`:76,348`).
- `--seed` [1234] (`:75`).
- `--L` [1] num layers (`:119`). `--ch` [256] channels (`:120`). `--psize` [8] patch (`:132`).
  `--ksize` [1] conv kernel for J=conv (`:133`). `--T` [8] recurrent steps (`:134`).
- `--J` [`conv`] connectivity: `conv` or `attn` (`:155`; klayer `:94-110`). **Decisive run uses
  `attn`.** Severed run uses `severed` (§6).
- `--N` [4] rotating dims (`:153`). `--gamma` [1.0] step size (`:154`, fixed non-grad param
  `knet.py:81`). `--heads` [8] (`:139`). `--gta` [True] (`:142`).
- `--c_norm` [`gn`] — `gn`|`sandb`|`none` (`:158-163`; klayer `:112-119`). **synths recipe uses
  `none`** for tetrominoes/clevr; replicate for clevrtex L=1 (see §4/§5 commands).
- `--use_omega` [False] (`:156`), `--global_omg` [False] (`:157`), `--init_omg` [0.01] (`:165`),
  `--learn_omg` [False] (`:167`), `--use_ro_x` [False] (`:168`), `--maxpool` [True] (`:135`),
  `--autorescale` [False] (`:131`), `--model_imsize` [None] (`:121`).
- **Ablation flags that must stay at defaults for a HONEST severance** (`:175-184`):
  `--no_ro` [False] (keep readout), `--project` [True] (keep tangent projection). The severance
  must touch **only** the coupling `connectivity`, NOT `project`/`c_norm`/readout (§6).

**Loss / training shape:** SimCLR on two augmented views, temp `--temp` [0.1] (`:115`,251-256,
`simclr` `:43-66`). Single-GPU: just `python train_obj.py ...` or `accelerate launch
--num_processes=1`; the multi-GPU `--multi-gpu` flag is omitted for 1 GPU (`scripts/synths.md:11`).

---

## 4. STEP 1 — TETROMINOES SMOKE (end-to-end plumbing, ~hours not days)

Recipe mirrors `scripts/synths.md:17` (AKOrN attn tetrominoes) exactly: `ch=128, psize=4, L=1,
J=attn, epochs=50, c_norm=none`. At 32px this is tiny; full global batch 256 fits on 24GB easily.

```bash
cd "$AKORN_HOME"
export DS=tetrominoes
accelerate launch --num_processes=1 train_obj.py \
  --exp_name=${DS}_akorn_attn_smoke --model=akorn --data=tetrominoes \
  --J=attn --L=1 --ch=128 --psize=4 --T=8 --epochs=50 --c_norm=none \
  --batchsize=256 --num_workers=8 --seed=1234
# checkpoints + model.pth/ema_model.pth land in runs/tetrominoes_akorn_attn_smoke/ (train_obj.py:201,377-382)
```
**Wall-clock:** order of a few hours on one 4090 (32px, 50 epochs, ~50k imgs). **Smoke-pass =
≥85 FG-ARI / ≥54 MBO** on the test split (§7). A pass proves: data→loader→augs→model→loss→
checkpoint→eval all wired. It is NOT a scientific result (§0.1).

**Optional severance smoke (do this once `native_severance.py` exists, before the multi-day
CLEVRTex severed run):** identical command with `--J=severed`. Expect param count to match
the attn smoke within 0 (verify with §6.3), and FG-ARI to land at the same ~85+ — this de-risks
the severance plumbing cheaply.

---

## 5. STEP 2 — CLEVRTex-full L=1 REPRODUCE (the decisive, multi-day run)

### 5.1 Recipe (param-faithful to the paper's AKOrN^attn L=1)
`ch=256, psize=8, L=1, J=attn, T=8, gta=True, c_norm=none`. (CLEVR in `synths.md:21` uses
`epochs=300`; the task specifies **epochs=500** — the train_obj default `:77` — for CLEVRTex.
Use 500.) **Target ≈ 75.79 FG-ARI / 54.94 MBO.**

### 5.2 OOM-SAFE batch size for 24GB (reasoning, then numbers)
- The default `--batchsize` in `train_obj.py` is **256** (`:104`), and the loader **doubles** it
  to 512 forward images per step (positive pairs, `train_obj.py:241,249`).
- Cost drivers at 128px, psize=8 → **16×16 = 256 tokens**, ch=256, **T=8 unrolled** attention
  with **GTA** (per-step q,k,v rep-multiply, `common_layers.py:356-384`, `gta.py:57-63`), and the
  recurrence keeps `xs` for all T steps (`klayer.py:159-164`) plus `es`. SimCLR also needs the
  full batch of output embeddings in the loss (`train_obj.py:251`). T=8 unrolled self-attention
  on 256 tokens is the memory hog; activations scale ~linearly in forward batch.
- 512 forward images on 24GB will **OOM**. Be frugal (MEMORY note: pool-before-accumulate, cap
  probe size). **Plan: per-step micro-batch small, recover effective batch 256 via gradient
  accumulation.**

**Problem:** `train_obj.py` has **no `--grad_accum` flag** and calls `opt.step()` every iteration
(`:258-262`); `accelerator.backward` is plain (`:259`). Two honest options:

- **(Preferred) Add gradient accumulation via Accelerate without changing the science.**
  Launch with `accelerate launch --gradient_accumulation_steps=K` AND wrap the train step in
  `with accelerator.accumulate(net):` — but the synced `train_obj.py` does **not** wrap it, so
  the flag alone is a no-op. This requires a **one-line patch** to `train_obj.py` (wrap lines
  244-262 in `with accelerator.accumulate(net):`). Document the patch in the run log; it does
  not alter the model, only the optimizer cadence. With `--batchsize=64` (→128 fwd imgs/step)
  and `K=4`, effective global batch = 256, matching the paper.
- **(No-patch fallback) Lower the global batch and accept it for BOTH arms.** Set
  `--batchsize=64` (128 fwd imgs/step, no accumulation). Effective batch 64 < 256, so absolute
  FG-ARI may sit a touch low vs the paper, BUT the **comparison is internally valid** because the
  severed arm uses the *identical* batch/epochs/seed. Only the repro-vs-paper target check (§8)
  is affected — if 64 underperforms the 1-2pt gate, escalate to the accumulation patch.

**Recommended concrete config (24GB):**
```
--batchsize=64   →  per-process 64, ×2 views = 128 forward imgs/step
+ gradient_accumulation_steps=4 (with the accumulate() wrap)  →  effective global batch 256
```
Start even smaller (`--batchsize=32`, K=8) on the first launch to confirm peak VRAM <24GB via
`nvidia-smi`, then raise. If you must avoid the patch, run `--batchsize=64` and note the reduced
effective batch in the log.

### 5.3 Commands
**(A) With accumulation patch (effective batch 256 — recommended):**
```bash
cd "$AKORN_HOME"
# one-time: wrap the per-iter block of train() in `with accelerator.accumulate(net):`
#   (train_obj.py lines ~244-262). Commit this patch; it changes optimizer cadence only.
accelerate launch --num_processes=1 --gradient_accumulation_steps=4 train_obj.py \
  --exp_name=clevrtex_akorn_attn_L1_repro --model=akorn --data=clevrtex_full \
  --J=attn --L=1 --ch=256 --psize=8 --T=8 --gta=True --c_norm=none \
  --batchsize=64 --epochs=500 --lr=1e-3 --num_workers=8 --seed=1234 \
  --checkpoint_every=50
```
**(B) No-patch fallback (effective batch 64):**
```bash
accelerate launch --num_processes=1 train_obj.py \
  --exp_name=clevrtex_akorn_attn_L1_repro_bs64 --model=akorn --data=clevrtex_full \
  --J=attn --L=1 --ch=256 --psize=8 --T=8 --gta=True --c_norm=none \
  --batchsize=64 --epochs=500 --lr=1e-3 --num_workers=8 --seed=1234 --checkpoint_every=50
```
Run **detached** (this is multi-day) and poll `runs/<exp>/log.txt` + `nvidia-smi`:
```bash
nohup accelerate launch ... > runs/clevrtex_akorn_attn_L1_repro/stdout.log 2>&1 &
```
Checkpoints every 50 epochs → `runs/<exp>/checkpoint_*.pt`, `ema_*.pt`; final
`model.pth`/`ema_model.pth` (`train_obj.py:367-382`). You can eval intermediate checkpoints to
watch the curve.

### 5.4 Wall-clock (be honest)
CLEVRTex-full ≈ **50k images**, 128px, T=8 unrolled attention, **500 epochs**, **on a single
4090**: this is a **multi-day** run — realistically **3–6 days** depending on the
micro-batch/accumulation you can fit and dataloader throughput. The accumulation patch lengthens
per-epoch wall-clock (K forward/backward per step) but is needed to hit effective batch 256.
**Plan for ~a week of GPU occupancy for the repro arm ALONE, then again for the severed arm**
(§6) — i.e. **~1.5–2 weeks total** for the AKOrN^attn vs severed pair at n=1 seed. Budget seeds
accordingly; do NOT promise n=12 here without a time/throughput plan.

---

## 6. STEP 3 — COUPLING-SEVERED RUN (identical budget)

### 6.1 What "coupling" is, exactly (so we sever the RIGHT thing)
The Kuramoto update is `x ← normalize(x + gamma * dxdt)` with
`dxdt = omg_x + reshape_back(Proj_x(J x + c))` (`klayer.py:126-150,152-165`). The **coupling**
is `J = self.connectivity` (`klayer.py:94-108`) — the ONLY cross-token mixing in the block:
- `J=attn` → `Attention(ch, heads=8, weight="conv", kernel_size=1, gta=True, hw=[H,W])`
  (`klayer.py:97-107`). Cross-token mixing = the softmax attention over the 256 tokens.
- `c` (stimulus) is added *after* J (`klayer.py:128-130`), normalized by `c_norm`
  (`klayer.py:155`); `project()` (`klayer.py:121-124`) and `normalize()` (`kutils.py:31-35`) are
  the geometry. **Severance must hold `c`, `c_norm`, `project()`, `normalize()`, `omg`, `gamma`,
  readout (`no_ro=False`), and `--project=True` ALL FIXED** — touch only `connectivity`.

### 6.2 Param-match spec (THE #1 KILL-RISK — get this exact)
Replace `connectivity` with a **per-token map: zero cross-token interaction, identical parameter
count** to the attn coupling. "Per-token" = acts independently on each of the 256 tokens (a 1×1
conv / MLP-mixer-on-channels), so the only thing removed is **token-to-token coupling**, while
capacity (params, nonlinearity budget) is preserved.

**Exact param budget to match (J=attn, ch=256, heads=8, psize=8, imsize=128 → H=W=16):**
From `common_layers.py:256-329`:
- `W_qkv = Conv2d(ch, 3ch, 1)` → `3·ch·ch + 3·ch` = 3·256·256 + 768 = **196,608 + 768**
- `W_o   = Conv2d(ch, ch, 1)`  → `ch·ch + ch`   = 65,536 + 256 = **65,536 + 256**
- GTA mats `mat_q/k/v/o`, each `[H·W, head_dim/2, 2, 2]` (`common_layers.py:318-325`,
  head_dim = ch/heads = 32, head_dim/2 = 16): each = 256·16·2·2 = **16,384**; ×4 = **65,536**
- **Total coupling params (attn) = 196,608+768 + 65,536+256 + 65,536 = 328,704.**

The severed `connectivity` must expose `forward(x)->same-shape` and total **328,704** trainable
params with **no token mixing**. A clean construction:
- a per-token channel MLP: `Conv2d(ch, h, 1)` + nonlinearity + `Conv2d(h, ch, 1)`, with `h`
  chosen so `(ch·h + h) + (h·ch + ch) ≈ 328,704 − (matched GTA-replacement params)`. To keep it
  *simple and exactly matched*, the cleanest approach is to **keep the same Attention module but
  replace the token-mixing op with identity-per-token** — i.e. retain `W_qkv`, `W_o`, and the
  four `mat_*` parameters (so param count is **identical by construction, 328,704**), and replace
  `scaled_dot_product_attention(q,k,v)` (`common_layers.py:372-374`) with a per-token op that
  does **not** mix tokens (e.g. return `v` directly, or apply a learned per-token gate). This
  guarantees param-exactness because it reuses the exact same `nn.Parameter` set and only changes
  the *functional* token-mixing, which is precisely the causal variable under test.

**This "freeze the parameter set, kill only the cross-token softmax" construction is the
defensible severance** — a reviewer cannot say "you removed capacity," because every weight tensor
still exists and is trained; only the `q·kᵀ`-softmax-over-tokens information route is severed.
`native_severance.py` should implement exactly this and register `J="severed"` in `KLayer`
(`klayer.py:94-110`) so the train command is a one-token swap.

### 6.3 Param-match AUDIT (run before trusting any severed number)
```bash
cd "$AKORN_HOME"
python - <<'PY'
import torch
from source.models.objs.knet import AKOrN
common = dict(ch=256, L=1, T=8, psize=8, imsize=128, c_norm='none', gta=True, heads=8)
full = AKOrN(4, J='attn',     **common)
sev  = AKOrN(4, J='severed',  **common)   # requires native_severance.py registered
pf = sum(p.numel() for p in full.parameters() if p.requires_grad)
ps = sum(p.numel() for p in sev.parameters()  if p.requires_grad)
print("AKOrN^attn  params:", pf)
print("AKOrN severed params:", ps)
print("DELTA:", pf-ps, "  MATCH" if pf==ps else "  *** MISMATCH — FIX BEFORE RUNNING ***")
# also confirm the coupling sub-module matches in isolation:
fc = sum(p.numel() for n,p in full.named_parameters() if 'connectivity' in n)
sc = sum(p.numel() for n,p in sev.named_parameters()  if 'connectivity' in n)
print("connectivity params  attn:", fc, " severed:", sc, "(expect 328704 / 328704 for ch=256)")
PY
```
**Do not launch the severed run unless DELTA == 0 and connectivity counts equal 328,704.**

### 6.4 Severed train command (identical budget to §5.3)
```bash
cd "$AKORN_HOME"
accelerate launch --num_processes=1 --gradient_accumulation_steps=4 train_obj.py \
  --exp_name=clevrtex_akorn_severed_L1 --model=akorn --data=clevrtex_full \
  --J=severed --L=1 --ch=256 --psize=8 --T=8 --gta=True --c_norm=none \
  --batchsize=64 --epochs=500 --lr=1e-3 --num_workers=8 --seed=1234 --checkpoint_every=50
# EVERYTHING except --J and --exp_name is byte-identical to the repro command (5.3A).
# Use the SAME --batchsize / accumulation / epochs / seed as whichever repro variant (A or B) you ran.
```

---

## 7. STEP 4 — EVAL (eval_obj.py)

**Run from `$AKORN_HOME`** (eval_obj.py is there, not in /tmp — §0.2). Eval reads the **test**
split: tetrominoes `tetrominoes_test.npz`; clevrtex test = index slice [0,0.1) of `clevrtex_full`
(`clevr_tex.py:30`). Pattern from `scripts/synths.md:42-46`; pass the **same model hparams** used
in training so the architecture reconstructs, and `--model_imsize` = dataset imsize.

**Tetrominoes smoke (32px):**
```bash
cd "$AKORN_HOME"
python eval_obj.py --model=akorn --data=tetrominoes \
  --model_path=runs/tetrominoes_akorn_attn_smoke \
  --model_imsize=32 --J=attn --L=1 --T=8 --ch=128 --psize=4 --c_norm=none
# expect FG-ARI ≥85, MBO ≥54  (smoke-pass; see §0.1)
```

**CLEVRTex repro (128px):**
```bash
python eval_obj.py --model=akorn --data=clevrtex_full \
  --model_path=runs/clevrtex_akorn_attn_L1_repro \
  --model_imsize=128 --J=attn --L=1 --T=8 --ch=256 --psize=8 --c_norm=none
# TARGET ~75.79 FG-ARI / ~54.94 MBO
```

**CLEVRTex severed (128px) — only after §8 gate passes:**
```bash
python eval_obj.py --model=akorn --data=clevrtex_full \
  --model_path=runs/clevrtex_akorn_severed_L1 \
  --model_imsize=128 --J=severed --L=1 --T=8 --ch=256 --psize=8 --c_norm=none
```
Metrics computed by `source/evals/objs/fgari.py:calc_fgari_score` (FG-ARI: drops bg=0 and
ignore=-1, `fgari.py:21-25`) and `mbo.py:calc_mean_best_overlap` (MBO, `mbo.py:67-89`). If your
`eval_obj.py` flags differ from `synths.md:46` (e.g. needs `--data_root`, `--eval_split`, or
`--use_ema`), inspect its argparse first: `python eval_obj.py -h`. Prefer evaluating the **EMA**
weights (`ema_model.pth`) if the script supports it — the paper's numbers are EMA.

---

## 8. THE REPRODUCE-BEFORE-ABLATE GATE (hard stop)

**Do NOT train or eval the severed model until the AKOrN^attn repro lands within ~1–2pt of
target on CLEVRTex-full.**

Gate condition (CLEVRTex-full, test split, EMA weights):
- **PASS:** repro FG-ARI ∈ [≈73.8, ≈77.8] (75.79 ± ~2) **and** MBO ≈ 54.94 ± ~2.
  → proceed to §6 severed run.
- **MARGINAL** (within ~3–4pt, batch-64 fallback): note it; the *internal* comparison is still
  valid (same budget both arms), but flag in the writeup that absolute repro was below target due
  to batch reduction, and prefer the accumulation patch (§5.2A) for the headline number.
- **FAIL** (>4pt low, or collapsed/NaN): **stop.** Debug repro first — check data integrity
  (§2.3), c_norm=none, gta=True, EMA used at eval, effective batch, and that you trained enough
  epochs. A severed-vs-broken-repro comparison is worthless and reviewer-fatal.

**Logic:** the whole claim is "severing coupling leaves accuracy unchanged." That is only
credible if the *un-severed* model first hits the published number. A null on a model that never
reproduced is a reproduction failure, not a finding (§0.1).

---

## 9. RUN ORDER (checklist)
1. §1.3 install (`sympy`!) → §1.4 pre-flight verify (incl. `eval_obj.py` exists at `$AKORN_HOME`).
2. §2.1 Tetrominoes npz ready → §2.3 validate.
3. §4 Tetrominoes SMOKE train → §7 eval → smoke-pass ≥85/≥54.
4. (when `native_severance.py` exists) §6.3 PARAM-MATCH AUDIT (DELTA==0) → optional §4 severed smoke.
5. §2.2 CLEVRTex-full download (~38GB) → §2.3 validate.
6. §5 CLEVRTex AKOrN^attn L=1 REPRO (multi-day) → §7 eval.
7. **§8 GATE** — repro within 1–2pt? If no, stop & debug.
8. §6.4 CLEVRTex SEVERED (identical budget) → §7 eval.
9. Compare FG-ARI + MBO (attn vs severed). Inert ⇒ Δ ≈ 0 within noise.

## 10. KNOWN GOTCHAS
- **`sympy` import** (`kutils.py:1`) silently breaks all model imports if missing — install it.
- **`--data=clevrtex` is invalid**; the train default is `clevrtex` (`train_obj.py:97`) but
  `load_data.py:6` only accepts `clevrtex_full`/`clevrtex_camo`/`clevrtex_outd`. Always pass
  `--data=clevrtex_full`.
- **`clevrtex_camo`/`clevrtex_outd` are eval-only** (`load_data.py:9-13` assert `is_eval`); not
  needed for Gate A.
- **`eval_obj.py` and `native_severance.py` are not in `/tmp/akorn_src`** (§0.2) — run eval from
  `$AKORN_HOME`; author severance separately.
- **Effective batch:** `batchsize` is global and the loader doubles it for pairs
  (`train_obj.py:241`). Keep both arms identical.
- **EMA at eval:** paper numbers are EMA; eval `ema_model.pth` if the script allows.
- **A100 confusion:** `gpu_info.txt` says A100-80GB; task says 4090-24GB. Size for 24GB.
