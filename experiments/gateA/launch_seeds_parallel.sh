#!/bin/bash
# ============================================================================
# PARALLEL n>=3 seed campaign launcher for a NEW multi-GPU / big-GPU box.
# Replaces the serial 4090 campaign (orchestrate_rigor.sh) with parallel jobs.
#
# PREREQS on the new box (do these first):
#   1. git clone the NC repo to $NC (default /root/NC), checkout main.
#   2. ./setup.sh   -> builds external/akorn from the pinned commit (eabbe27) and
#      applies experiments/gateA/akorn_gateA.patch (the GATE-A wiring: --J none /
#      --norm_ablate / --phase_noise in train_obj.py + eval_obj.py).
#      VERIFY: `cd external/akorn && git apply --check ../../experiments/gateA/akorn_gateA.patch`
#      then `git apply ...` if setup.sh didn't already.
#   3. CLEVRTex data at $DR (clevrtex_full) and clevrtex_outd (for retrieval later).
#   4. python venv at $NC/venv with the akorn deps (torch, ema_pytorch, fastcluster, ...).
#   5. (optional) scp the backed-up seed-1234 finals (ckpt_backup_seed1234/ckpts_final.tgz)
#      into external/akorn/runs/ so you DON'T need to re-train seed 1234.
#
# WHAT IT RUNS: {full, jnone(=severed), normclamp, itrsa} x SEEDS, each = train 500ep -> eval (FG-ARI),
#   in parallel across all visible GPUs. Seed 1234 results are already committed (from the 4090), so the
#   default SEEDS=(1 2) gives n=3 with the committed 1234. Add 1234 to SEEDS for a fully self-contained redo.
#
# USAGE:
#   # one 80GB card (A100/H100), 4 trainings at once (~17GB each):
#   SLOTS_PER_GPU=4 ./launch_seeds_parallel.sh
#   # multi-GPU box (auto-detects N cards, one training per card):
#   ./launch_seeds_parallel.sh
#   # fully self-contained (re-train 1234 too) on an 8-GPU box:
#   SEEDS="1234 1 2" ./launch_seeds_parallel.sh
#   # just the headline arms (drop itrsa/normclamp):
#   ARMS="full jnone" ./launch_seeds_parallel.sh
# ============================================================================
set -u
NC=${NC:-/root/NC}; AK=$NC/external/akorn; GA=$NC/experiments/gateA
DR=${DR:-/root/data/clevrtex/clevrtex_full}
SEEDS=(${SEEDS:-1 2})                  # add 1234 for self-contained re-derivation
ARMS=(${ARMS:-full jnone normclamp itrsa})
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}      # set 3-4 on an 80GB card; each train ~17GB

cd "$AK" || { echo "no $AK -- run setup.sh first"; exit 1; }
source "$NC/venv/bin/activate" 2>/dev/null
export PYTHONPATH=$NC:$AK AKORN_SRC=$AK
TC="--data_root=$DR --data=clevrtex_full --L=1 --checkpoint_every=25 --epochs=500"
EC="--data_root=$DR --data=clevrtex_full --L=1 --model_imsize=128"

arm_flags(){ case "$1" in   # train & eval share the same model/connectivity flags
  full)      echo "--model=akorn --J=attn";;
  jnone)     echo "--model=akorn --J=none";;
  normclamp) echo "--model=akorn --J=attn --norm_ablate clamp";;
  itrsa)     echo "--model=vit --gta=False --ch=256 --psize=8 --T=8";;
  *) echo "UNKNOWN_ARM_$1" >&2; return 1;;
esac; }

# build the (arm:seed) job list
JOBS=()
for s in "${SEEDS[@]}"; do for a in "${ARMS[@]}"; do JOBS+=("$a:$s"); done; done

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
NSLOT=$((NGPU * SLOTS_PER_GPU))
echo "[$(date +%H:%M)] GPUs=$NGPU slots_per_gpu=$SLOTS_PER_GPU -> $NSLOT slots | ${#JOBS[@]} jobs: ${JOBS[*]}"

run_job(){                              # $1=arm:seed  $2=gpu_id
  local a=${1%:*} s=${1#*:} g=$2 exp
  exp=clvtex_${a}_s${s}
  if [ -f "runs/$exp/ema_499.pth" ]; then echo "[gpu $g] SKIP $exp (ema_499 exists)"; return; fi
  echo "[$(date +%H:%M)] [gpu $g] TRAIN $exp"
  CUDA_VISIBLE_DEVICES=$g python train_obj.py $TC $(arm_flags "$a") --exp_name=$exp --seed=$s > "$GA/train_${exp}.log" 2>&1
  echo "[$(date +%H:%M)] [gpu $g] EVAL  $exp"
  CUDA_VISIBLE_DEVICES=$g python eval_obj.py  $EC $(arm_flags "$a") --model_path=runs/$exp/ema_499.pth > "$GA/eval_${exp}.log" 2>&1
  echo "[$(date +%H:%M)] [gpu $g] DONE  $exp -> eval_${exp}.log"
}

# dispatch: NSLOT parallel workers; each worker runs its share of jobs SEQUENTIALLY on its assigned GPU
for slot in $(seq 0 $((NSLOT - 1))); do
  gpu=$((slot % NGPU))
  (
    for i in "${!JOBS[@]}"; do
      [ $((i % NSLOT)) -eq "$slot" ] && run_job "${JOBS[$i]}" "$gpu"
    done
  ) &
done
wait
echo "[$(date +%H:%M)] === ALL SEED JOBS DONE ==="
echo "NEXT: per-seed FG-ARI is in eval_*.log; run the per-seed retrieval+probe with native_retrieval.py /"
echo "      t_sweep.py / floors_retrieval.py per seed (cheap, eval-only) for seed-level CIs / TOST."
