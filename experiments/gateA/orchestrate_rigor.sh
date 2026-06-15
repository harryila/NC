#!/bin/bash
# RIGOR CAMPAIGN (path A, sequential on one 4090): trained-desync (decisive) + n>=3 on load-bearing arms.
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn
GA=/root/NC/experiments/gateA; LOG=$GA/gate_orchestration.log; DR=/root/data/clevrtex/clevrtex_full
TC="--data_root=$DR --model=akorn --data=clevrtex_full --L=1 --checkpoint_every=25"
EC="--data_root=$DR --model=akorn --data=clevrtex_full --L=1 --model_imsize=128"
log(){ echo "[$(date +%m-%d_%H:%M)] $1" >> $LOG; }
until grep -q "PATH-A COMPLETE" $LOG 2>/dev/null; do sleep 180; done
log "RIGOR CAMPAIGN start (one 4090, sequential)"

# ===== PHASE D: trained-desync (the NON-CIRCULAR synchrony-necessity test) =====
for sig in 0.5 1.0; do
  tag=pn${sig/./}
  log "PhaseD trained-desync sigma=$sig train..."
  python train_obj.py $TC --exp_name=clvtex_$tag --J=attn --phase_noise $sig --seed=1234 > $GA/train_$tag.log 2>&1
  python eval_obj.py  $EC --J=attn --phase_noise $sig --model_path=runs/clvtex_$tag/ema_499.pth > $GA/eval_${tag}_noisy.log 2>&1
  python eval_obj.py  $EC --J=attn --phase_noise 0   --model_path=runs/clvtex_$tag/ema_499.pth > $GA/eval_${tag}_clean.log 2>&1
  python $GA/native_decompose.py --ckpt runs/clvtex_$tag/ema_499.pth --src /root/NC/external/akorn --data clevrtex_full --data_root $DR --phase_noise $sig --n_images 48 --bs 8 --n_clusters 11 --out $GA/decompose_$tag.json >> $LOG 2>&1
  log "PhaseD sigma=$sig DONE"
done

# ===== PHASE N: n>=3 seeds (1,2) for load-bearing arms (seed 1234 already on disk) =====
for seed in 1 2; do
  python train_obj.py $TC --exp_name=clvtex_full_s$seed --J=attn --seed=$seed > $GA/train_full_s$seed.log 2>&1
  python eval_obj.py  $EC --J=attn --model_path=runs/clvtex_full_s$seed/ema_499.pth > $GA/eval_full_s$seed.log 2>&1
  log "PhaseN full seed=$seed DONE"
  python train_obj.py $TC --exp_name=clvtex_jnone_s$seed --J=none --seed=$seed > $GA/train_jnone_s$seed.log 2>&1
  python eval_obj.py  $EC --J=none --model_path=runs/clvtex_jnone_s$seed/ema_499.pth > $GA/eval_jnone_s$seed.log 2>&1
  log "PhaseN jnone seed=$seed DONE"
  python train_obj.py $TC --exp_name=clvtex_normclamp_s$seed --J=attn --norm_ablate clamp --seed=$seed > $GA/train_normclamp_s$seed.log 2>&1
  python eval_obj.py  $EC --J=attn --norm_ablate clamp --model_path=runs/clvtex_normclamp_s$seed/ema_499.pth > $GA/eval_normclamp_s$seed.log 2>&1
  log "PhaseN normclamp seed=$seed DONE"
done
log "=== RIGOR CAMPAIGN COMPLETE ==="
