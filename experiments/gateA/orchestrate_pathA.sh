#!/bin/bash
# PATH A (CORRECTED per design-verification w7c0mv62k): dissociate SYNCHRONIZATION from ROUTING from NORMALIZATION.
cd /root/NC/external/akorn
source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn
GA=/root/NC/experiments/gateA; LOG=$GA/gate_orchestration.log; DR=/root/data/clevrtex/clevrtex_full
until grep -q "SEVERED full eval DONE" $LOG 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] PATH-A start (corrected ladder: A1 projoff / A3 normclamp / A4 itrsa)" >> $LOG

# A1: tangent-PROJECTION off (relabeled; NOT synchrony-off). apply_proj=False, sphere+coupling ON. param-identical.
python train_obj.py --exp_name=clvtex_projoff --data_root=$DR --model=akorn --data=clevrtex_full \
  --J=attn --L=1 --project False --checkpoint_every=25 --seed=1234 > $GA/train_projoff.log 2>&1
python eval_obj.py --data_root=$DR --model=akorn --data=clevrtex_full --J=attn --L=1 --project False \
  --model_path=runs/clvtex_projoff/ema_499.pth --model_imsize=128 > $GA/eval_projoff.log 2>&1
echo "[$(date +%H:%M)] A1 projoff EVAL DONE -> eval_projoff.log" >> $LOG

# A3: sphere-NORMALIZATION ablation (clamp + apply_proj=False), attention ON. param-identical. THE decisive arm.
python train_obj.py --exp_name=clvtex_normclamp --data_root=$DR --model=akorn --data=clevrtex_full \
  --J=attn --L=1 --norm_ablate clamp --checkpoint_every=25 --seed=1234 > $GA/train_normclamp.log 2>&1
python eval_obj.py --data_root=$DR --model=akorn --data=clevrtex_full --J=attn --L=1 --norm_ablate clamp \
  --model_path=runs/clvtex_normclamp/ema_499.pth --model_imsize=128 > $GA/eval_normclamp.log 2>&1
echo "[$(date +%H:%M)] A3 normclamp EVAL DONE -> eval_normclamp.log  (DECISIVE: FG-ARI + R_global)" >> $LOG

# A4: ItrSA floor (config-pinned ch=256 psize=8 T=8, 500ep). External floor, NOT a single-mechanism control.
python train_obj.py --exp_name=clvtex_itrsa --data_root=$DR --model=vit --data=clevrtex_full \
  --L=1 --gta=False --ch=256 --psize=8 --T=8 --epochs=500 --checkpoint_every=25 --seed=1234 > $GA/train_itrsa.log 2>&1
python eval_obj.py --data_root=$DR --model=vit --data=clevrtex_full --L=1 --gta=False --ch=256 --psize=8 --T=8 \
  --model_path=runs/clvtex_itrsa/ema_499.pth --model_imsize=128 > $GA/eval_itrsa.log 2>&1
echo "[$(date +%H:%M)] A4 itrsa EVAL DONE -> eval_itrsa.log" >> $LOG
echo "[$(date +%H:%M)] === PATH-A COMPLETE: full=75.5 / J=none / projoff / normclamp / itrsa ===" >> $LOG
