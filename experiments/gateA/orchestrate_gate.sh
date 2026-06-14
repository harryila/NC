#!/bin/bash
# Box-side autonomous GATE-A orchestration (survives ssh drops). Logs each step.
cd /root/NC/external/akorn
source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn
export AKORN_SRC=/root/NC/external/akorn
RUN=runs/clvtex_akorn_attn_repro
GA=/root/NC/experiments/gateA
LOG=$GA/gate_orchestration.log
echo "[$(date +%H:%M)] orchestration started; waiting for reproduction (ema_499.pth)" > $LOG

# 1) wait for reproduction to finish
until [ -f $RUN/ema_499.pth ]; do sleep 120; done
echo "[$(date +%H:%M)] STEP1 reproduction DONE (ema_499.pth present)" >> $LOG

# 2) converged mechanistic predictor read (larger probe), on the full model
echo "[$(date +%H:%M)] STEP2 running native_decompose on ema_499 (n=64)..." >> $LOG
python $GA/native_decompose.py --ckpt $RUN/ema_499.pth --src /root/NC/external/akorn \
  --data clevrtex_full --data_root /root/data/clevrtex/clevrtex_full \
  --n_images 64 --bs 8 --n_clusters 11 --device cuda \
  --out $GA/native_decompose_ema499.json >> $LOG 2>&1
echo "[$(date +%H:%M)] STEP2 predictor DONE -> native_decompose_ema499.json" >> $LOG

# 3) definitive full eval of the reproduction (FG-ARI/MBO vs 75.6 target)
echo "[$(date +%H:%M)] STEP3 full eval_obj on ema_499 (this can take a while)..." >> $LOG
python eval_obj.py --data_root=/root/data/clevrtex/clevrtex_full --model=akorn --data=clevrtex_full \
  --J=attn --L=1 --model_path=$RUN/ema_499.pth --model_imsize=128 > $GA/eval_repro_full.log 2>&1
echo "[$(date +%H:%M)] STEP3 full eval DONE -> eval_repro_full.log" >> $LOG

# 4) launch the severance retrain (param-matched, --J none, identical budget)
echo "[$(date +%H:%M)] STEP4 launching severance retrain (--J none)..." >> $LOG
nohup python train_obj.py --exp_name=clvtex_severed_none --data_root=/root/data/clevrtex/clevrtex_full \
  --model=akorn --data=clevrtex_full --J=none --L=1 --checkpoint_every=25 --seed=1234 \
  > $GA/train_clvtex_severed_none.log 2>&1 &
echo "[$(date +%H:%M)] STEP4 severance retrain launched pid $!" >> $LOG
echo "[$(date +%H:%M)] ORCHESTRATION: repro+eval done, severance retraining (~9h). Check eval_repro_full.log + train_clvtex_severed_none.log" >> $LOG
