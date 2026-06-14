#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn
GA=/root/NC/experiments/gateA; LOG=$GA/gate_orchestration.log
until [ -f runs/clvtex_normclamp/ema_499.pth ]; do sleep 120; done
sleep 30
echo "[$(date +%H:%M)] A3 ckpt present; measuring R_global on normclamp..." >> $LOG
python $GA/native_decompose.py --ckpt runs/clvtex_normclamp/ema_499.pth --src /root/NC/external/akorn \
  --data clevrtex_full --data_root /root/data/clevrtex/clevrtex_full --norm_ablate clamp \
  --n_images 64 --bs 8 --n_clusters 11 --out $GA/native_decompose_normclamp.json >> $LOG 2>&1
echo "[$(date +%H:%M)] A3 R_global DONE -> native_decompose_normclamp.json" >> $LOG
