#!/bin/bash
cd /root/NC/external/akorn
source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn
RUN=runs/clvtex_severed_none
GA=/root/NC/experiments/gateA
LOG=$GA/gate_orchestration.log
until [ -f $RUN/ema_499.pth ]; do sleep 120; done
echo "[$(date +%H:%M)] SEVERED reproduction DONE; running full eval..." >> $LOG
python eval_obj.py --data_root=/root/data/clevrtex/clevrtex_full --model=akorn --data=clevrtex_full \
  --J=none --L=1 --model_path=$RUN/ema_499.pth --model_imsize=128 > $GA/eval_severed_full.log 2>&1
echo "[$(date +%H:%M)] SEVERED full eval DONE -> eval_severed_full.log" >> $LOG
echo "[$(date +%H:%M)] === GATE COMPLETE: compare eval_repro_full.log (full) vs eval_severed_full.log (severed) ===" >> $LOG
