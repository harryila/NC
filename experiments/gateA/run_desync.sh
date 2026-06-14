#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn CUDA_VISIBLE_DEVICES=""
python /root/NC/experiments/gateA/native_desync_probe.py \
  --ckpt runs/clvtex_akorn_attn_repro/ema_499.pth \
  --data clevrtex_full --data_root /root/data/clevrtex/clevrtex_full \
  --n_images 24 --bs 6 --device cpu \
  --out /root/NC/experiments/gateA/native_desync_probe_full.json > /root/NC/experiments/gateA/desync_full.log 2>&1
echo "DESYNC DONE" >> /root/NC/experiments/gateA/gate_orchestration.log
