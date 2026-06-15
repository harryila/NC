#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn CUDA_VISIBLE_DEVICES=""
python /root/NC/experiments/gateA/native_usefulness_controls.py --n_scenes 500 --bs 8 --device cpu \
  --out /root/NC/experiments/gateA/results/native_usefulness_controls.json > /root/NC/experiments/gateA/controls2.log 2>&1
echo "CONTROLS2 DONE" >> /root/NC/experiments/gateA/controls2.log
