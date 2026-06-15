#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
GA=/root/NC/experiments/gateA
python $GA/native_usefulness_controls.py --n_scenes 500 --bs 8 --device cpu --out $GA/results/native_usefulness_controls.json > $GA/controls3.log 2>&1
echo "CONTROLS3 DONE" >> $GA/controls3.log
python $GA/analyze_usefulness.py > $GA/analyze2.log 2>&1
echo "ANALYZE2 DONE" >> $GA/analyze2.log
