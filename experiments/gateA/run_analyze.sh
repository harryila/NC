#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
python /root/NC/experiments/gateA/analyze_usefulness.py > /root/NC/experiments/gateA/analyze.log 2>&1
echo "ANALYZE DONE" >> /root/NC/experiments/gateA/analyze.log
