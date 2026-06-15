#!/bin/bash
cd /root/NC/external/akorn; source /root/NC/venv/bin/activate 2>/dev/null
export PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
python /root/NC/experiments/gateA/native_retrieval.py --data_type outd \
  --data_root /root/data/clevrtex/clevrtex_outd --n_scenes 2000 --bs 8 --device cpu \
  --out /root/NC/experiments/gateA/results/retrieval_outd.json > /root/NC/experiments/gateA/retrieval_ood.log 2>&1
echo "RETRIEVAL_OOD DONE" >> /root/NC/experiments/gateA/retrieval_ood.log
