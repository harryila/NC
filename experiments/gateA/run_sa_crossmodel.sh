#!/bin/bash
cd /root/NC/experiments/gateA
export OMP_NUM_THREADS=4
echo "=== SA cross-model: FULL (in-dist) n=2000 ==="
python crossmodel_slotattn.py --data_type full --data_root /root/data/clevrtex/clevrtex_full \
  --n_scenes 2000 --device cpu --out results/crossmodel_sa_full.json 2>&1 | grep -vE "FutureWarning|weights_only|UserWarning|warnings.warn|_VF.meshgrid|torch.load"
echo "=== SA cross-model: OOD (decider) n=2000 ==="
python crossmodel_slotattn.py --data_type outd --data_root /root/data/clevrtex/clevrtex_outd \
  --n_scenes 2000 --device cpu --out results/crossmodel_sa_outd.json 2>&1 | grep -vE "FutureWarning|weights_only|UserWarning|warnings.warn|_VF.meshgrid|torch.load"
echo "=== SA CROSSMODEL ALL DONE ==="
