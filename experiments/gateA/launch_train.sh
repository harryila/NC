#!/bin/bash
# usage: launch_train.sh <exp_name> <J>   (J=attn for full repro, J=none for severed)
EXP=$1; J=$2
cd /root/NC/external/akorn
PYTHONPATH=/root/NC:/root/NC/external/akorn AKORN_SRC=/root/NC/external/akorn \
nohup python train_obj.py --exp_name=$EXP --data_root=/root/data/clevrtex/clevrtex_full \
  --model=akorn --data=clevrtex_full --J=$J --L=1 --checkpoint_every=25 --seed=1234 \
  > /root/NC/experiments/gateA/train_$EXP.log 2>&1 &
echo "launched $EXP (J=$J) pid $!"
