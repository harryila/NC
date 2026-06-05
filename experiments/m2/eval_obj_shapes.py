"""Minimal fgari (foreground-ARI binding) eval for R6 (project=True) vs R5 (project=False) object-discovery
models on Shapes. Builds the object AKOrN with the EXACT train_obj defaults, loads the raw model.pth state_dict,
and reuses eval_obj.eval_dataset (clustering + fgari) with the Shapes branch we patched in.

Run on box:  cd /root/NC/external/akorn && python3 /root/NC/experiments/m2/eval_obj_shapes.py \
                 --r6 runs/R6_obj/model.pth --r5 runs/R5_obj/model.pth
"""
import argparse
import os
import sys

os.chdir("/root/NC/external/akorn")
sys.path.insert(0, "/root/NC/external/akorn")

import numpy as np
import torch
from source.models.objs.knet import AKOrN
import eval_obj


def build(project, L=1):
    # exact train_obj.py defaults for Shapes (imsize=40, ch=256, T=8, psize=8, N=4, heads=8, gta=True)
    return AKOrN(4, ch=256, L=L, T=8, J="conv", use_omega=False, global_omg=False, c_norm="gn",
                 psize=8, imsize=40, autorescale=False, maxpool=True, project=project, heads=8,
                 use_ro_x=False, no_ro=False, gta=True).cuda()


def run_arm(arm, project, path, pca, L=1):
    net = build(project, L=L)
    sd = torch.load(path, map_location="cuda", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[{arm}] load: missing={len(missing)} unexpected={len(unexpected)} (first miss {missing[:2]})", flush=True)
    net.eval()
    scores, _ = eval_obj.eval_dataset(net, data="Shapes", imsize=40, batchsize=100, instance=True,
                                      method="kmeans", saccade_r=1, pca=pca)
    fg = np.concatenate([s["fgari"] for s in scores], 0)
    print(f"[{arm}] fgari mean={fg.mean():.4f} std={fg.std():.4f} n={len(fg)}", flush=True)
    return float(fg.mean()), float(fg.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r6", default="runs/R6_obj/model.pth")
    ap.add_argument("--r5", default="runs/R5_obj/model.pth")
    ap.add_argument("--pca", default="false")
    ap.add_argument("--tag", default="seed1234")
    ap.add_argument("--L", type=int, default=1)
    a = ap.parse_args()
    pca = a.pca.lower() == "true"
    r6 = run_arm("R6", True, a.r6, pca, a.L)
    r5 = run_arm("R5", False, a.r5, pca, a.L)
    print(f"=== OBJECT-DISCOVERY fgari ({a.tag}) ===")
    print(f"R6(project=True)={r6[0]:.4f}  R5(project=False)={r5[0]:.4f}  DELTA(R6-R5)={r6[0]-r5[0]:+.4f}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
