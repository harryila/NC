"""
Calibrate epochs/task: train R6 on ONE task and watch where val-accuracy plateaus.
Set epochs/task = plateau + small margin for the WHOLE matrix (epochs must be identical
across all arms — it's a controlled variable). This makes the plasticity guard honest and
is the single biggest cost lever (AKOrN's 400 is its from-scratch CIFAR-10 recipe, not a
CL-per-task recipe). A "forgets less" result that's really "learned each task less" is the
#1 NeurIPS reviewer kill — so do NOT guess this number.

Run:  python calibrate_epochs.py --rung R6 --max_epochs 120
Pick the epoch where task0_val_acc flattens; use it as --epochs for run_matrix.py.
"""
import argparse
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
from avalanche_backbone import LadderClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="R6")
    ap.add_argument("--max_epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from avalanche.benchmarks.classic import SplitCIFAR100
    bench = SplitCIFAR100(n_experiences=10, seed=0)
    exp0, test0 = bench.train_stream[0], bench.test_stream[0]

    model = LadderClassifier(a.rung, num_classes=100).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    tl = torch.utils.data.DataLoader(exp0.dataset, batch_size=128, shuffle=True, num_workers=4)
    vl = torch.utils.data.DataLoader(test0.dataset, batch_size=256, num_workers=4)

    print(f"calibrating {a.rung} on task-0 (10 classes), up to {a.max_epochs} epochs")
    for ep in range(1, a.max_epochs + 1):
        model.train()
        for xb, yb, *_ in tl:
            xb, yb = xb.to(a.device), yb.to(a.device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb, *_ in vl:
                xb, yb = xb.to(a.device), yb.to(a.device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.numel()
        print(f"epoch {ep:3d}  task0_val_acc {100*correct/total:6.2f}")


if __name__ == "__main__":
    main()
