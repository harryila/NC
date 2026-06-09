"""STEP 24 — VERIFICATION PROBES for audit risks #1 (binding) and #3 (trunk leak). Cheap, no CL.
PART A (trunk leak, risk #3): linear probe on trunk(x) features -> class, for RANDOM trunk vs CIFAR-AE learned
trunk, on TIGHT shapes. Random should be ~chance (task-agnostic = unbypassable); learned elevated = leak.
PART B (binding, risk #1): does a PRESENCE-DETECTOR solve the task? Build a per-shape-type presence feature
(max spatial correlation with each of the 4 shape templates) -> linear probe -> class. If high acc, the task is
solvable by TYPE-PRESENCE with NO binding/segregation -> 'binding' framing overstated.
"""
import sys, os, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import m2_shapes_construct as shp
shp.TIGHT = True
import m2_hypernet as H

device = "cuda" if torch.cuda.is_available() else "cpu"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# data
X, y = shp._gen_split(300, seed=0)   # (3000,3,32,32), labels 0-9
Xt = torch.tensor(X, dtype=torch.float32)

# ---- PART A: trunk-feature linear probe (random vs learned/CIFAR-AE) ----
def random_trunk(seed=0):
    return H._build_pieces("R6", device, seed)[1]   # frozen-random conv -> 64
def cifar_ae_trunk(seed=0):
    import importlib.util
    spec = importlib.util.spec_from_file_location("s23", "/tmp/step23_transfer_trunk.py")
    m = importlib.util.module_from_spec(spec)
    # step23 sets shp.TIGHT and patches H._build_pieces on import; we only want its _pretrain_trunk
    import m2_shapes_construct as _shp; _shp.TIGHT = True
    spec.loader.exec_module(m)
    return m._pretrain_trunk(seed, device)

def trunk_probe(trunk):
    feats = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            feats.append(trunk(Xt[i:i+256].to(device)).cpu().numpy())
    F = np.concatenate(feats)
    return float(cross_val_score(LogisticRegression(max_iter=2000), F, y, cv=5).mean())

print("=== PART A: trunk-feature -> class linear probe (chance 0.10) ===")
rt = trunk_probe(random_trunk(0))
print(f"  RANDOM trunk probe acc = {rt:.3f}  (should be ~0.10 = task-agnostic/unbypassable)")
try:
    lt = trunk_probe(cifar_ae_trunk(0))
    print(f"  CIFAR-AE learned trunk probe acc = {lt:.3f}  (if >> 0.10 -> trunk leaks task info -> unbypass broken under learned trunks)")
except Exception as e:
    print("  (learned trunk probe failed:", e, ")")

# ---- PART B: presence-detector (does type-presence solve the task?) ----
print("=== PART B: presence-detector -> class (chance 0.10) ===")
templates = shp._shape_templates() if hasattr(shp, "_shape_templates") else None
# build templates from the source (triangle/square/circle/diamond) at 28x28 like the construct
import numpy as _np
def _templates():
    # reproduce shp shapes at H=28 (the construct's H)
    H_ = 28; yy, xx = _np.mgrid[0:H_,0:H_]; cx=cy=(H_-1)/2.0
    square=_np.zeros((H_,H_),_np.float32); square[3:H_-3,3:H_-3]=1.0
    circle=(((yy-cy)**2+(xx-cx)**2)<=(H_*0.42)**2).astype(_np.float32)
    diamond=((_np.abs(yy-cy)+_np.abs(xx-cx))<=H_*0.46).astype(_np.float32)
    tri=_np.zeros((H_,H_),_np.float32)
    for i in range(H_):
        half=(i/(H_-1))*(H_/2.0); lo=int(round(cx-half)); hi=int(round(cx+half)); tri[i,max(0,lo):min(H_,hi+1)]=1.0
    return [tri,square,circle,diamond]
T = _templates()
# presence feature: for each template, max normalized cross-correlation over the (single-channel) image
import torch.nn.functional as Fnn
imgs = Xt[:, 0]   # (N,32,32), grayscale
pres = np.zeros((len(imgs), 4), np.float32)
with torch.no_grad():
    for k, t in enumerate(T):
        tt = torch.tensor(t)[None,None]; tt = Fnn.interpolate(tt, size=(8,8), mode='area')  # small template
        tt = (tt - tt.mean())
        resp = Fnn.conv2d(imgs[:,None], tt, padding=2)  # (N,1,..)
        pres[:, k] = resp.amax(dim=(1,2,3)).numpy()
acc_pres = float(cross_val_score(LogisticRegression(max_iter=2000), pres, y, cv=5).mean())
print(f"  presence-feature (4-dim, type-presence only, NO binding) probe acc = {acc_pres:.3f}")
print(f"  => if >> chance (0.10) and high, the task is PRESENCE-SOLVABLE -> 'binding required' is OVERSTATED")
print("STEP24_DONE")
