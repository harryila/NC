"""Fast, dependency-free generator for AKOrN's *Shapes* object-discovery dataset (matches the repo's
create_shapes.py output format so source/data/datasets/objs/shapes.py loads it). 2-4 filled shapes from
{triangle, square, circle, diamond} on a random-bg 40x40 canvas. Saves train/val/test .npz with keys:
  images (N,1,40,40) float32 in [0,1] ; labels (N,40,40) int8 instance (0=bg, k=object, -1=overlap)
  ; pixelwise_class_labels (N,40,40) int8 (0=bg, 1-4=shape-type, -1=overlap).

Usage (on box, from external/akorn so it writes ./data/Shapes/):
  python gen_shapes.py --out ./data/Shapes --ntrain 10000 --nval 1000 --ntest 1000
"""
import argparse
import os
import numpy as np

H = 18


def _templates():
    yy, xx = np.mgrid[0:H, 0:H]
    cx = cy = (H - 1) / 2.0
    square = np.zeros((H, H), np.float32); square[3:H - 3, 3:H - 3] = 1.0
    circle = (((yy - cy) ** 2 + (xx - cx) ** 2) <= (H * 0.42) ** 2).astype(np.float32)
    diamond = ((np.abs(yy - cy) + np.abs(xx - cx)) <= H * 0.46).astype(np.float32)
    triangle = np.zeros((H, H), np.float32)
    for i in range(H):
        half = (i / (H - 1)) * (H / 2.0)
        lo = int(round(cx - half)); hi = int(round(cx + half))
        triangle[i, max(0, lo):min(H, hi + 1)] = 1.0
    return [triangle, square, circle, diamond]   # ids 1..4


def gen(n, seed, canvas=40, min_s=2, max_s=4):
    rng = np.random.default_rng(seed)
    T = _templates()
    imgs = np.zeros((n, 1, canvas, canvas), np.float32)
    inst = np.zeros((n, canvas, canvas), np.int8)
    cls = np.zeros((n, canvas, canvas), np.int8)
    for k in range(n):
        bg = float(rng.uniform(0.1, 0.6))
        img = np.full((canvas, canvas), bg, np.float32)
        im = np.zeros((canvas, canvas), np.int8)
        cm = np.zeros((canvas, canvas), np.int8)
        ns = int(rng.integers(min_s, max_s + 1))
        for iid in range(1, ns + 1):
            sid = int(rng.integers(1, 5))
            m = T[sid - 1]
            px = int(rng.integers(0, canvas - H + 1)); py = int(rng.integers(0, canvas - H + 1))
            reg_i = im[px:px + H, py:py + H]
            reg_c = cm[px:px + H, py:py + H]
            sp = m > 0
            overlap = sp & (reg_i != 0)
            new = sp & (reg_i == 0)
            reg_i[new] = iid; reg_c[new] = sid
            reg_i[overlap] = -1; reg_c[overlap] = -1
            sub = img[px:px + H, py:py + H]; sub[sp] = 1.0
        imgs[k, 0] = img; inst[k] = im; cls[k] = cm
    return imgs, inst, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data/Shapes")
    ap.add_argument("--ntrain", type=int, default=10000)
    ap.add_argument("--nval", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split, n, sd in [("train", a.ntrain, 1), ("val", a.nval, 2), ("test", a.ntest, 3)]:
        imgs, inst, cls = gen(n, seed=sd)
        np.savez_compressed(os.path.join(a.out, f"{split}.npz"),
                            images=imgs, labels=inst, pixelwise_class_labels=cls)
        print(f"wrote {split}: images{imgs.shape} labels{inst.shape} -> {a.out}/{split}.npz", flush=True)
    print("SHAPES_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
