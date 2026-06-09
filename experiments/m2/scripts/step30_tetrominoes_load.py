"""STEP 30 — load + INSPECT the real DeepMind Tetrominoes TFRecords (sanity-check before trusting). Uses the
official deepmind feature spec. --inspect: parse N records, print shape/color/visibility distributions + save
sample images (so we can confirm they're real tetrominoes + design a binding-classification task honestly).
--build: convert N images + per-object (shape,color,position) attrs to numpy for the CL task (after inspect).
"""
import sys, os, argparse, numpy as np
import tensorflow as tf

IMAGE_SIZE = [35, 35]; MAX_ENT = 4
def _features():
    return {
        'image': tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
        'mask': tf.io.FixedLenFeature([MAX_ENT] + IMAGE_SIZE + [1], tf.string),
        'x': tf.io.FixedLenFeature([MAX_ENT, 1], tf.float32),
        'y': tf.io.FixedLenFeature([MAX_ENT, 1], tf.float32),
        'shape': tf.io.FixedLenFeature([MAX_ENT, 1], tf.float32),
        'color': tf.io.FixedLenFeature([MAX_ENT, 3], tf.float32),
        'visibility': tf.io.FixedLenFeature([MAX_ENT, 1], tf.float32),
    }
def _decode(proto):
    ex = tf.io.parse_single_example(proto, _features())
    ex['image'] = tf.squeeze(tf.io.decode_raw(ex['image'], tf.uint8), axis=-1)
    return ex

def load(path, n):
    ds = tf.data.TFRecordDataset(path, compression_type='GZIP').map(_decode).take(n)
    imgs, shapes, colors, vis = [], [], [], []
    for ex in ds:
        imgs.append(ex['image'].numpy()); shapes.append(ex['shape'].numpy().reshape(-1))
        colors.append(ex['color'].numpy()); vis.append(ex['visibility'].numpy().reshape(-1))
    return np.array(imgs), np.array(shapes), np.array(colors), np.array(vis)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/root/tetrominoes_train.tfrecords")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    imgs, shapes, colors, vis = load(a.path, a.n)
    print(f"loaded {len(imgs)} images, image shape {imgs.shape[1:]}, dtype {imgs.dtype}, range [{imgs.min()},{imgs.max()}]")
    print(f"entities/example: {vis.shape[1]} | visibility sum per img (mean) = {vis.sum(1).mean():.2f} (n objects)")
    uniq_shapes = np.unique(np.round(shapes[vis>0.5], 3))
    print(f"SHAPE values (visible objs): {len(uniq_shapes)} unique -> {uniq_shapes[:20]}")
    cv = colors[vis>0.5]
    print(f"COLOR (visible objs) RGB ranges: R[{cv[:,0].min():.2f},{cv[:,0].max():.2f}] G[{cv[:,1].min():.2f},{cv[:,1].max():.2f}] B[{cv[:,2].min():.2f},{cv[:,2].max():.2f}]")
    # distinct colors? round and count
    uc = np.unique(np.round(cv, 2), axis=0)
    print(f"COLOR distinct (rounded 0.01): {len(uc)} -> sample {uc[:8].tolist()}")
    if a.inspect:
        # save a small grid of sample images for visual sanity-check
        try:
            from PIL import Image
            grid = np.concatenate([np.concatenate([imgs[r*6+c] for c in range(6)],1) for r in range(4)],0)
            Image.fromarray(grid).save("/root/NC/experiments/m2/tetrominoes_samples.png")
            print("saved sample grid -> experiments/m2/tetrominoes_samples.png")
        except Exception as e: print("img save skipped:", e)
    print("INSPECT_DONE")
