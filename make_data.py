"""Download and preprocess MNIST reproducibly.  Usage: python make_data.py [--seed 0]"""
import argparse

import numpy as np

from sudoku.data import save_processed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = save_processed(args.seed)
    for k in ("train", "val", "test"):
        x, y = d[f"x_{k}"], d[f"y_{k}"]
        print(f"{k:5s}: x{tuple(x.shape)} range[{x.min():.1f},{x.max():.1f}] "
              f"labels {y.min()}..{y.max()} counts {np.bincount(y)[1:].tolist()}")
    print("Saved data/mnist19.npz")
