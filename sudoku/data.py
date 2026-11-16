"""MNIST loading and preprocessing (digits 1-9 only)."""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "mnist19.npz"

# Mirror on GitHub (original yann.lecun.com is not always reachable).
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        dest = raw_dir / name
        if not dest.exists():
            urllib.request.urlretrieve(MIRROR + name, dest)
        got = hashlib.md5(dest.read_bytes()).hexdigest()
        if got != md5:
            dest.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}: {got} != {md5}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def preprocess(seed: int = 0, raw_dir: Path = RAW_DIR) -> dict[str, np.ndarray]:
    """Split 50k/10k (train/val) from the official training set, keep the
    official test set untouched, then drop digit 0 everywhere. Pixels in [0,1],
    flattened to 784. Labels are 1..9."""
    x = _read_idx(raw_dir / "train-images-idx3-ubyte.gz")
    y = _read_idx(raw_dir / "train-labels-idx1-ubyte.gz")
    xt = _read_idx(raw_dir / "t10k-images-idx3-ubyte.gz")
    yt = _read_idx(raw_dir / "t10k-labels-idx1-ubyte.gz")
    perm = np.random.default_rng(seed).permutation(len(x))
    tr, va = perm[:50_000], perm[50_000:]

    def prep(images, labels):
        keep = labels != 0
        return (images[keep].reshape(-1, 784).astype(np.float32) / 255.0,
                labels[keep].astype(np.int64))

    out = {}
    for name, (im, lb) in {"train": (x[tr], y[tr]), "val": (x[va], y[va]),
                           "test": (xt, yt)}.items():
        out[f"x_{name}"], out[f"y_{name}"] = prep(im, lb)
    return out


def save_processed(seed: int = 0, path: Path = PROCESSED) -> dict[str, np.ndarray]:
    download()
    d = preprocess(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def load_processed(path: Path = PROCESSED) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python make_data.py` first.")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
