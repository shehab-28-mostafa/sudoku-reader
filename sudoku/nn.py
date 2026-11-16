"""Small NumPy MLP digit classifier (784-128-64-9) with temperature scaling,
plus calibration metrics. Labels are 1..9 externally, 0..8 internally."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(784, 128, 64, 9), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.T = 1.0  # temperature (fitted on validation set)

    # ---- forward ----
    def _forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def logits(self, x):
        return self._forward(x)[-1]

    def predict_proba(self, x, calibrated: bool = True):
        return softmax(self.logits(x), self.T if calibrated else 1.0)

    def predict(self, x):
        return self.logits(x).argmax(1) + 1

    # ---- training (mini-batch SGD + momentum, L2) ----
    def fit(self, x, y, epochs=12, lr=0.05, batch=128, momentum=0.9, wd=1e-4,
            seed=0, x_val=None, y_val=None, verbose=True):
        rng = np.random.default_rng(seed)
        y0 = y - 1
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        n = len(x)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                acts = self._forward(x[bi])
                d = softmax(acts[-1])
                d[np.arange(len(bi)), y0[bi]] -= 1
                d /= len(bi)
                for l in range(len(self.W) - 1, -1, -1):
                    gW = acts[l].T @ d + wd * self.W[l]
                    gb = d.sum(0)
                    if l > 0:
                        d = (d @ self.W[l].T) * (acts[l] > 0)
                    vW[l] = momentum * vW[l] - lr * gW
                    vb[l] = momentum * vb[l] - lr * gb
                    self.W[l] += vW[l]
                    self.b[l] += vb[l]
            if verbose and x_val is not None:
                acc = (self.predict(x_val) == y_val).mean()
                print(f"epoch {ep + 1:2d}/{epochs}  val acc {acc:.4f}")
        return self

    # ---- temperature scaling ----
    def fit_temperature(self, x_val, y_val):
        z = self.logits(x_val)
        y0 = y_val - 1
        best_T, best = 1.0, np.inf
        for T in np.linspace(0.5, 5.0, 91):
            p = softmax(z, T)
            nll = -np.log(p[np.arange(len(y0)), y0] + 1e-12).mean()
            if nll < best:
                best, best_T = nll, T
        self.T = float(best_T)
        return self.T

    # ---- persistence ----
    def save(self, path):
        np.savez(path, T=self.T, **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            m = cls()
            n = len(m.W)
            m.W = [f[f"W{i}"] for i in range(n)]
            m.b = [f[f"b{i}"] for i in range(n)]
            m.T = float(f["T"])
        return m


def expected_calibration_error(proba, y, bins: int = 15) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) + 1 == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def confusion_matrix(pred, y, k: int = 9):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y - 1, pred - 1), 1)
    return cm
