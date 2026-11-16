"""Seeded generators: full grids, unique-solution puzzles, 9x9 image mosaics and
injected hard (misread-prone) digits.

Documented rule for injected misreads (see docs/data_dictionary.md):
  * Pool "wrong[d]": test images of true digit d that the trained NN misclassifies
    but whose true digit is still within its top-3 predictions.
  * Pool "right[d]": test images of digit d that the NN classifies correctly.
  * For a puzzle with m misreads, m random givens receive an image from wrong[d];
    all other givens receive an image from right[d]. Blank cells are dark noise.
"""
from __future__ import annotations

import numpy as np

from .csp import solve

BLANK_SIGMA = 0.01
BLANK_MEAN_THRESHOLD = 0.04   # brightness rule used to detect blank cells


def full_grid(rng: np.random.Generator) -> list[int]:
    """Random complete grid via randomised backtracking with MRV."""
    from .csp import PEERS
    val = [0] * 81

    def legal(c):
        used = {val[p] for p in PEERS[c] if val[p]}
        return [v for v in range(1, 10) if v not in used]

    def rec():
        empties = [i for i in range(81) if not val[i]]
        if not empties:
            return True
        c = min(empties, key=lambda i: len(legal(i)))
        opts = legal(c)
        rng.shuffle(opts)
        for v in opts:
            val[c] = v
            if rec():
                return True
        val[c] = 0
        return False

    assert rec()
    return val


def _carve(rng, n_givens):
    sol = full_grid(rng)
    puz = sol[:]
    for c in rng.permutation(81):
        if sum(1 for v in puz if v) <= n_givens:
            break
        keep = puz[c]
        puz[c] = 0
        r = solve(puz, fc=True, ac3=True, order="mrv", max_solutions=2)
        if r.n_solutions != 1:
            puz[c] = keep
    return puz, sol


def make_puzzle(rng: np.random.Generator, n_givens: int = 30, attempts: int = 50):
    """Remove cells (random order) while the solution stays unique, retrying with
    fresh grids until exactly `n_givens` givens remain (or `attempts` is used up,
    in which case the puzzle with the fewest givens is returned).
    Returns (puzzle, solution)."""
    best = None
    for _ in range(attempts):
        puz, sol = _carve(rng, n_givens)
        k = sum(1 for v in puz if v)
        if best is None or k < best[2]:
            best = (puz, sol, k)
        if k == n_givens:
            break
    return best[0], best[1]


class DigitPool:
    """Test-set images grouped by digit into correctly / wrongly read sets."""

    def __init__(self, x, y, proba):
        pred = proba.argmax(1) + 1
        top3 = np.argsort(-proba, axis=1)[:, :3] + 1
        in_top3 = (top3 == y[:, None]).any(1)
        self.x = x
        self.right = {d: np.where((y == d) & (pred == d))[0] for d in range(1, 10)}
        self.wrong = {d: np.where((y == d) & (pred != d) & in_top3)[0] for d in range(1, 10)}


def make_instance(rng, pool: DigitPool, n_givens=30, n_misreads=0, puzzle=None):
    """Return dict with truth puzzle/solution, image array (81,784) and injected cells."""
    puz, sol = puzzle if puzzle is not None else make_puzzle(rng, n_givens)
    givens = [i for i, v in enumerate(puz) if v]
    candidates = [i for i in givens if len(pool.wrong[puz[i]]) > 0]
    m = min(n_misreads, len(candidates))
    injected = sorted(rng.choice(candidates, size=m, replace=False).tolist()) if m else []
    imgs = np.clip(rng.normal(0, BLANK_SIGMA, (81, 784)), 0, 1).astype(np.float32)
    for i in givens:
        d = puz[i]
        src = pool.wrong[d] if i in injected else pool.right[d]
        imgs[i] = pool.x[rng.choice(src)]
    return {"puzzle": puz, "solution": sol, "images": imgs, "injected": injected}


def to_mosaic(images: np.ndarray) -> np.ndarray:
    """(81,784) -> (252,252) mosaic for display."""
    t = images.reshape(9, 9, 28, 28)
    return t.transpose(0, 2, 1, 3).reshape(252, 252)


def blank_mask(images: np.ndarray) -> np.ndarray:
    """Simple brightness rule (NOT the NN): a cell is blank if its mean pixel is low."""
    return images.mean(axis=1) < BLANK_MEAN_THRESHOLD
