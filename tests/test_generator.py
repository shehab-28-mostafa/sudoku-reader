import numpy as np
import pytest

from sudoku.csp import solve
from sudoku.generator import DigitPool, blank_mask, full_grid, make_instance, make_puzzle, to_mosaic


@pytest.mark.parametrize("givens", [25, 30, 35])
def test_puzzle_has_unique_solution_and_requested_givens(givens):
    rng = np.random.default_rng(3)
    puz, sol = make_puzzle(rng, givens)
    assert sum(map(bool, puz)) == givens
    r = solve(puz, fc=True, ac3=True, order="mrv", max_solutions=2)
    assert r.n_solutions == 1 and r.solution == sol


def test_seed_reproducible():
    a = make_puzzle(np.random.default_rng(7), 30)
    b = make_puzzle(np.random.default_rng(7), 30)
    assert a == b
    assert make_puzzle(np.random.default_rng(8), 30) != a


def test_full_grid_valid():
    g = full_grid(np.random.default_rng(0))
    assert solve(g).solved and 0 not in g


def test_instance_injection_and_blank_rule():
    rng = np.random.default_rng(0)
    # synthetic pool: digit d image = constant-ish pattern; fake probabilities
    n = 400
    y = np.repeat(np.arange(1, 10), n // 9 + 1)[:n]
    x = np.zeros((n, 784), np.float32) + 0.2
    proba = np.full((n, 9), 0.01)
    proba[np.arange(n), y - 1] = 0.9
    proba[::5, :] = 0.01
    proba[::5, 0] = 0.5            # these are "wrong" reads unless y == 1
    proba[::5, (y[::5] - 1)] = 0.3  # true digit stays in top-3
    pool = DigitPool(x, y, proba)
    inst = make_instance(rng, pool, 30, 2)
    assert len(inst["injected"]) == 2
    assert (blank_mask(inst["images"]) == (np.array(inst["puzzle"]) == 0)).all()
    assert to_mosaic(inst["images"]).shape == (252, 252)
