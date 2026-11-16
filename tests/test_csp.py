import pytest

from sudoku.csp import PEERS, ac3_detect, find_conflicts, solve

PUZZLE = [int(c) for c in
          "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOLUTION = [int(c) for c in
            "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]
CONFIGS = [dict(), dict(fc=True), dict(ac3=True), dict(fc=True, ac3=True), dict(mac=True),
           dict(fc=True, order="mrv")]


def valid(sol):
    from sudoku.csp import UNITS
    return all(sorted(sol[i] for i in u) == list(range(1, 10)) for u in UNITS)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_all_configs_solve_known_puzzle(cfg):
    r = solve(PUZZLE, **cfg)
    assert r.solved and r.solution == SOLUTION and valid(r.solution)


def test_peers_structure():
    assert all(len(p) == 20 for p in PEERS)


def test_fc_reduces_work_and_ac3_helps_more():
    bt, fc, ac = solve(PUZZLE), solve(PUZZLE, fc=True), solve(PUZZLE, fc=True, ac3=True)
    assert fc.assignments < bt.assignments
    assert ac.assignments <= fc.assignments
    assert bt.assignments > 0 and bt.backtracks > 0 and ac.ac3_revisions > 0


def test_conflict_pairs_not_duplicated():
    g = PUZZLE[:]
    g[2] = 5
    assert len(find_conflicts(g)) == 1


def test_empty_grid_is_solvable():
    r = solve([0] * 81, fc=True, order="mrv")
    assert r.solved and valid(r.solution)


def test_full_valid_grid_edge():
    r = solve(SOLUTION)
    assert r.solved and r.assignments == 0


def test_uniqueness_detection_edge():
    r = solve(PUZZLE, fc=True, ac3=True, max_solutions=2)
    assert r.n_solutions == 1
    g = PUZZLE[:]
    g[0] = 0  # still unique; clear a whole box -> many solutions
    for i in (0, 1, 2, 9, 10, 11, 18, 19, 20):
        g[i] = 0
    r2 = solve(g, fc=True, ac3=True, max_solutions=2, order="mrv")
    assert r2.n_solutions == 2


def test_limit_edge():
    r = solve([0] * 81, max_assignments=5)
    assert r.status == "limit"


@pytest.mark.parametrize("cfg", [dict(), dict(fc=True, ac3=True)])
def test_impossible_conflicting_givens(cfg):
    g = PUZZLE[:]
    g[2] = 5
    r = solve(g, **cfg)
    assert r.status == "no_solution" and "conflicting givens" in r.reason
    ok, reason, _ = ac3_detect(g)
    assert not ok and "conflicting" in reason


def test_impossible_without_given_conflict():
    # cell r1c1 sees 1..8 among givens in row/col and 9 elsewhere in box -> empty domain
    g = [0] * 81
    for k, c in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        g[c] = k + 1 if c < 8 else 0
    g[1:8] = [1, 2, 3, 4, 5, 6, 7]
    g[9] = 8
    g[10] = 9  # 9 in box of r1c1; r1c1 now needs 8 or 9 (both used) -> impossible
    r = solve(g, fc=True, ac3=True)
    assert r.status == "no_solution"
