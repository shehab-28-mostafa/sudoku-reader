import itertools
import math

import numpy as np
import pytest

from sudoku.search import INF, CorrectionProblem, brute_force, search

PUZ = [int(c) for c in
       "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOL = [int(c) for c in
       "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


def make_proba(seed=0, conf=0.97, misreads=()):
    """Probabilities for PUZ: truth with `conf`; `misreads` = {cell: (wrong, p_wrong, p_true)}."""
    rng = np.random.default_rng(seed)
    P = np.full((81, 9), (1 - conf) / 8)
    blank = np.array([v == 0 for v in PUZ])
    for i, v in enumerate(PUZ):
        if v:
            P[i, v - 1] = conf
            noise = rng.dirichlet(np.ones(9) * 2) * (1 - conf)
            P[i] = np.where(np.arange(9) == v - 1, conf, noise * 8 / 8)
            P[i] /= P[i].sum()
    for i, (w, pw, pt) in dict(misreads).items():
        rest = (1 - pw - pt) / 7
        P[i] = rest
        P[i, w - 1], P[i, PUZ[i] - 1] = pw, pt
    P[blank] = 1 / 9
    return P, blank


# ---------- known-answer example ----------
def test_known_answer_single_misread():
    P, blank = make_proba(misreads={0: (3, 0.52, 0.45)})   # 5 read as 3 (dup with r1c2)
    for strat in ("ucs", "greedy", "astar"):
        r = search(CorrectionProblem(P, blank), strat)
        assert r.status == "solved"
        assert [(c, o, n) for c, o, n, _ in r.corrections] == [(0, 3, 5)]
        assert r.cost == pytest.approx(math.log(0.52 / 0.45))
        assert r.csp.solution == SOL


def test_two_misreads_known_answer():
    P, blank = make_proba(misreads={0: (3, 0.52, 0.45), 8 * 9 + 8: (1, 0.55, 0.40)})
    r = search(CorrectionProblem(P, blank), "ucs")
    assert r.status == "solved" and r.csp.solution == SOL
    assert {c for c, *_ in r.corrections} == {0, 80}
    assert r.cost == pytest.approx(math.log(.52 / .45) + math.log(.55 / .40))


# ---------- optimality vs brute force & heuristic admissibility ----------
@pytest.mark.parametrize("seed", range(6))
def test_ucs_and_astar_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    givens = [i for i, v in enumerate(PUZ) if v]
    cells = rng.choice(givens, size=2 + seed % 2, replace=False)
    mis = {}
    for c in cells:
        w = int(rng.choice([d for d in range(1, 10) if d != PUZ[c]]))
        mis[int(c)] = (w, 0.5, 0.35)
    P, blank = make_proba(seed, misreads=mis)
    mk = lambda: CorrectionProblem(P, blank, max_corrections=3, n_uncertain=8, k_alt=3)
    best, _ = brute_force(mk())
    u = search(mk(), "ucs")
    a = search(mk(), "astar")
    assert u.status == a.status == "solved"
    assert u.cost == pytest.approx(best) and a.cost == pytest.approx(best)
    assert a.nodes_expanded <= u.nodes_expanded


@pytest.mark.parametrize("seed", range(4))
def test_heuristic_never_overestimates(seed):
    rng = np.random.default_rng(seed + 100)
    givens = [i for i, v in enumerate(PUZ) if v]
    mis = {int(c): (int(rng.choice([d for d in range(1, 10) if d != PUZ[c]])), 0.5, 0.35)
           for c in rng.choice(givens, 2, replace=False)}
    P, blank = make_proba(seed, misreads=mis)
    prob = CorrectionProblem(P, blank, n_uncertain=8, k_alt=2)
    # true remaining cost from a state = min over goal supersets (brute force)
    cands = [(c, d) for c in prob.candidates for d, _ in prob.alt[c]]
    def true_rest(state):
        best = INF
        for k in range(0, prob.max_corr - len(state) + 1):
            for extra in itertools.combinations(cands, k):
                cells = [c for c, _ in state] + [c for c, _ in extra]
                if len(set(cells)) < len(cells) or any(c <= state[-1][0] for c, _ in extra if state):
                    continue
                s = tuple(sorted(state + extra))
                if prob.is_goal(s)[0]:
                    best = min(best, sum(prob._cost(c, d) for c, d in extra))
        return best
    checked = 0
    for state in [()] + [((c, d),) for c, d in cands[:6]]:
        assert prob.h(state) <= true_rest(state) + 1e-9
        checked += 1
    assert checked == 7


# ---------- edge cases ----------
def test_already_solvable_reading_needs_no_correction():
    P, blank = make_proba()
    r = search(CorrectionProblem(P, blank), "ucs")
    assert r.status == "solved" and r.corrections == [] and r.cost == 0


def test_node_counters_populated_and_limit():
    P, blank = make_proba(misreads={0: (3, 0.52, 0.45)})
    r = search(CorrectionProblem(P, blank), "ucs")
    assert r.nodes_generated >= r.nodes_expanded >= 1 and r.goal_tests >= 1
    r2 = search(CorrectionProblem(P, blank), "ucs", max_expansions=0)
    assert r2.status == "limit"


def test_unique_goal_mode():
    P, blank = make_proba(misreads={0: (3, 0.52, 0.45)})
    r = search(CorrectionProblem(P, blank, goal="unique"), "ucs")
    assert r.status == "solved" and r.csp.n_solutions == 1


# ---------- impossible ----------
def test_impossible_no_corrections_allowed_reports_unreadable():
    P, blank = make_proba(misreads={0: (3, 0.52, 0.45)})
    r = search(CorrectionProblem(P, blank, max_corrections=0), "ucs")
    assert r.status == "unreadable" and r.least_confident and "no puzzle reachable" in r.reason


def test_impossible_too_many_conflicts_for_budget():
    mis = {0: (3, 0.52, 0.45), 4: (6, 0.52, 0.45), 20 + 1: (1, 0.52, 0.45)}
    P, blank = make_proba(misreads=mis)
    for strat in ("ucs", "astar"):
        r = search(CorrectionProblem(P, blank, max_corrections=1, n_uncertain=10), strat)
        assert r.status == "unreadable"
