"""Search over digit corrections.

State  : tuple of (cell, new_digit) pairs, sorted by cell (max `max_corrections`).
Action : change one *uncertain* cell (not yet corrected, index larger than every
         corrected cell so each set is generated exactly once) to one of its top
         alternative digits.
Cost   : sum over corrections of max(0, log p(read) - log p(new))   (nats, >= 0).
Goal   : the corrected grid passes AC-3 and the CSP finds a solution
         ("solvable"), or exactly one solution ("unique", extension).
Strategies: ucs (priority g), greedy (priority h, tie-break g), astar (g + h).

A* heuristic h: every equal-digit conflict in a unit forces at least one of its two
cells to change. Take a greedy set of cell-disjoint conflicts; for each add the
cheapest cost of changing either of its cells. Disjoint cells => costs add up =>
never overestimates the true remaining cost (admissible). If the disjoint
conflicts need more cells than corrections left, h = infinity (state is dead).
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .csp import CSPResult, find_conflicts, solve

INF = math.inf


@dataclass
class SearchResult:
    status: str                         # "solved" | "unreadable" | "limit"
    corrections: list = field(default_factory=list)   # (cell, read, new, cost)
    cost: float = 0.0
    grid: list | None = None
    csp: CSPResult | None = None
    nodes_generated: int = 0
    nodes_expanded: int = 0
    goal_tests: int = 0
    reason: str = ""
    least_confident: list = field(default_factory=list)


class CorrectionProblem:
    def __init__(self, proba: np.ndarray, blank: np.ndarray, max_corrections=3,
                 n_uncertain=20, k_alt=3, goal="solvable", csp_kwargs=None):
        """proba: (81, 9) class probabilities (rows of blank cells are ignored)."""
        assert goal in ("solvable", "unique")
        self.proba, self.blank = proba, blank
        self.max_corr, self.goal = max_corrections, goal
        self.csp_kwargs = csp_kwargs or dict(fc=True, ac3=True, order="mrv")
        self.read = [0 if blank[i] else int(proba[i].argmax()) + 1 for i in range(81)]
        conf = [(-1.0 if blank[i] else float(proba[i].max())) for i in range(81)]
        filled = [i for i in range(81) if not blank[i]]
        self.least_confident = sorted(filled, key=lambda i: conf[i])
        self.candidates = sorted(self.least_confident[:n_uncertain])
        self.alt: dict[int, list[tuple[int, float]]] = {}
        for i in self.candidates:
            order = np.argsort(-proba[i])
            r = self.read[i] - 1
            alts = [(int(d) + 1, self._cost(i, int(d) + 1)) for d in order if d != r][:k_alt]
            self.alt[i] = alts
        self.min_change = {i: min(c for _, c in self.alt[i]) for i in self.candidates}

    def _cost(self, cell, digit) -> float:
        p = self.proba[cell]
        return max(0.0, math.log(max(p[self.read[cell] - 1], 1e-12))
                   - math.log(max(p[digit - 1], 1e-12)))

    # --- problem interface ---
    def initial(self):
        return ()

    def grid(self, state) -> list[int]:
        g = list(self.read)
        for cell, d in state:
            g[cell] = d
        return g

    def g_cost(self, state) -> float:
        return sum(self._cost(c, d) for c, d in state)

    def successors(self, state):
        if len(state) >= self.max_corr:
            return
        last = state[-1][0] if state else -1
        for cell in self.candidates:
            if cell > last:
                for d, _ in self.alt[cell]:
                    yield state + ((cell, d),)

    def h(self, state) -> float:
        g = self.grid(state)
        corrected = {c for c, _ in state}
        used: set[int] = set()
        total, needed = 0.0, 0
        for a, b, _ in find_conflicts(g):
            if a in used or b in used:
                continue
            opts = [self.min_change[x] for x in (a, b)
                    if x in self.min_change and x not in corrected]
            if not opts:
                return INF
            used.update((a, b))
            total += min(opts)
            needed += 1
        if needed > self.max_corr - len(state):
            return INF
        return total

    def is_goal(self, state):
        res = solve(self.grid(state), max_solutions=2 if self.goal == "unique" else 1,
                    **self.csp_kwargs)
        ok = res.n_solutions == 1 if self.goal == "unique" else res.solved
        return ok, res


def search(problem: CorrectionProblem, strategy="ucs", max_expansions=20000,
           trace: list | None = None) -> SearchResult:
    assert strategy in ("ucs", "greedy", "astar")
    out = SearchResult("unreadable")
    counter = itertools.count()

    def prio(state):
        g = problem.g_cost(state)
        if strategy == "ucs":
            return (g, 0.0), g
        h = problem.h(state)
        if strategy == "greedy":
            return (h, g), g
        return (g + h, g), g

    start = problem.initial()
    key, g0 = prio(start)
    frontier = [(key, next(counter), start)]
    out.nodes_generated = 1
    while frontier:
        key, _, state = heapq.heappop(frontier)
        if key[0] == INF:
            continue                      # dead state (heuristic proves no completion)
        out.goal_tests += 1
        ok, res = problem.is_goal(state)
        if trace is not None:
            trace.append(dict(order=out.goal_tests, state=state, g=problem.g_cost(state),
                              priority=key[0], goal=ok))
        if ok:
            out.status, out.csp, out.grid = "solved", res, problem.grid(state)
            out.corrections = [(c, problem.read[c], d, problem._cost(c, d)) for c, d in state]
            out.cost = problem.g_cost(state)
            return out
        if out.nodes_expanded >= max_expansions:
            out.status, out.reason = "limit", f"stopped after {max_expansions} expansions"
            break
        out.nodes_expanded += 1
        for child in problem.successors(state):
            k, _ = prio(child)
            if k[0] < INF:
                out.nodes_generated += 1
                heapq.heappush(frontier, (k, next(counter), child))
    if out.status == "unreadable":
        out.reason = (f"no puzzle reachable with <= {problem.max_corr} corrections of the "
                      f"{len(problem.candidates)} least confident cells")
    out.least_confident = problem.least_confident[:5]
    return out


def brute_force(problem: CorrectionProblem):
    """Exhaustively enumerate every correction set; return (min_cost, state) of goals."""
    best = (INF, None)
    cands = [(c, d) for c in problem.candidates for d, _ in problem.alt[c]]
    for k in range(0, problem.max_corr + 1):
        for combo in itertools.combinations(cands, k):
            cells = [c for c, _ in combo]
            if len(set(cells)) < k:
                continue
            state = tuple(sorted(combo))
            g = problem.g_cost(state)
            if g < best[0] and problem.is_goal(state)[0]:
                best = (g, state)
    return best
