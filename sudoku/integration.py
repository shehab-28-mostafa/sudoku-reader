"""End-to-end pipeline: NN reads the grid -> AC-3 looks for conflicts -> Search tries
the cheapest corrections -> CSP tests each corrected puzzle; first solvable wins.

Modes (for experiment 3):
  "nn"     NN-guided correction (UCS/Greedy/A* with probability costs)
  "none"   no correction: solve the reading as-is
  "random" random correction sets (probabilities ignored), same CSP goal test
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .csp import SudokuCSP, ac3_detect, cell_name, solve
from .generator import blank_mask
from .search import CorrectionProblem, SearchResult, search


@dataclass
class PipelineResult:
    status: str                       # "solved" | "unreadable"
    solution: list | None = None
    reading: list = field(default_factory=list)
    blank: np.ndarray | None = None
    proba: np.ndarray | None = None
    ac3_ok: bool = True
    ac3_reason: str = ""
    corrections: list = field(default_factory=list)   # (cell, read, new, cost)
    cost: float = 0.0
    search: SearchResult | None = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    goal_tests: int = 0
    message: str = ""
    least_confident: list = field(default_factory=list)

    def correction_log(self) -> list[str]:
        return [f"{cell_name(c)}: read {o} -> {n} (cost {k:.3f})"
                for c, o, n, k in self.corrections]


def read_grid(images: np.ndarray, model):
    """Brightness rule finds blanks; NN gives class probabilities for the rest."""
    blank = blank_mask(images)
    proba = np.full((81, 9), 1 / 9)
    if (~blank).any():
        proba[~blank] = model.predict_proba(images[~blank])
    reading = [0 if blank[i] else int(proba[i].argmax()) + 1 for i in range(81)]
    return reading, blank, proba


def _random_search(problem: CorrectionProblem, rng, budget):
    """Random corrections among *all* filled cells; ignores NN probabilities."""
    filled = [i for i in range(81) if not problem.blank[i]]
    tests = 0
    ok, res = problem.is_goal(())
    tests += 1
    if ok:
        return (), res, tests
    for _ in range(budget):
        k = int(rng.integers(1, problem.max_corr + 1))
        cells = rng.choice(filled, size=k, replace=False)
        state = []
        for c in sorted(cells):
            choices = [d for d in range(1, 10) if d != problem.read[c]]
            state.append((int(c), int(rng.choice(choices))))
        tests += 1
        ok, res = problem.is_goal(tuple(state))
        if ok:
            return tuple(state), res, tests
    return None, None, tests


def run_pipeline(images, model, mode="nn", strategy="ucs", goal="solvable",
                 max_corrections=3, n_uncertain=20, k_alt=3, max_expansions=3000,
                 rng=None, random_budget=300) -> PipelineResult:
    reading, blank, proba = read_grid(images, model)
    out = PipelineResult("unreadable", reading=reading, blank=blank, proba=proba)
    out.ac3_ok, out.ac3_reason, _ = ac3_detect(reading)
    problem = CorrectionProblem(proba, blank, max_corrections, n_uncertain, k_alt, goal)
    out.least_confident = problem.least_confident[:5]

    if mode == "none":
        ok, res = problem.is_goal(())
        out.goal_tests = 1
        if ok:
            out.status, out.solution = "solved", res.solution
        else:
            out.message = f"unreadable: {out.ac3_reason or 'no solution'} (no correction attempted)"
        return out

    if mode == "random":
        rng = rng or np.random.default_rng(0)
        state, res, tests = _random_search(problem, rng, random_budget)
        out.goal_tests = tests
        if state is not None:
            out.status, out.solution = "solved", res.solution
            out.corrections = [(c, problem.read[c], d, problem._cost(c, d)) for c, d in state]
            out.cost = problem.g_cost(state)
        else:
            out.message = "unreadable: random corrections found no solvable puzzle"
        return out

    sr = search(problem, strategy, max_expansions)
    out.search = sr
    out.nodes_expanded, out.nodes_generated, out.goal_tests = (
        sr.nodes_expanded, sr.nodes_generated, sr.goal_tests)
    if sr.status == "solved":
        out.status, out.solution = "solved", sr.csp.solution
        out.corrections, out.cost = sr.corrections, sr.cost
        out.message = ("solved without corrections" if not sr.corrections
                       else f"solved with {len(sr.corrections)} correction(s)")
    else:
        cells = ", ".join(cell_name(c) for c in sr.least_confident)
        out.message = f"unreadable: {sr.reason}. Least confident cells: {cells}"
    return out
