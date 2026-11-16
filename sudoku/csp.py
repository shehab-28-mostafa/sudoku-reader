"""Sudoku as a CSP: 81 variables, domains 1-9, all-different on rows/cols/boxes.

Three separately switchable techniques (with counters):
  * Backtracking (BT)           - always on
  * Forward Checking (fc=True)  - remove an assigned digit from peers' domains
  * AC-3 (ac3=True)             - arc consistency before search (shrinks domains,
                                  detects conflicts such as two equal givens in a
                                  unit); mac=True also maintains AC-3 during search.
Grids are length-81 lists, 0 = blank. Domains are 9-bit masks (bit v-1 = digit v).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

ALL = 0b111111111


def _build():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
             for br in range(3) for bc in range(3)]
    peers = []
    for i in range(81):
        r, c = divmod(i, 9)
        s = set(rows[r]) | set(cols[c]) | set(boxes[(r // 3) * 3 + c // 3])
        s.discard(i)
        peers.append(tuple(sorted(s)))
    return rows, cols, boxes, tuple(peers)


ROWS, COLS, BOXES, PEERS = _build()
UNITS = ROWS + COLS + BOXES


def cell_name(i: int) -> str:
    return f"r{i // 9 + 1}c{i % 9 + 1}"


def find_conflicts(grid) -> list[tuple[int, int, int]]:
    """Pairs (cell_a, cell_b, digit) of equal non-blank cells sharing a unit."""
    out, found = [], set()
    for unit in UNITS:
        seen: dict[int, int] = {}
        for i in unit:
            v = grid[i]
            if v:
                if v in seen:
                    if (seen[v], i) not in found:
                        found.add((seen[v], i))
                        out.append((seen[v], i, v))
                else:
                    seen[v] = i
    return out


@dataclass
class CSPResult:
    status: str                       # "solved" | "no_solution" | "limit"
    solution: list[int] | None = None
    n_solutions: int = 0
    assignments: int = 0
    backtracks: int = 0
    ac3_revisions: int = 0
    pruned: int = 0                   # values removed from domains (FC + AC-3)
    reason: str = ""
    conflicts: list = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    @property
    def unique(self) -> bool:
        return self.n_solutions == 1


class _Limit(Exception):
    pass


class _Stop(Exception):
    pass


class SudokuCSP:
    def __init__(self, fc=False, ac3=False, mac=False, order="static",
                 max_assignments=None, max_solutions=1):
        assert order in ("static", "mrv")
        self.fc, self.ac3, self.mac = fc or mac, ac3 or mac, mac
        self.order, self.max_assignments, self.max_solutions = order, max_assignments, max_solutions

    # ------------------------------------------------------------------
    def solve(self, grid) -> CSPResult:
        self.res = CSPResult("no_solution")
        self.val = list(grid)
        self.solutions: list[list[int]] = []
        conf = find_conflicts(grid)
        if conf:
            a, b, v = conf[0]
            self.res.conflicts = conf
            self.res.reason = (f"conflicting givens: {cell_name(a)} and {cell_name(b)} "
                               f"both contain {v}")
            return self.res

        self.use_dom = self.fc or self.ac3
        self.trail: list[tuple[int, int]] = []
        if self.use_dom:
            self.dom = [(1 << (v - 1)) if v else ALL for v in self.val]
            if self.fc:  # forward-check the givens themselves (not counted as assignments)
                for i, v in enumerate(self.val):
                    if v and not self._forward_check(i, v):
                        return self._finish("a given empties a neighbour's domain")
            if self.ac3 and not self._ac3(deque((i, j) for i in range(81) for j in PEERS[i])):
                return self._finish("AC-3 emptied a domain (puzzle inconsistent)")
        try:
            self._search()
        except _Limit:
            self.res.status = "limit"
            self.res.reason = f"stopped after {self.max_assignments} assignments"
            return self.res
        except _Stop:
            pass
        return self._finish("search exhausted - no consistent completion")

    def _finish(self, why: str) -> CSPResult:
        r = self.res
        r.n_solutions = len(self.solutions)
        if self.solutions:
            r.status, r.solution, r.reason = "solved", self.solutions[0], ""
        else:
            r.status, r.reason = "no_solution", r.reason or why
        return r

    # ------------------------------------------------------------------
    def _legal(self, c: int) -> int:
        if self.use_dom:
            return self.dom[c]
        used = 0
        for p in PEERS[c]:
            if self.val[p]:
                used |= 1 << (self.val[p] - 1)
        return ALL & ~used

    def _pick(self):
        empties = [i for i in range(81) if not self.val[i]]
        if not empties:
            return None
        if self.order == "static":
            return empties[0]
        return min(empties, key=lambda i: self._legal(i).bit_count())

    def _search(self):
        c = self._pick()
        if c is None:
            self.solutions.append(list(self.val))
            if len(self.solutions) >= self.max_solutions:
                raise _Stop
            return
        mask = self._legal(c)
        for v in range(1, 10):
            bit = 1 << (v - 1)
            if not mask & bit:
                continue
            self.res.assignments += 1
            if self.max_assignments and self.res.assignments > self.max_assignments:
                raise _Limit
            mark = len(self.trail)
            self.val[c] = v
            ok = True
            if self.use_dom:
                self._set(c, bit)
                if self.mac:
                    ok = self._ac3(deque((p, c) for p in PEERS[c]))
                elif self.fc:
                    ok = self._forward_check(c, v)
            if ok:
                self._search()
            self.val[c] = 0
            self._undo(mark)
            self.res.backtracks += 1

    # ---- domain helpers with trail ----
    def _set(self, c, mask):
        if self.dom[c] != mask:
            self.trail.append((c, self.dom[c]))
            self.dom[c] = mask

    def _undo(self, mark):
        while len(self.trail) > mark:
            c, m = self.trail.pop()
            self.dom[c] = m

    def _forward_check(self, c, v) -> bool:
        bit = 1 << (v - 1)
        ok = True
        for p in PEERS[c]:
            if not self.val[p] and self.dom[p] & bit:
                self._set(p, self.dom[p] & ~bit)
                self.res.pruned += 1
                if not self.dom[p]:
                    ok = False  # wipe-out; keep pruning cheap: stop early
                    break
        return ok

    def _ac3(self, queue: deque) -> bool:
        """Standard AC-3 for the binary not-equal constraints."""
        dom = self.dom
        while queue:
            xi, xj = queue.popleft()
            self.res.ac3_revisions += 1
            dj = dom[xj]
            if dj & (dj - 1) == 0 and dom[xi] & dj:  # xj fixed -> remove its value from xi
                self._set(xi, dom[xi] & ~dj)
                self.res.pruned += 1
                if not dom[xi]:
                    return False
                queue.extend((xk, xi) for xk in PEERS[xi] if xk != xj)
        return True


def solve(grid, fc=False, ac3=False, mac=False, order="static",
          max_assignments=None, max_solutions=1) -> CSPResult:
    return SudokuCSP(fc, ac3, mac, order, max_assignments, max_solutions).solve(grid)


def ac3_detect(grid):
    """Run AC-3 alone; return (consistent, reason, conflict_pairs)."""
    conf = find_conflicts(grid)
    s = SudokuCSP(ac3=True)
    s.res = CSPResult("no_solution")
    s.use_dom, s.fc, s.trail = True, False, []
    s.val = list(grid)
    s.dom = [(1 << (v - 1)) if v else ALL for v in grid]
    ok = s._ac3(deque((i, j) for i in range(81) for j in PEERS[i]))
    reason = ""
    if not ok:
        reason = (f"conflicting givens ({len(conf)} pair(s))" if conf
                  else "AC-3 emptied a domain")
    return ok, reason, conf
