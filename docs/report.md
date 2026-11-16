# Handwritten Sudoku Reader and Solver: combining a neural network, search and a CSP

_Project P7. All numbers come from `results/tables.md` (5 seeds, mean ± std)._

## 1. Problem
A Sudoku is written in handwritten digits. The neural network (NN) may misread a digit, which
can make the puzzle contradictory or change its solution. The system must find the most
likely misreads, correct them, and solve the puzzle. Success is measured by **exact recovery**:
the returned solution equals the true solution. If no corrected puzzle is solvable, the
system reports "unreadable" and lists the least confident cells.

## 2. Design
**Pipeline.** NN reads the grid → AC-3 checks the reading for conflicts → Search tries
correction sets in order of plausibility → CSP tests each corrected puzzle → the first
accepted one wins (`sudoku/integration.py`).

**Data.** MNIST digits 1-9 (45,053 train / 9,024 validation / 9,020 test). Blank cells are
detected by a brightness rule (brightest pixel < 0.25), not by the NN. Puzzles, mosaics and
injected hard digits are generated (see `docs/data_dictionary.md`). This is not photographed
Sudoku.

**NN.** A NumPy MLP (784-128-64-9, ReLU, SGD with momentum, 12 epochs), followed by temperature
scaling (T = 1.45, fitted on validation). Baseline: logistic regression.

**CSP.** 81 variables, domains 1-9, not-equal constraints on rows, columns and boxes. Backtracking,
Forward Checking and AC-3 are independent switches with counters (assignments, backtracks, AC-3
revisions, pruned values). AC-3 runs before search; a "maintain AC-3" option exists.

**Search.** State = set of corrections (≤ 3). Action = change one of the 20 least-confident
cells to one of its 3 best alternative digits; each set is generated once. Cost = max(0,
ln p(read) − ln p(new)), never negative. UCS, Greedy (priority h) and A* (g + h). The heuristic
h: every equal-digit conflict forces at least one of its two cells to change; for a set of
cell-disjoint conflicts add the cheapest change cost of either cell. Disjoint cells make the
costs add up, so h never overestimates (tested against brute force). If the disjoint conflicts
need more cells than corrections remain, h = ∞ and the state is dropped.

## 3. Neural network results (test set)
| model | accuracy | ECE |
|---|---|---|
| MLP (temperature-scaled) | 97.82 % | 0.0033 (0.0089 before scaling) |
| Logistic regression | 92.16 % | 0.0061 |

Confusion matrices and a reliability diagram are in `results/nn_evaluation.png`. The MLP beats
the baseline by 5.7 points. Temperature scaling cut the calibration error by about 63 %.
Only one training seed was run.

## 4. Experiments
### Experiment 1: UCS vs Greedy vs A* (30 givens, 0-4 misreads)
Recovery (UCS, solvable goal) is 100 / 79.8 / 71.2 / 59.0 / 0 % for 0-4 misreads. UCS and A*
always return the same cost, as expected from two optimal algorithms. Greedy returns a
costlier set in 11 % (2 misreads) to 21 % (3 misreads) of instances and recovers fewer exact
solutions (62.2 % vs 71.2 % at 2 misreads).

Nodes expanded at 3 misreads: UCS 48.5, Greedy 22.0, A* 8.4 (values in `tables.md`).
The A* heuristic saves a lot over UCS and, unlike Greedy, keeps optimality. At 4 misreads UCS
expands ~1,400 nodes vs ~40 for A* / Greedy, but no method can recover exactly because 4 errors
exceed the 3-correction budget (recovery 0 %; a solvable puzzle is still found for 55 % of
cases, which is then not the true one).

### Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (static cell order)
| givens | BT | BT+FC | BT+FC+AC-3 |
|---|---|---|---|
| 25 | 52,489 ± 12,957 | 12,844 ± 2,380 | 9,926 ± 1,257 |
| 30 | 4,701 ± 1,009 | 1,391 ± 229 | 718 ± 338 |
| 35 | 520 ± 61 | 181 ± 13 | 68 ± 15 |

(mean assignments ± std over 5 seeds of 50 puzzles.) Each technique cuts work: FC about 3-4×, AC-3
on top another 1.3-2.7×. At 35 givens AC-3 alone nearly solves the puzzle (68 assignments for
~46 empty cells). Six of 2,250 static-order runs hit the 500,000-assignment cap, so the BT numbers
at 25 givens are lower bounds.
**Caveat.** With the MRV ordering the three variants give identical counts. In my implementation
plain BT computes legal values on the fly for MRV, which is effectively forward checking, and
AC-3 on not-equal constraints adds nothing beyond singleton propagation there. The static-order
table is therefore the meaningful comparison.

### Experiment 3: NN-guided correction vs none vs random
| misreads | NN-guided (UCS) | No correction | Random | NN-guided, unique goal (A*) |
|---|---|---|---|---|
| 0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| 1 | 79.8 ± 3.3 | 0.0 | 15.4 ± 2.2 | 93.6 ± 1.5 |
| 2 | 71.2 ± 7.5 | 0.0 | 0.4 ± 0.5 | 89.2 ± 2.3 |
| 3 | 59.0 ± 5.3 | 0.0 | 0.0 | 86.0 ± 2.0 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 |

(exact recovery, %.) NN-guided correction is the only method that works at 2-3 misreads. Random
correction needs ~120-290 CSP calls and mostly fails; NN guidance needs 3-50. The "solvable"
goal's main weakness: at 1 misread, 87 of 101 failures (seed-pooled) accepted the misread grid
without any correction because it was still solvable, with a different solution. Requiring a
**unique** solution fixes most of these and raises recovery to 93.6 / 89.2 / 86.0 %, with fewer CSP
calls because wrong candidates are rejected early.

### Confidence flags (added after the first demo showed a silently wrong answer)
A solved puzzle can be a valid Sudoku that is not the written one (for example when more misreads
exist than corrections allowed). The system therefore attaches a confidence tier to every answer:
**low** if the corrected puzzle has more than one solution; **medium** if a correction was very
unlikely (> 2.5 nats) or the whole correction budget was used; otherwise **high**. Evaluated on
1,000 generated puzzles (5 seeds x 40 per misread level 0-4; `results/confidence.md`):

| tier | answers | % exactly right | share of all wrong answers |
|---|---|---|---|
| high | 471 | 90.2 | 14.6 |
| medium | 248 | 78.6 | 16.8 |
| low | 216 | 0.0 | 68.6 |

85.4 % of wrong answers are flagged (medium or low) at the price of flagging 31.5 % of correct
ones; the "low" tier was never right. The flag is a warning, not a guarantee: 14.6 % of wrong
answers still come out as "high".

## 5. Integration, limits and honesty notes
* The CSP is the final judge. The NN only orders the search; AC-3 only explains why the reading is
  broken.
* A bug found during evaluation: the first blank-cell rule (mean brightness) misclassified ~0.3 % of
  thin digits as blank, lowering recovery by about 5 points. It was replaced by a max-pixel rule and all
  experiments were re-run; the numbers above are the corrected ones.
* Injected misreads are constructed: the true digit is always in the NN's top-3, so recovery is
  optimistic compared with real photographs. The natural MNIST error rate is only 2.2 %.
* Seeds vary the generated puzzles and misreads, not the NN weights.
* Silent errors: see the confidence flags above; 14.6 % of wrong answers are not flagged.
* Limits: ≤ 3 corrections; a misread that leaves a unique, solvable puzzle (e.g. a different but
  consistent grid) cannot be detected at all; confident misreads outside the 20 least-confident
  cells would be missed (none occurred in these experiments).

## 6. Conclusions
1. The three parts cooperate: NN probabilities make the search cheap and meaningful, AC-3 and FC make
   each CSP check fast, and the CSP guarantees every answer is a valid Sudoku.
2. A* with the conflict heuristic is as good as UCS and several times cheaper; Greedy is cheaper still
   but sometimes picks a worse correction.
3. The uniqueness goal is the most effective single improvement (+14 to +27 points).
4. Accuracy of the whole system depends mostly on how many misreads occur: up to 3 are manageable.
