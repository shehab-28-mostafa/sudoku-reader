# Handwritten Sudoku Reader and Solver (Project P7)

A neural network reads a Sudoku written in handwritten (MNIST) digits. When a
misread digit breaks the puzzle, **Search** finds the cheapest set of
corrections (cost derived from NN probabilities) and a **CSP** solver
(Backtracking, Forward Checking, AC-3) tests each corrected puzzle.

**Status:** initialising (see commit history for progress).

## Objectives
- NN: classify digits 1-9 with calibrated probabilities (vs. logistic regression).
- Search: UCS / Greedy / A* over correction sets (max 3 corrections).
- CSP: 81 variables, switchable BT / FC / AC-3, with counters.
- Integration: NN reads -> AC-3 detects conflict -> Search corrects -> CSP solves.

## Stack
Python 3.12, NumPy, scikit-learn, matplotlib, pytest.

## Planned structure
```
sudoku/        data, nn, csp, search, generator, integration, eval
tests/         pytest suite
make_data.py   download + preprocess MNIST (seeded)
run_demo.py    integrated demo
```

## Prerequisites
Python >= 3.10, git.
