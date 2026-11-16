# Handwritten Sudoku Reader and Solver (Project P7)

A neural network reads a Sudoku written in handwritten (MNIST) digits. When a misread digit
breaks the puzzle, **Search** finds the cheapest set of corrections (cost from NN probabilities)
and a **CSP** solver (Backtracking, Forward Checking, AC-3) tests each corrected puzzle; the
first solvable one wins. If none works, the system reports "unreadable" and lists the least
confident cells.

**Status:** complete - 54 passing tests, 3 experiments x 5 seeds, demo, report draft.

## Features
- NumPy MLP digit classifier (97.8 % test accuracy) with temperature scaling; logistic-regression baseline.
- CSP with independently switchable BT / FC / AC-3 (+ optional maintained AC-3) and counters.
- UCS, Greedy and A* (admissible conflict heuristic) over correction sets, with node counters and a search trace.
- Optional "unique solution" goal; seeded puzzle / mosaic / injected-misread generator.
- Integrated pipeline, experiments, tables and plots.

## Project structure
```
make_data.py        download + preprocess MNIST (seeded, checksum-verified)
train_nn.py         train MLP + logistic regression, save metrics
run_demo.py         integrated demo (prints trace, writes figures)
run_experiments.py  the three experiments (resumable, 5 seeds)
sudoku/             data, nn, csp, search, generator, integration, eval
tests/              pytest suite (54 tests)
docs/               data dictionary, report draft, presentation outline,
                    compliance checklist, contribution template
results/            metrics, tables.md, plots, raw experiment JSON
```

## Setup (Python >= 3.10; tested on 3.12)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # pinned versions
```
Needs internet for the first `make_data.py` (downloads ~11 MB from a GitHub MNIST mirror).

## Run
```bash
python make_data.py               # -> data/mnist19.npz
python train_nn.py                # -> data/mlp.npz, results/nn_metrics.json (~30 s on CPU)
python run_demo.py                # integrated demo, figures in results/demo_*.png
python run_demo.py --seed 5 --misreads 3 --givens 30   # other cases
python run_experiments.py         # ~5 min; writes results/tables.md and plots
python run_experiments.py --seeds 5 --n 100 --n-csp 50  # defaults, explicit
```
Note: `data/mlp.npz` and `data/mnist19.npz` are git-ignored; regenerate with the two first commands.

## Test
```bash
python -m pytest -q               # 54 tests (CSP, generator, search, system cases)
```

## Usage from Python
```python
from sudoku.nn import MLP
from sudoku.integration import run_pipeline
model = MLP.load("data/mlp.npz")
r = run_pipeline(images, model, strategy="astar", goal="unique")   # images: (81, 784) in [0,1]
print(r.status, r.correction_log(), r.message)
```

## Key results (mean ± std, 5 seeds; details in `results/tables.md`)
| misreads | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exact recovery, NN-guided (solvable goal) | 100 % | 79.8 ± 3.3 | 71.2 ± 7.5 | 59.0 ± 5.3 | 0 (exceeds 3-correction budget) |
| exact recovery, unique-solution goal | 100 % | 93.6 ± 1.5 | 89.2 ± 2.3 | 86.0 ± 2.0 | 0 |
| no correction / random correction | 100 % | 0 / 15 | 0 / 0.4 | 0 / 0 | 0 / 0 |

## Limitations
Misreads are *injected* (constructed from hard MNIST digits), the data are MNIST mosaics rather
than photographed puzzles, only one NN training seed was used, and at most 3 corrections are
allowed. See `docs/report.md` section 5.

## Recreate the environment on another computer
```bash
git clone <repo> && cd sudoku-reader
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python make_data.py && python train_nn.py && python -m pytest -q && python run_demo.py
```
