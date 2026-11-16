# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A neural network reads a Sudoku written in handwritten digits. When it misreads a digit, a search finds the cheapest set of corrections (UCS, Greedy or A*), and a CSP solver (Backtracking, Forward Checking, AC-3 - each switchable) tests every corrected puzzle. If no solvable reading is found within 3 corrections, the system reports `unreadable` and lists the least confident cells. Every solved puzzle also gets a confidence tier (high / medium / low) that warns when the answer is a valid Sudoku but may not be the written one.

Course: Intelligent Systems | Name: ShehabEldin Mostafa Ibrahim | Report: [docs/report.docx]

## How the three parts cooperate
1. **NN** classifies each handwritten cell (digits 1-9) and outputs temperature-scaled probabilities. Blank cells are detected by a brightness rule (brightest pixel < 0.25), not by the NN.
2. **AC-3** checks the first reading. Two equal digits in one row, column or box empty a domain, so the reading is "broken".
3. **Search** tries correction sets. State = corrections made so far, action = change one uncertain cell to one of its top alternatives, cost = ln p(read) - ln p(new) (never negative), limit = 3 corrections.
4. **CSP** tests each corrected puzzle. The first one that is solvable (or has exactly one solution, with `goal="unique"`) wins.
5. **Confidence flag** assesses the answer: *low* if the corrected puzzle has more than one solution, *medium* if a correction was very unlikely (> 2.5 nats) or all 3 corrections were used, otherwise *high*.

## Setup (Python 3.10+, tested on 3.12)
```bash
python -m venv venv
venv\Scripts\activate             # Windows
source venv/bin/activate          # Mac / Linux
pip install -r requirements.txt
```
On Windows PowerShell, run `$env:PYTHONUTF8 = "1"` once per terminal window.

## Run commands (in this order)
```bash
python make_data.py                  # download + preprocess MNIST (seed 0)  -> data/mnist19.npz
python train_nn.py                   # MLP + logistic-regression baseline    -> data/mlp.npz, results/nn_metrics.json,
                                     #                                          results/confusion_mlp.npy, confusion_logreg.npy
python -m pytest -q                  # 59 tests: normal, edge and impossible cases, confidence flags
python run_demo.py                   # integrated demo (2 misreads)          -> results/demo_*.png
python run_demo.py --misreads 4      # 4 misreads > limit of 3 -> unreadable, or a solution flagged low/medium confidence
python run_experiments.py            # 3 experiments + confidence evaluation -> results/
```

Useful options: `run_demo.py --misreads 3 --seed 5 --givens 30`,
`train_nn.py --epochs 12 --seed 0`, `run_experiments.py --seeds 5 --n 100 --n-csp 50`,
`run_experiments.py --plots-only` (rebuild tables and plots from saved results).

Notes
* `make_data.py` downloads the four MNIST `.gz` files from a GitHub mirror and verifies their MD5 checksums. A checksum mismatch deletes the file and stops with an error.
* If the download fails, put the four MNIST `.gz` files in `data/raw/` by hand (names as listed in `sudoku/data.py`) and rerun; existing files are only checksum-verified.
* The demo always writes to the same two image files. If one is open in an image viewer (Windows locks it), the demo saves a copy with a new name and prints a `NOTE:` line. Close the viewer before rerunning.
* Training on a different computer gives slightly different network weights, so demo instances and regenerated numbers can differ in the last digits from the ones below.
* A full experiments run takes roughly 5-10 minutes on one CPU core; it is resumable per seed.

## Where the results are
| File | Contents |
|---|---|
| `results/tables.md` | all experiment tables (mean ± std over 5 seeds) |
| `results/confidence.md` | accuracy of the high / medium / low confidence tiers |
| `results/nn_metrics.json` | test accuracy, ECE before/after temperature scaling, baseline |
| `results/nn_evaluation.png` | confusion matrices and reliability diagram |
| `results/exp1_search.png`, `exp2_csp.png`, `exp3_correction.png` | experiment plots |
| `results/demo_overview.png`, `demo_cooperation.png` | demo figures |
| `results/exp_search.json`, `exp_csp.json`, `exp_confidence.json` | raw per-run records |

## Headline results (real MNIST, 5 seeds)
* **NN:** MLP 97.82 % test accuracy vs 92.16 % for logistic regression; ECE 0.0089 before and 0.0033 after temperature scaling (T = 1.45).
* **Exp 1 (UCS / Greedy / A*):** A* and UCS always return the same optimal cost; A* expands far fewer nodes (3 misreads: UCS 48.5, Greedy 22.0, A* 8.4). Greedy returns a costlier correction set in about 11 % (2 misreads) to 21 % (3 misreads) of instances. Exact-solution recovery of A* with 1 / 2 / 3 misreads: 79.8 / 71.2 / 59.0 % with goal *solvable*, 93.6 / 89.2 / 86.0 % with goal *unique*. Level 4 exceeds the 3-correction limit and recovers 0 by design.
* **Exp 2 (CSP, fixed cell order):** BT > BT+FC > BT+FC+AC-3 in assignments at every difficulty (25 givens: about 52,500 / 12,800 / 9,900; 35 givens: 520 / 181 / 68). 6 of 2,250 runs hit the 500,000-assignment cap, so those counts are lower bounds. With MRV ordering the three variants give identical counts (see design notes).
* **Exp 3 (NN-guided vs none vs random):** NN-guided recovers 79.8 / 71.2 / 59.0 % (1 / 2 / 3 misreads); random correction (up to 300 random sets) recovers 15.4 / 0.4 / 0.0 %; no correction recovers 0 whenever a cell is misread.
* **Main failure mode:** a *silent wrong answer*, where the corrected puzzle is solvable but not the true one. With goal *solvable* this happened in 101 / 144 / 205 of 500 puzzles (1 / 2 / 3 misreads, UCS); in most 1-misread cases the misread grid was still solvable, so no correction was made at all. The *unique* goal removes most of these.
* **Confidence flag:** over 1,000 generated puzzles the *low* tier was never exactly right; 85.4 % of wrong answers were flagged (medium or low) at the cost of flagging 31.5 % of correct answers. It is a warning, not a guarantee: 14.6 % of wrong answers are still labelled high.

## Project layout
| Path | Purpose |
|---|---|
| `make_data.py`, `sudoku/data.py` | reproducible download + preprocessing (digits 1-9, 50k/10k split from the training set, official test set untouched) |
| `train_nn.py`, `sudoku/nn.py` | NumPy MLP and logistic-regression baseline, accuracy / ECE, confusion matrix, temperature scaling |
| `sudoku/csp.py` | Sudoku CSP: Backtracking, Forward Checking, AC-3 as separate switches, with counters |
| `sudoku/search.py` | correction search: UCS / Greedy / A*, admissible heuristic, counters, trace, brute-force checker |
| `sudoku/generator.py` | seeded puzzle generator (unique solution), mosaic builder, injected hard-digit pools |
| `sudoku/integration.py` | read -> AC-3 -> search -> CSP pipeline; modes `nn`, `none`, `random`; confidence assessment |
| `sudoku/eval.py`, `run_experiments.py` | the three experiments, confidence evaluation, tables and plots |
| `run_demo.py` | integrated demo |
| `tests/` | 59 tests |
| `docs/` | report, presentation outline|

## Design notes
* **Cost:** ln p(read) - ln p(new). The read digit is the arg-max, so cost >= 0, which UCS and A* need.
* **Search space:** a correction *set*, each cell changed at most once, generated in increasing cell order so every set appears exactly once. Candidates = the 20 least confident given cells, 3 alternatives each, at most 3 corrections. The pipeline stops after 3,000 expansions.
* **A\* heuristic (admissible):** every equal-digit conflict forces at least one of its two cells to change. Take a greedy set of cell-disjoint conflicts and add, per conflict, the cheapest change cost of either cell. Disjoint conflicts cannot share a changed cell, so the sum never overestimates (tested against brute force). A state that needs more changes than corrections left is dropped as a dead end.
* **Goal test:** the corrected grid goes to the CSP, which first rejects duplicate givens at no search cost, then runs forward checking with a one-off AC-3 pass and MRV ordering. `goal="unique"` needs exactly one solution (the solver stops counting at 2).
* **AC-3 in the CSP:** runs once before search (it detects conflicting givens and shrinks domains); a maintained-AC-3 option exists but the pipeline does not use it.
* **Experiment protocol:** seeds change the puzzles and images, not the trained network. Non-injected cells use correctly classified images, so the injected level equals the real number of misreads. Injected images are misclassified test digits whose true digit is in the NN's top 3. Random mode = up to 300 random correction sets. Exp 2 reports fixed cell order as the main comparison; with MRV, plain BT already computes legal values on the fly, which acts like forward checking, so the three variants coincide. Exp 2 caps plain BT at 500,000 assignments.

## Known limitations
* Cost-based correction can return a confident wrong answer (see main failure mode above); the uniqueness goal and the confidence flag reduce this but do not remove it.
* At most 3 corrections, so 4 or more misreads end as `unreadable` or as a low/medium-confidence guess.
* Misreads are injected from hard MNIST digits, so recovery is optimistic compared with real photographs; the sudokus are generated mosaics of MNIST digits, not photographs of real handwritten puzzles.
* Only one NN training run (seed 0); the 5 seeds vary puzzles and misreads, not the network weights.