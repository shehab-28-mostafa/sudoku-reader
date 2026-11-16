# Requirements compliance checklist (PDF common list + P7)

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Modules: data, nn, search, csp, integration, eval, tests | Done | `sudoku/*.py`, `tests/` |
| 2 | `make_data.py` reproducible, fixed seed | Done | `make_data.py`, `sudoku/data.py` |
| 3 | Data dictionary (real vs generated) | Done | `docs/data_dictionary.md` |
| 4 | Trained NN + test results vs baseline | Done: MLP 97.82 % vs logistic regression 92.16 % | `train_nn.py`, `results/nn_metrics.json`, `results/nn_evaluation.png` |
| 5 | Search with counters, known-answer test | Done | `sudoku/search.py`, `tests/test_search.py` |
| 6 | BT, FC, AC-3 switchable, counters | Done | `sudoku/csp.py`, `tests/test_csp.py` |
| 7 | `python run_demo.py`: input, solution, plots | Done | `run_demo.py`, `results/demo_*.png` |
| 8 | README with pinned requirements and commands | Done | `README.md`, `requirements.txt` |
| 9 | >=5 normal, >=3 edge, >=2 impossible tests ("no solution" message) | Done: 54 tests (e.g. 6 normal, 4 edge, 3 impossible at system level) | `tests/test_integration.py` + others |
| 10 | 3 experiments, >=5 seeds, mean +- std, tables + plots | Done (5 seeds) | `results/tables.md`, `results/exp*.png` |
| 11 | Report (max 12 pages) | Draft written from real results; **you review/edit** | `docs/report.md` |
| 12 | 10-min talk + 5-min live demo | Outline and demo script prepared; **you present** | `docs/presentation.md` |
| 13 | Individual contribution statements + git log | Template only: **each member writes theirs**; git log included | `docs/contributions.md`, `git log` |

## P7-specific
| Requirement | Status |
|---|---|
| Calibrated classifier: accuracy, confusion matrix, calibration error vs logistic regression | Done (temperature scaling, ECE 0.0089 -> 0.0033; LogReg ECE 0.0061) |
| Puzzle + mosaic generator with seed | Done (`sudoku/generator.py`) |
| Correction log per solved puzzle (cell, read, new, cost) | Done (`PipelineResult.correction_log()`, shown in demo) |
| Exact-solution recovery on 100 puzzles per misread level | Done (100 per level per seed, 5 seeds) |
| Minimum: NN + full CSP + UCS, "solvable" goal, up to 2 misreads | Done (up to 3 and tested at 4) |
| Extensions: uniqueness goal, A* heuristic | Done; temperature scaling done; per-student/other not applicable |
| Experiments 1-3 as specified | Done |

## Known limitations / incomplete
* 4 misreads exceed the 3-correction budget by design -> 0 % recovery (reported, not hidden).
* Only one NN training seed: seed variance covers puzzle generation, not NN re-training.
* Injected "hard digits" are constructed (true digit always in the NN's top-3); real-world photographed Sudoku not covered.
* Experiment 2, static ordering: 6 of 2,250 runs hit the 500k-assignment cap (counts are lower bounds).
* Git author identity is a placeholder (`Shehab <shehab@example.com>`); rewrite if needed.
