# 10-minute talk + 5-minute live demo

## Talk (10 min)
1. (1 min) Problem: handwritten Sudoku, misreads break puzzles. Show `results/demo_overview.png`.
2. (1.5 min) Architecture: NN → AC-3 → Search → CSP loop (one slide, arrows).
3. (1.5 min) NN: MLP vs logistic regression, calibration (`nn_evaluation.png`).
4. (1.5 min) CSP: BT / FC / AC-3 and counters. Experiment 2 table.
5. (1.5 min) Search: cost = ln(p_read / p_new), UCS vs Greedy vs A*, heuristic argument. Experiment 1.
6. (1.5 min) Experiment 3: NN-guided vs none vs random; solvable vs unique goal.
7. (1 min) Limits (constructed misreads, 3-correction budget, blank rule bug we found) and conclusions.

## Live demo (5 min)
1. `python run_demo.py` (≈10 s). Walk through the printed steps: reading → AC-3 conflicts → **search trace** (costs, rejected / SOLVABLE) → correction log.
2. Open `results/demo_cooperation.png`: **CSP pruning with and without FC / AC-3** (bottom-left) and **system with NN on vs off** (bottom-right).
3. Optional: `python run_demo.py --seed 5 --misreads 3` for a harder case, or `--misreads 4` to show the clear "unreadable" outcome with the least-confident cells.
