"""Run all three experiments (resumable per seed), write tables + plots to results/.
Usage: python run_experiments.py [--seeds 5] [--n 100] [--n-csp 50]"""
import argparse

from sudoku import eval as ev

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--n", type=int, default=100, help="puzzles per misread level per seed")
ap.add_argument("--n-csp", type=int, default=50, help="puzzles per givens level per seed")
ap.add_argument("--plots-only", action="store_true")
a = ap.parse_args()

if not a.plots_only:
    ev.exp_csp(a.seeds, a.n_csp)
    ev.exp_search(a.seeds, a.n)
    ev.exp_confidence(a.seeds)
import json
sr = json.loads((ev.RESULTS / "exp_search.json").read_text(encoding="utf-8"))
cr = json.loads((ev.RESULTS / "exp_csp.json").read_text(encoding="utf-8"))
(ev.RESULTS / "tables.md").write_text(ev.tables(sr, cr), encoding="utf-8")
conf = json.loads((ev.RESULTS / "exp_confidence.json").read_text(encoding="utf-8"))
(ev.RESULTS / "confidence.md").write_text(ev.confidence_table(conf), encoding="utf-8")
ev.plots(sr, cr)
ev.nn_plots()
print("Wrote results/tables.md and plots")
