"""Integrated demo:  python run_demo.py [--seed 3] [--misreads 2] [--givens 30]

Shows the input (image mosaic), the NN reading, the AC-3 conflict, the search trace,
the CSP solution and plots of how the three parts cooperate. Figures -> results/demo_*.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sudoku.csp import cell_name, find_conflicts, solve
from sudoku.data import load_processed
from sudoku.generator import DigitPool, make_instance, to_mosaic
from sudoku.integration import run_pipeline
from sudoku.nn import MLP

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--misreads", type=int, default=2)
ap.add_argument("--givens", type=int, default=30)
args = ap.parse_args()
out = Path("results"); out.mkdir(exist_ok=True)

d = load_processed()
model = MLP.load("data/mlp.npz")
pool = DigitPool(d["x_test"], d["y_test"], model.predict_proba(d["x_test"]))
rng = np.random.default_rng(args.seed)


def save_fig(fig, name):
    """Save a figure; if Windows refuses (file open in a viewer, sync lock...) use a new name."""
    path = out / name
    try:
        fig.savefig(path, dpi=130)
    except OSError as e:
        import time
        alt = out / f"{path.stem}_{int(time.time())}{path.suffix}"
        fig.savefig(alt, dpi=130)
        print(f"NOTE: could not write {path} ({e}); saved {alt} instead. "
              f"Close any program that has the old image open.")
    plt.close(fig)


def draw_grid(ax, grid, title, red=(), green=(), orig=None):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)
    for i in range(81):
        r, c = divmod(i, 9)
        if i in red:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#f4a6a6"))
        if i in green:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#a6e3a6"))
        if grid[i]:
            ax.text(c + .5, r + .55, str(grid[i]), ha="center", va="center", fontsize=12,
                    color="black" if (orig is None or orig[i]) else "#1f5fbf")
    for k in range(10):
        lw = 2 if k % 3 == 0 else .5
        ax.plot([0, 9], [k, k], "k", lw=lw); ax.plot([k, k], [0, 9], "k", lw=lw)


# --- pick the first instance of the requested difficulty the system recovers exactly
for attempt in range(200):
    inst = make_instance(rng, pool, args.givens, args.misreads)
    res = run_pipeline(inst["images"], model, mode="nn", strategy="ucs")
    if res.status == "solved" and res.solution == inst["solution"] and len(res.corrections) == len(inst["injected"]):
        break
print(f"=== INPUT: {args.givens} givens, {len(inst['injected'])} injected hard digit(s) "
      f"(instance #{attempt}, seed {args.seed}) ===")
print("Hard digits at:", ", ".join(cell_name(c) for c in inst["injected"]))
print("\n=== 1. NN reads the grid ===")
wrong = [i for i in range(81) if res.reading[i] != inst["puzzle"][i]]
for i in wrong:
    p = res.proba[i]
    top = np.argsort(-p)[:2] + 1
    print(f"  {cell_name(i)}: true {inst['puzzle'][i]}, read {res.reading[i]} "
          f"(p={p[res.reading[i]-1]:.2f}); runner-up {top[1]} (p={p[top[1]-1]:.2f})")
print("\n=== 2. AC-3 / consistency check on the reading ===")
print("  consistent:", res.ac3_ok, "|", res.ac3_reason or "no conflict")
for a, b, v in find_conflicts(res.reading):
    print(f"  conflict: {cell_name(a)} and {cell_name(b)} both read {v}")

print("\n=== 3. Search trace (UCS: lowest-cost correction sets tested first) ===")
trace = []
run_pipeline(inst["images"], model, mode="nn", strategy="ucs", trace=trace)
for t in trace[:12]:
    desc = ", ".join(f"{cell_name(c)}->{v}" for c, v in t["state"]) or "(no correction)"
    print(f"  test {t['order']:2d}: cost {t['g']:.3f}  {desc:40s} -> {'SOLVABLE' if t['goal'] else 'rejected'}")
if len(trace) > 12:
    print(f"  ... {len(trace) - 12} more; last: {'SOLVABLE' if trace[-1]['goal'] else 'rejected'}")

print("\n=== 4. Correction log + CSP solution ===")
for line in res.correction_log():
    print("  ", line)
print("  status:", res.status, "| exact recovery (known only because this is a generated test):",
      res.solution == inst["solution"])
print(f"  CONFIDENCE: {res.confidence.upper()}")
for w in res.warnings:
    print(f"  !! WARNING: {w}")
if res.confidence in ("low", "medium"):
    print("  !! Treat this solution as UNVERIFIED - it is a valid Sudoku but may not be the written one.")

# --- plots: overview
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
mos = to_mosaic(inst["images"])
ax[0].imshow(mos, cmap="gray_r"); ax[0].axis("off"); ax[0].set_title("Input: handwritten Sudoku (red = injected hard digit)", fontsize=10)
for i in inst["injected"]:
    r, c = divmod(i, 9)
    ax[0].add_patch(plt.Rectangle((c * 28, r * 28), 28, 28, fill=False, ec="red", lw=2))
for k in range(10):
    ax[0].plot([0, 252], [k * 28] * 2, color="#bbb", lw=.5 + (k % 3 == 0)); ax[0].plot([k * 28] * 2, [0, 252], color="#bbb", lw=.5 + (k % 3 == 0))
conf_cells = {x for a, b, _ in find_conflicts(res.reading) for x in (a, b)}
draw_grid(ax[1], res.reading, "NN reading (red = conflict)", red=conf_cells)
draw_grid(ax[2], res.solution or [0] * 81, "Corrected + solved (green = corrected, blue = CSP)",
          green={c for c, *_ in res.corrections}, orig=res.reading)
if res.confidence in ("low", "medium"):
    ax[2].text(4.5, 9.9, f"WARNING: {res.confidence.upper()} CONFIDENCE - treat as unverified",
               ha="center", color="red", fontsize=11, fontweight="bold")
fig.tight_layout(); save_fig(fig, "demo_overview.png")

# --- plots: cooperation of the three parts
fixed = res.reading[:]
for c, _, n, _ in res.corrections:
    fixed[c] = n
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
gs = [t["g"] for t in trace]
ax[0, 0].plot(range(1, len(gs) + 1), np.minimum.accumulate(gs) if False else gs, marker="o")
ax[0, 0].scatter([t["order"] for t in trace if t["goal"]], [t["g"] for t in trace if t["goal"]], c="green", s=90, zorder=3, label="accepted by CSP")
ax[0, 0].set_xlabel("goal test #"); ax[0, 0].set_ylabel("correction cost g (nats)"); ax[0, 0].set_title("Search trace (UCS): costs never decrease"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
names, exp = [], []
for s in ("ucs", "greedy", "astar"):
    rr = run_pipeline(inst["images"], model, strategy=s)
    names.append({"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}[s]); exp.append(rr.nodes_expanded)
ax[0, 1].bar(names, exp, color=["#4c78a8", "#f58518", "#54a24b"]); ax[0, 1].set_ylabel("nodes expanded"); ax[0, 1].set_title("Search strategies on this puzzle")
cfgs = [("BT", {}), ("BT+FC", dict(fc=True)), ("BT+FC+AC-3", dict(fc=True, ac3=True))]
rs = [solve(fixed, **kw) for _, kw in cfgs]
w = .38
ax[1, 0].bar(np.arange(3) - w / 2, [r.assignments for r in rs], w, label="assignments")
ax[1, 0].bar(np.arange(3) + w / 2, [r.backtracks for r in rs], w, label="backtracks")
ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels([n for n, _ in cfgs]); ax[1, 0].set_yscale("symlog"); ax[1, 0].legend()
ax[1, 0].set_title("CSP pruning on the corrected puzzle")
print("\n=== 5. CSP pruning with / without FC and AC-3 (corrected puzzle) ===")
for (n, _), r in zip(cfgs, rs):
    print(f"  {n:11s}: assignments {r.assignments:5d}  backtracks {r.backtracks:5d}  AC-3 revisions {r.ac3_revisions}")
modes = [("NN on\n(UCS)", dict(mode="nn")), ("NN off\n(no correction)", dict(mode="none")), ("random\ncorrection", dict(mode="random"))]
ok = []
print("\n=== 6. System with NN on versus off ===")
for lab, kw in modes:
    rr = run_pipeline(inst["images"], model, rng=np.random.default_rng(0), **kw)
    good = rr.status == "solved" and rr.solution == inst["solution"]
    ok.append(1 if good else 0)
    flag = f"  confidence={rr.confidence}" if rr.status == "solved" else ""
    print(f"  {lab.replace(chr(10), ' '):28s}: {rr.status:10s} exact={good}  CSP calls={rr.goal_tests}{flag}")
ax[1, 1].bar([m[0] for m in modes], ok, color=["#54a24b", "#e45756", "#e45756"]); ax[1, 1].set_ylim(0, 1.2)
ax[1, 1].set_yticks([0, 1]); ax[1, 1].set_yticklabels(["failed", "exact solution"]); ax[1, 1].set_title("NN on vs off")
fig.suptitle("How NN, Search and CSP cooperate"); fig.tight_layout(); save_fig(fig, "demo_cooperation.png")
print("\nFigures: results/demo_overview.png, results/demo_cooperation.png")
