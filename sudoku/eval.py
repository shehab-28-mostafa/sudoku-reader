"""Experiments, tables and plots.

exp_search : strategies (UCS/Greedy/A*) and correction modes (NN-guided/none/random)
             on instances with 0-4 injected misreads  -> Experiments 1 and 3
exp_csp    : BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25/30/35 givens -> Experiment 2
Every experiment is repeated for several seeds (puzzle-generator seeds); tables show
mean +- std across the per-seed averages.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .csp import solve
from .data import load_processed
from .generator import DigitPool, make_instance, make_puzzle
from .integration import run_pipeline
from .nn import MLP

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEVELS = [0, 1, 2, 3, 4]
SEARCH_RUNS = [("ucs", dict(mode="nn", strategy="ucs")),
               ("greedy", dict(mode="nn", strategy="greedy")),
               ("astar", dict(mode="nn", strategy="astar")),
               ("none", dict(mode="none")),
               ("random", dict(mode="random")),
               ("astar_unique", dict(mode="nn", strategy="astar", goal="unique"))]
CSP_CONFIGS = [("BT", dict()), ("BT+FC", dict(fc=True)), ("BT+FC+AC-3", dict(fc=True, ac3=True))]


def _load():
    d = load_processed()
    model = MLP.load(Path(__file__).resolve().parent.parent / "data" / "mlp.npz")
    pool = DigitPool(d["x_test"], d["y_test"], model.predict_proba(d["x_test"]))
    return model, pool


# ----------------------------------------------------------------------------
def exp_search(seeds=5, n=100, givens=30, out=RESULTS / "exp_search.json", log=print):
    model, pool = _load()
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["seed"] for r in records}
    for s in range(seeds):
        if s in done:
            continue
        rng = np.random.default_rng(1000 + s)
        t0 = time.time()
        for lvl in LEVELS:
            for k in range(n):
                inst = make_instance(rng, pool, givens, lvl)
                cand_cover = None
                for name, kw in SEARCH_RUNS:
                    t = time.perf_counter()
                    r = run_pipeline(inst["images"], model, rng=np.random.default_rng(s * 100000 + lvl * 1000 + k), **kw)
                    dt = time.perf_counter() - t
                    if name == "ucs":  # can the injected cells even be reached?
                        from .search import CorrectionProblem
                        cand_cover = set(inst["injected"]) <= set(CorrectionProblem(r.proba, r.blank).candidates)
                    records.append(dict(
                        seed=s, level=lvl, run=name, solved=r.status == "solved",
                        exact=r.status == "solved" and r.solution == inst["solution"],
                        expanded=r.nodes_expanded, generated=r.nodes_generated,
                        goal_tests=r.goal_tests, cost=r.cost if r.status == "solved" else None,
                        n_corr=len(r.corrections), covered=cand_cover, time=dt,
                        n_injected=len(inst["injected"])))
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(records), encoding="utf-8")
        log(f"[exp_search] seed {s} done in {time.time() - t0:.0f}s")
    return records


def exp_csp(seeds=5, n=50, out=RESULTS / "exp_csp.json", cap=500_000, log=print):
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["seed"] for r in records}
    for s in range(seeds):
        if s in done:
            continue
        rng = np.random.default_rng(2000 + s)
        for g in (25, 30, 35):
            for _ in range(n):
                puz, sol = make_puzzle(rng, g)
                for order in ("static", "mrv"):
                    for name, kw in CSP_CONFIGS:
                        t = time.perf_counter()
                        r = solve(puz, order=order, max_assignments=cap, **kw)
                        dt = time.perf_counter() - t
                        records.append(dict(seed=s, givens=g, order=order, config=name,
                                            status=r.status, correct=r.solution == sol,
                                            assignments=r.assignments, backtracks=r.backtracks,
                                            ac3_revisions=r.ac3_revisions, pruned=r.pruned, time=dt))
        out.write_text(json.dumps(records), encoding="utf-8")
        log(f"[exp_csp] seed {s} done")
    return records


# ----------------------------------------------------------------------------
def _agg(records, keys, value, filt=lambda r: True):
    """mean and std (across seeds) of per-seed means of `value`, grouped by `keys`."""
    groups: dict[tuple, dict[int, list]] = {}
    for r in records:
        if not filt(r) or r.get(value) is None:
            continue
        groups.setdefault(tuple(r[k] for k in keys), {}).setdefault(r["seed"], []).append(float(r[value]))
    out = {}
    for g, per_seed in groups.items():
        means = np.array([np.mean(v) for v in per_seed.values()])
        out[g] = (float(means.mean()), float(means.std(ddof=1)) if len(means) > 1 else 0.0, len(means))
    return out


def _fmt(ms, pct=False, digits=2):
    m, s, _ = ms
    return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.{digits}f} ± {s:.{digits}f}"


def tables(search_rec, csp_rec) -> str:
    L = []
    n_seeds = len({r["seed"] for r in search_rec})
    L.append(f"_All numbers: mean ± std over {n_seeds} seeds (per-seed averages of 100 puzzles/level)._\n")
    names = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*", "none": "No correction", "random": "Random", "astar_unique": "NN-guided, unique goal"}

    # --- Experiment 1
    L.append("## Experiment 1 - UCS vs Greedy vs A* (30 givens, max 3 corrections)\n")
    for metric, title, pct in [("exact", "Exact-solution recovery rate (%)", True),
                               ("expanded", "Nodes expanded", False),
                               ("generated", "Nodes generated", False),
                               ("goal_tests", "Goal tests (CSP calls)", False)]:
        agg = _agg(search_rec, ("run", "level"), metric, lambda r: r["run"] in ("ucs", "greedy", "astar"))
        L.append(f"**{title}**\n\n| misreads | UCS | Greedy | A* |\n|---|---|---|---|")
        for lvl in LEVELS:
            L.append(f"| {lvl} | " + " | ".join(_fmt(agg[(k, lvl)], pct) for k in ("ucs", "greedy", "astar")) + " |")
        L.append("")
    # paired cost on instances where all three strategies solved
    by = {}
    for r in search_rec:
        if r["run"] in ("ucs", "greedy", "astar"):
            by.setdefault((r["seed"], r["level"]), {}).setdefault(r["run"], []).append(r)
    L.append("**Mean correction cost (nats) on instances solved by all three strategies**\n\n"
             "| misreads | UCS | Greedy | A* | Greedy worse than UCS (% of instances) |\n|---|---|---|---|---|")
    for lvl in LEVELS[1:]:
        cs = {k: [] for k in ("ucs", "greedy", "astar")}
        worse, tot = 0, 0
        for (seed, l), runs in by.items():
            if l != lvl:
                continue
            for u, g, a in zip(runs["ucs"], runs["greedy"], runs["astar"]):
                if u["solved"] and g["solved"] and a["solved"]:
                    for k, x in (("ucs", u), ("greedy", g), ("astar", a)):
                        cs[k].append(x["cost"])
                    tot += 1
                    worse += g["cost"] > u["cost"] + 1e-9
        if tot:
            L.append(f"| {lvl} | " + " | ".join(f"{np.mean(cs[k]):.2f}" for k in ("ucs", "greedy", "astar"))
                     + f" | {100 * worse / tot:.1f} |")
    L.append("")
    cov = _agg(search_rec, ("level",), "covered", lambda r: r["run"] == "ucs")
    L.append("**Share of puzzles whose injected cells are all among the 20 least-confident cells "
             "(upper bound on recovery)**\n\n| misreads | covered (%) |\n|---|---|")
    for lvl in LEVELS:
        L.append(f"| {lvl} | {_fmt(cov[(lvl,)], True)} |")
    L.append("")

    # --- Experiment 3
    L.append("## Experiment 3 - NN-guided correction vs no correction vs random correction\n")
    for metric, title, pct in [("exact", "Exact-solution recovery rate (%)", True),
                               ("solved", "Any solvable puzzle found (%)", True),
                               ("goal_tests", "Goal tests (CSP calls)", False)]:
        agg = _agg(search_rec, ("run", "level"), metric, lambda r: r["run"] in ("ucs", "none", "random", "astar_unique"))
        L.append(f"**{title}**\n\n| misreads | NN-guided (UCS, solvable goal) | No correction | Random | NN-guided (A*, unique-solution goal, extension) |\n|---|---|---|---|---|")
        for lvl in LEVELS:
            L.append(f"| {lvl} | " + " | ".join(_fmt(agg[(k, lvl)], pct) for k in ("ucs", "none", "random", "astar_unique")) + " |")
        L.append("")

    # --- Experiment 2
    L.append("## Experiment 2 - BT vs BT+FC vs BT+FC+AC-3 (5 seeds x 50 puzzles per givens level)\n")
    for order in ("static", "mrv"):
        L.append(f"### Variable ordering: {order}\n")
        for metric, title in [("assignments", "Assignments"), ("backtracks", "Backtracks"),
                              ("time", "Time (s)")]:
            agg = _agg(csp_rec, ("config", "givens"), metric, lambda r: r["order"] == order)
            d = 4 if metric == "time" else 1
            L.append(f"**{title}**\n\n| givens | BT | BT+FC | BT+FC+AC-3 |\n|---|---|---|---|")
            for g in (25, 30, 35):
                L.append(f"| {g} | " + " | ".join(_fmt(agg[(c, g)], digits=d) for c, _ in CSP_CONFIGS) + " |")
            L.append("")
        lim = sum(r["status"] == "limit" for r in csp_rec if r["order"] == order)
        bad = sum(r["status"] == "solved" and not r["correct"] for r in csp_rec if r["order"] == order)
        L.append(f"_Runs hitting the assignment cap: {lim}; wrong solutions: {bad}._\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------
def plots(search_rec, csp_rec, outdir=RESULTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)
    names = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*", "none": "No correction", "random": "Random", "astar_unique": "NN-guided, unique goal"}

    def lines(ax, runs, metric, pct=False, log=False):
        agg = _agg(search_rec, ("run", "level"), metric)
        for k in runs:
            m = np.array([agg[(k, l)][0] for l in LEVELS]) * (100 if pct else 1)
            s = np.array([agg[(k, l)][1] for l in LEVELS]) * (100 if pct else 1)
            ax.errorbar(LEVELS, m, yerr=s, marker="o", capsize=3, label=names[k])
        ax.set_xlabel("injected misreads"); ax.set_xticks(LEVELS); ax.grid(alpha=.3); ax.legend()
        if log:
            ax.set_yscale("log")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lines(ax[0], ["ucs", "greedy", "astar"], "exact", pct=True); ax[0].set_ylabel("exact recovery (%)")
    lines(ax[1], ["ucs", "greedy", "astar"], "expanded", log=True); ax[1].set_ylabel("nodes expanded (log)")
    fig.suptitle("Experiment 1: UCS vs Greedy vs A*"); fig.tight_layout(); fig.savefig(outdir / "exp1_search.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lines(ax[0], ["ucs", "astar_unique", "none", "random"], "exact", pct=True); ax[0].set_ylabel("exact recovery (%)")
    lines(ax[1], ["ucs", "astar_unique", "none", "random"], "goal_tests", log=True); ax[1].set_ylabel("CSP calls (log)")
    fig.suptitle("Experiment 3: NN-guided vs no vs random correction"); fig.tight_layout(); fig.savefig(outdir / "exp3_correction.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, order in enumerate(("static", "mrv")):
        for col, metric in enumerate(("assignments", "backtracks")):
            agg = _agg(csp_rec, ("config", "givens"), metric, lambda r, o=order: r["order"] == o)
            ax = axes[row, col]
            for j, (c, _) in enumerate(CSP_CONFIGS):
                m = [max(agg[(c, g)][0], 0.5) for g in (25, 30, 35)]
                s = [agg[(c, g)][1] for g in (25, 30, 35)]
                ax.bar(np.arange(3) + j * .27, m, .27, yerr=s, capsize=3, label=c)
            ax.set_xticks(np.arange(3) + .27); ax.set_xticklabels(["25", "30", "35"])
            ax.set_yscale("log"); ax.set_xlabel("givens"); ax.set_ylabel(f"{metric} (log)")
            ax.set_title(f"{order} ordering"); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Experiment 2: CSP techniques"); fig.tight_layout(); fig.savefig(outdir / "exp2_csp.png", dpi=130); plt.close(fig)


def nn_plots(outdir=RESULTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .nn import expected_calibration_error

    d = load_processed()
    model = MLP.load(Path(__file__).resolve().parent.parent / "data" / "mlp.npz")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, name in zip(ax[:2], ("mlp", "logreg")):
        cm = np.load(outdir / f"confusion_{name}.npy")
        a.imshow(cm, cmap="Blues"); a.set_title(f"Confusion matrix: {name}")
        a.set_xticks(range(9)); a.set_xticklabels(range(1, 10)); a.set_yticks(range(9)); a.set_yticklabels(range(1, 10))
        a.set_xlabel("predicted"); a.set_ylabel("true")
        for i in range(9):
            for j in range(9):
                if cm[i, j] > 0 and (i != j):
                    a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    for cal, lab in ((False, "before T-scaling"), (True, "after T-scaling")):
        p = model.predict_proba(d["x_test"], calibrated=cal)
        conf, corr = p.max(1), (p.argmax(1) + 1 == d["y_test"])
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax[2].plot(xs, ys, marker="o", label=f"{lab} (ECE {expected_calibration_error(p, d['y_test']):.4f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=.8); ax[2].set_xlabel("confidence"); ax[2].set_ylabel("accuracy")
    ax[2].set_title("Reliability diagram (MLP, test)"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "nn_evaluation.png", dpi=130); plt.close(fig)


def exp_confidence(seeds=5, n=40, givens=30, out=RESULTS / "exp_confidence.json", log=print):
    """How well do the confidence tiers (high/medium/low) separate right from wrong
    answers? Solvable-goal A*; misread levels 0-4; only puzzles the system 'solved'."""
    model, pool = _load()
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["seed"] for r in records}
    for s in range(seeds):
        if s in done:
            continue
        rng = np.random.default_rng(3000 + s)
        for lvl in LEVELS:
            for _ in range(n):
                inst = make_instance(rng, pool, givens, lvl)
                r = run_pipeline(inst["images"], model, strategy="astar")
                records.append(dict(seed=s, level=lvl, solved=r.status == "solved",
                                    exact=r.status == "solved" and r.solution == inst["solution"],
                                    tier=r.confidence))
        out.write_text(json.dumps(records), encoding="utf-8")
        log(f"[exp_confidence] seed {s} done")
    return records


def confidence_table(rec) -> str:
    solved = [r for r in rec if r["solved"]]
    wrong = [r for r in solved if not r["exact"]]
    L = ["## Confidence flags (solvable goal, A*, misread levels 0-4 pooled)\n",
         f"{len(rec)} puzzles, {len(solved)} solved, {len(wrong)} of those solved answers are wrong "
         f"(valid Sudoku but not the written one).\n",
         "| tier | answers | % exactly right | share of all wrong answers caught |", "|---|---|---|---|"]
    for t in ("high", "medium", "low"):
        rs = [r for r in solved if r["tier"] == t]
        if rs:
            right = 100 * np.mean([r["exact"] for r in rs])
            caught = 100 * sum(not r["exact"] for r in rs) / max(len(wrong), 1)
            L.append(f"| {t} | {len(rs)} | {right:.1f} | {caught:.1f} |")
    flagged = [r for r in wrong if r["tier"] != "high"]
    fa = [r for r in solved if r["exact"] and r["tier"] != "high"]
    L.append(f"\nWrong answers flagged (medium or low): {100 * len(flagged) / max(len(wrong), 1):.1f} %. "
             f"Correct answers flagged (false alarms): {100 * len(fa) / max(sum(r['exact'] for r in solved), 1):.1f} %.")
    return "\n".join(L)
