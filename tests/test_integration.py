"""System-level cases: >=5 normal, >=3 edge, >=2 impossible (clear 'unreadable' message)."""
import numpy as np
import pytest

from sudoku.integration import read_grid, run_pipeline
from tests.conftest import PUZ, SOL, build_images

# ------------------------------- NORMAL ---------------------------------
def test_normal_1_clean_reading_needs_no_correction(model):
    r = run_pipeline(build_images(), model)
    assert r.status == "solved" and r.solution == SOL and r.corrections == []


def test_normal_2_single_misread_is_corrected(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45)}), model)
    assert r.status == "solved" and r.solution == SOL
    assert [(c, o, n) for c, o, n, _ in r.corrections] == [(0, 3, 5)]
    assert not r.ac3_ok and "conflicting" in r.ac3_reason
    assert r.correction_log()[0].startswith("r1c1: read 3 -> 5")


def test_normal_3_two_misreads(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45), 80: (1, 0.55, 0.40)}), model)
    assert r.status == "solved" and r.solution == SOL and len(r.corrections) == 2


@pytest.mark.parametrize("strategy", ["ucs", "greedy", "astar"])
def test_normal_4_all_strategies_recover(model, strategy):
    r = run_pipeline(build_images(misreads={4: (6, 0.5, 0.4)}), model, strategy=strategy)
    assert r.status == "solved" and r.solution == SOL


def test_normal_5_unique_goal(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45)}), model, goal="unique")
    assert r.status == "solved" and r.solution == SOL


def test_normal_6_no_correction_mode_fails_on_misread(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45)}), model, mode="none")
    assert r.status == "unreadable" and "no correction attempted" in r.message


# -------------------------------- EDGE ----------------------------------
def test_edge_1_all_blank_grid(model):
    r = run_pipeline(build_images([0] * 81), model)
    assert r.status == "solved" and 0 not in r.solution


def test_edge_2_complete_grid_with_one_misread(model):
    r = run_pipeline(build_images(SOL, misreads={40: (3, 0.5, 0.45)}), model)
    assert r.status == "solved" and r.solution == SOL


def test_edge_3_expansion_limit_reported(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45)}), model, max_expansions=0)
    assert r.status == "unreadable" and "stopped after 0 expansions" in r.message


def test_edge_4_blank_detection_is_brightness_rule_not_nn(model):
    reading, blank, _ = read_grid(build_images(), model)
    assert (blank == (np.array(PUZ) == 0)).all()
    assert reading == PUZ


# ------------------------------ IMPOSSIBLE ------------------------------
def test_impossible_1_confident_conflict_no_budget(model):
    imgs = build_images(misreads={0: (3, 0.97, 0.01)})     # confidently wrong, duplicate 3
    r = run_pipeline(imgs, model, max_corrections=0)
    assert r.status == "unreadable"
    assert r.message.startswith("unreadable") and "Least confident cells" in r.message
    assert len(r.least_confident) == 5


def test_impossible_2_contradictory_puzzle_conflict_free(model):
    p = [0] * 81
    p[1:8] = [1, 2, 3, 4, 5, 6, 7]
    p[9], p[10] = 8, 9        # r1c1 would need 8 or 9, both seen -> empty domain, no equal pair
    r = run_pipeline(build_images(p), model, max_corrections=1, n_uncertain=4, k_alt=1)
    assert r.status == "unreadable" and "no puzzle reachable" in r.message


def test_impossible_3_random_mode_reports_unreadable(model):
    imgs = build_images(misreads={0: (3, 0.97, 0.01)})
    r = run_pipeline(imgs, model, mode="random", rng=np.random.default_rng(0), random_budget=5)
    assert r.status == "unreadable" and "unreadable" in r.message


# --------------------------- CONFIDENCE FLAGS ---------------------------
def test_confidence_high_for_clean_unique_solution(model):
    r = run_pipeline(build_images(), model)
    assert r.confidence == "high" and r.warnings == [] and "WARNING" not in r.message


def test_confidence_low_when_solution_not_unique(model):
    p = PUZ[:]
    for i in (0, 1, 2, 9, 10, 11, 18, 19, 20):     # clear a whole box -> many solutions
        p[i] = 0
    r = run_pipeline(build_images(p), model)
    assert r.status == "solved" and r.confidence == "low"
    assert "more than one solution" in r.message and "WARNING" in r.message


def test_confidence_medium_for_very_unlikely_correction(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.97, 0.01)}), model)   # cost ~4.6 nats
    assert r.status == "solved" and r.solution == SOL
    assert r.confidence == "medium" and "very unlikely" in r.message


def test_confidence_medium_when_budget_used(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.52, 0.45)}), model, max_corrections=1)
    assert r.status == "solved" and r.confidence == "medium" and "allowed corrections were used" in r.message


def test_confidence_not_set_when_unreadable(model):
    r = run_pipeline(build_images(misreads={0: (3, 0.97, 0.01)}), model, max_corrections=0)
    assert r.status == "unreadable" and r.confidence == "n/a" and r.warnings == []
