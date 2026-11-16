# Data dictionary

## Source (real)
MNIST handwritten digits (60,000 train / 10,000 test, 28x28 grey; licence not verified from the
original page - confirm before redistributing; raw files are git-ignored). Downloaded from the GitHub mirror `fgnt/mnist`, MD5-verified against the
official checksums in `sudoku/data.py`; `make_data.py` is deterministic).

## Processed file `data/mnist19.npz` (made by `make_data.py --seed 0`)
| key | shape | type | real/generated | description |
|---|---|---|---|---|
| `x_train` | 45053 x 784 | float32 in [0,1] | real | flattened images, digits 1-9 only |
| `y_train` | 45053 | int64 1..9 | real | labels |
| `x_val`, `y_val` | 9024 | same | real | validation split (temperature scaling, model selection) |
| `x_test`, `y_test` | 9020 | same | real | official MNIST test set, untouched until evaluation |

Split: a seeded permutation of the 60,000 official training images gives 50,000 train /
10,000 validation; digit 0 is then dropped everywhere (hence 45,053 / 9,024 / 9,020).

## Generated parts
| item | how it is made |
|---|---|
| Full Sudoku grid | randomised backtracking, MRV, `numpy.random.default_rng(seed)` |
| Puzzle | cells removed in random order while the CSP (FC + AC-3, 2-solution limit) still finds exactly 1 solution; retried (<=50 grids) until exactly 25/30/35 givens remain |
| Blank cell image | Gaussian noise N(0, 0.01), clipped to [0,1] (max pixel ~0.055) |
| Blank detection (not the NN) | cell is blank iff its brightest pixel < 0.25. Every MNIST digit has max pixel 1.0. A mean-brightness rule (<0.04) was tried first and wrongly blanked ~0.3% of thin digits (mostly 1s), which silently made puzzles non-unique; fixed and noted in the report |
| Given-cell image | random test image of the right digit |
| **Injected hard digit** | `wrong[d]` pool = test images of digit d that the trained NN misclassifies but whose true digit is in its top-3. For a puzzle with m misreads, m random givens draw from `wrong[d]`; all other givens draw from `right[d]` (correctly read images). So "misread level" is exact. This is *not* natural MNIST error rate |
| Mosaic | 9x9 grid of 28x28 cells -> 252x252 image (display) / (81, 784) array (NN input) |
| Correction cost | `max(0, ln p(read) - ln p(new))` with temperature-scaled probabilities (T = 1.45) |

## Parameters (defaults)
givens 25/30/35 (30 for experiments 1 and 3) · max 3 corrections · 20 least-confident
candidate cells · top-3 alternative digits per cell · generator seeds 1000-1004 (exp. 1/3),
2000-2004 (exp. 2) · NN training seed 0 · assignment cap 500,000 (exp. 2).
