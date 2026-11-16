"""Train MLP + logistic-regression baseline; save model and metrics.
Usage: python train_nn.py [--seed 0] [--epochs 12]"""
import argparse, json, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from sudoku.data import load_processed
from sudoku.nn import MLP, expected_calibration_error, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=12)
args = ap.parse_args()

d = load_processed()
res = Path("results"); res.mkdir(exist_ok=True)

t = time.time()
mlp = MLP(seed=args.seed).fit(d["x_train"], d["y_train"], epochs=args.epochs,
                              seed=args.seed, x_val=d["x_val"], y_val=d["y_val"])
print(f"MLP trained in {time.time() - t:.0f}s")
ece_before = expected_calibration_error(mlp.predict_proba(d["x_test"], False), d["y_test"])
T = mlp.fit_temperature(d["x_val"], d["y_val"])
mlp.save("data/mlp.npz")
p = mlp.predict_proba(d["x_test"])
metrics = {
    "mlp": {"test_acc": float((p.argmax(1) + 1 == d["y_test"]).mean()),
            "ece_before_T": ece_before,
            "ece_after_T": expected_calibration_error(p, d["y_test"]),
            "temperature": T},
}
np.save(res / "confusion_mlp.npy", confusion_matrix(p.argmax(1) + 1, d["y_test"]))

t = time.time()
lr = LogisticRegression(max_iter=1000, random_state=args.seed)
lr.fit(d["x_train"], d["y_train"])
pl = lr.predict_proba(d["x_test"])
print(f"LogReg trained in {time.time() - t:.0f}s")
metrics["logreg"] = {"test_acc": float((lr.predict(d["x_test"]) == d["y_test"]).mean()),
                     "ece": expected_calibration_error(pl, d["y_test"])}
np.save(res / "confusion_logreg.npy", confusion_matrix(lr.predict(d["x_test"]), d["y_test"]))
(res / "nn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
