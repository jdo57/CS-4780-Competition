"""CS 4780 Spring 2024 – Heart‑Disease Detection 🩺
A tidy, reproducible pipeline for the course Kaggle competition.

Usage (in Colab):
>>> !python kaggle_heart_disease_pipeline.py --drive

The script will:
1. Mount Google Drive (optional –‑k|--drive flag).
2. Load *train.csv*, *validation.csv*, and *test.csv* from the Drive root.
3. Train several models (LogReg, tuned SVM, tuned RF, LightGBM).
4. Evaluate on *validation.csv* and pick the best 1‑metric model.
5. Persist:
   • The fitted pipeline →  *models/best_model.joblib*
   • Validation metrics    → *models/metrics.json*
   • Test predictions      → *predictions/pred_<model>.csv*

Author: Josue Ortiz‑Ordonez
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Optional (but will be skipped gracefully if unavailable)
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover – LightGBM might be absent in tiny runtimes
    lgb = None  # type: ignore

RANDOM_STATE = 42
DATAFILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "label"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Heart‑Disease Kaggle pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/gdrive/My Drive"),
        help="Directory containing train/validation/test CSVs",
    )
    parser.add_argument("--drive", action="store_true", help="Mount Google Drive (Colab only)")
    return parser.parse_args(argv)

def mount_drive() -> None:
    """Mount Google Drive inside Colab, if available."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/gdrive", force_remount=False)
    except ModuleNotFoundError:
        print("[WARN] google.colab not detected – skipping drive.mount()")


def read_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test CSVs into DataFrames."""
    data: Dict[str, pd.DataFrame] = {}
    for split, fname in DATAFILES.items():
        file_path = data_dir / fname
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file '{file_path}' not found.")
        data[split] = pd.read_csv(file_path)
        print(f"✓ Loaded {split} ({len(data[split])} rows)")
    return data["train"], data["validation"], data["test"]

def build_base_pipeline(model) -> Pipeline:
    """Return a full sklearn pipeline: impute → scale → estimator."""

    numeric_transformer = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURES)], remainder="drop"
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def model_zoo() -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    """Define models and their parameter grids for tuning."""
    models: Dict[str, Tuple[Pipeline, Dict[str, list]]] = {}

    # Logistic Regression ======================================================
    lr = build_base_pipeline(LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
    models["logreg"] = (lr, {"model__C": [0.1, 1, 10]})

    # SVM (RBF) ================================================================
    svm = build_base_pipeline(SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE))
    models["svm"] = (
        svm,
        {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.1, 0.01, 0.001],
        },
    )

    # Random Forest ============================================================
    rf = build_base_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    models["rf"] = (
        rf,
        {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 20, 30],
            "model__min_samples_split": [2, 5],
        },
    )

    # LightGBM (optional) ======================================================
    if lgb is not None:
        lgbm_clf = build_base_pipeline(
            lgb.LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
        )
        models["lgbm"] = (
            lgbm_clf,
            {
                "model__num_leaves": [31, 63],
                "model__feature_fraction": [0.9, 0.95],
                "model__bagging_fraction": [0.8, 0.9],
            },
        )
    return models

def tune_and_eval(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[float, Pipeline, dict]:
    """Run GridSearchCV → return val accuracy, best pipeline, best_params."""

    print(f"\n▶ Tuning {model_name}…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_model: Pipeline = gs.best_estimator_

    y_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"  → best CV = {gs.best_score_:.4f}, val acc = {val_acc:.4f}")

    return val_acc, best_model, gs.best_params_

# ──────────────────────────────────────────────────────── MAIN ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.drive:
        mount_drive()

    train_df, val_df, test_df = read_data(args.data_dir)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    best_overall = {"name": None, "acc": 0.0, "model": None, "params": {}}
    metrics: Dict[str, float] = {}

    for name, (pipeline, grid) in model_zoo().items():
        acc, fitted_model, best_params = tune_and_eval(
            name, pipeline, grid, X_train, y_train, X_val, y_val
        )
        metrics[name] = acc
        if acc > best_overall["acc"]:
            best_overall.update(
                {"name": name, "acc": acc, "model": fitted_model, "params": best_params}
            )

    # ───────────────────────── Persist artefacts ────────────────────────────
    models_dir = args.data_dir / "models"
    preds_dir = args.data_dir / "predictions"
    models_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    assert best_overall["model"] is not None, "No model was trained!"

    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_overall["model"], model_path)
    print(f"\n💾 Saved best model → {model_path.relative_to(args.data_dir)}")

    metrics_path = models_dir / "metrics.json"
    with metrics_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"💾 Metrics           → {metrics_path.relative_to(args.data_dir)}")

    # ───────────────────────── Predictions on test ──────────────────────────
    X_test = test_df[FEATURES]
    y_test_pred = best_overall["model"].predict(X_test)

    sub = pd.DataFrame({"id": test_df["id"], "label": y_test_pred})
    sub_path = preds_dir / f"pred_{best_overall['name']}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"💾 Test predictions   → {sub_path.relative_to(args.data_dir)}")

    # Print summary
    print("\n═══════════ SUMMARY ═══════════")
    for n, acc in metrics.items():
        flag = " ← best" if n == best_overall["name"] else ""
        print(f"{n:<10}: {acc:.4f}{flag}")
    print("══════════════════════════════")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
