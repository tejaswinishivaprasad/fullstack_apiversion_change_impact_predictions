#!/usr/bin/env python3
"""
train_model.py

Train models (Logistic Regression, Random Forest, Gradient Boosting) on features.csv.
Exports model artifacts and model_metrics.json and predictions.csv.

Usage:
    python src/train_model.py --features src/analysis_outputs/features.csv --out-dir src/analysis_outputs --seed 42
"""
from __future__ import annotations
import argparse
import os
import json
import joblib
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# Label inference (patched)
# -------------------------
def infer_labels_with_heuristics(df: pd.DataFrame, min_positive_frac: float = 0.001) -> pd.Series:
    """
    Infer binary labels when an explicit 'label' column is not present.

    Heuristics (in order):
      1. If 'label' present, use it.
      2. Use 'breaking_change', 'is_remove' if present (logical OR).
      3. Use numeric 'pr_risk' >= 0.5 if present.
      4. Use dependency signals (shared_schemas_count > 0 or calls_services_count > 0).
      5. Use textual signals in 'detail' (keywords: required, removed, deprecated, breaking, error).
      6. If still too few positives, mark top-k by detail_len where k = max(5, int(min_positive_frac * n_rows))
    """
    if "label" in df.columns:
        return df["label"].astype(int)

    n = len(df)
    y = pd.Series(0, index=df.index, dtype=int)

    # 2: breaking_change
    if "breaking_change" in df.columns:
        try:
            y = y | df["breaking_change"].fillna(0).astype(int)
        except Exception:
            pass

    # 3: is_remove
    if "is_remove" in df.columns:
        try:
            y = y | df["is_remove"].fillna(0).astype(int)
        except Exception:
            pass

    # 4: pr_risk
    if "pr_risk" in df.columns:
        try:
            y = y | (df["pr_risk"].astype(float) >= 0.5).astype(int)
        except Exception:
            pass

    # 5: dependency signals
    heur = pd.Series(0, index=df.index, dtype=int)
    if "shared_schemas_count" in df.columns:
        heur = heur | (df["shared_schemas_count"].fillna(0).astype(int) > 0).astype(int)
    if "calls_services_count" in df.columns:
        heur = heur | (df["calls_services_count"].fillna(0).astype(int) > 0).astype(int)

    y = y | heur

    # 6: textual cues
    if "detail" in df.columns:
        small = df["detail"].fillna("").astype(str).str.lower()
        keywords = ["required", "removed", "remove", "deprecated", "breaking", "break", "error", "fail"]
        text_mask = small.apply(lambda s: any(k in s for k in keywords))
        y = y | text_mask.astype(int)

    # 7: fallback top-k by detail_len (proportional)
    if y.sum() == 0:
        min_k = 5
        frac_k = int(max(min_k, np.ceil(min_positive_frac * max(1, n))))
        k = min(frac_k, n)
        topk_idx = df["detail_len"].fillna(0).nlargest(k).index
        y.loc[topk_idx] = 1

    # final guard: ensure dtype int
    return y.astype(int)

# -------------------------
# Model training helpers
# -------------------------
def fit_and_eval(model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        try:
            raw = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-raw))
        except Exception:
            probs = np.zeros(len(X_test))
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    except Exception:
        roc = float("nan")
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(pr_rec, pr_prec) if len(pr_rec) > 0 else float("nan")
    brier = brier_score_loss(y_test, probs)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }, probs, float(t1 - t0)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="./src/analysis_outputs/features.csv")
    parser.add_argument("--out-dir", default="./src/analysis_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-positive-frac", type=float, default=0.001,
                        help="Minimum fraction of positives to synthesise if none found")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    # Ensure folders exist
    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load features
    df = pd.read_csv(args.features)

    # Select numeric features only
    exclude = {"ace_id", "pair_id", "type", "method", "path", "detail", "raw_type",
               "project", "old_version", "new_version"}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "biuf"]
    if not feat_cols:
        raise SystemExit("No numeric feature columns detected. Check features.csv")

    X_all = df[feat_cols].fillna(0)

    # Infer labels
    y = infer_labels_with_heuristics(df, min_positive_frac=float(args.min_positive_frac))

    print("Label classes found:", sorted(list(y.unique().astype(int))),
          "counts:", y.value_counts().to_dict())

    # Split data
    stratify = y if len(set(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.25, random_state=int(args.seed), stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.25, random_state=int(args.seed)
        )

    # Fallback if only one class
    if len(set(y_train)) < 2:
        print("Only one label class available. Using Dummy baseline.")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)

        probs = dummy.predict_proba(X_test)[:, 1]
        preds = dummy.predict(X_test)

        preds_df = pd.DataFrame({
            "ace_id": df.loc[X_test.index, "ace_id"].values,
            "prob": probs,
            "pred": preds,
            "model": "DummyBaseline",
        })

        preds_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }

        out = {
            "version": "0.1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": int(args.seed),
            "models": {"DummyBaseline": metrics},
            "calibration": {},
        }

        with open(os.path.join(args.out_dir, "model_metrics.json"), "w") as fh:
            json.dump(out, fh, indent=2)

        print("Wrote Dummy baseline. Exiting.")
        return

    # Define models
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=int(args.seed)))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=int(args.seed)),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=int(args.seed)),
    }

    results = {}
    all_predictions = []

    # --------------------------
    # PR CURVE SETUP
    # --------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, model in models.items():
        print(f"Training {name}...")

        metrics, probs, elapsed = fit_and_eval(
            model, X_train, X_test, y_train, y_test
        )

        # Save model as .pkl
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)

        # Save predictions for this model
        preds_df = pd.DataFrame({
            "ace_id": df.loc[X_test.index, "ace_id"].values,
            "prob": probs,
            "pred": (probs >= 0.5).astype(int),
            "model": name,
        })
        all_predictions.append(preds_df)

        # Save PR curve as REAL PNG
        try:
            # Correct PR curve generation inside the model loop

            pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(pr_rec, pr_prec, label=f"{name}")

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            # Here is the important fix
            ax.set_title(f"Precision Recall Curve for {name}")

            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.legend(loc="best")

            png_path = os.path.join(args.out_dir, f"pr_{name}.png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved PR curve at {png_path}")

        except Exception as e:
            print(f"Failed to generate PR curve for {name}: {e}")

        metrics["train_time_seconds"] = float(elapsed)
        metrics["model_path"] = model_path
        results[name] = metrics

    # Combine prediction tables
    preds_all = pd.concat(all_predictions, ignore_index=True)
    preds_all.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    # --------------------------
    # Calibration for Logistic Regression
    # --------------------------
    try:
        lr_probs = preds_all[preds_all["model"] == "LogisticRegression"]["prob"].values
        lr_ace_ids = preds_all[preds_all["model"] == "LogisticRegression"]["ace_id"].values

        y_test_map = pd.Series(y_test.values,
                               index=df.loc[X_test.index, "ace_id"].values)

        y_cal = y_test_map.loc[lr_ace_ids].values

        bins = np.linspace(0, 1, 11)
        inds = np.digitize(lr_probs, bins) - 1

        calibration = {"bins": [], "predicted": [], "observed": []}
        for i in range(len(bins) - 1):
            mask = inds == i
            calibration["bins"].append([float(bins[i]), float(bins[i+1])])
            if mask.sum() == 0:
                calibration["predicted"].append(0.0)
                calibration["observed"].append(0.0)
            else:
                calibration["predicted"].append(float(np.mean(lr_probs[mask])))
                calibration["observed"].append(float(np.mean(y_cal[mask])))

    except Exception:
        calibration = {"bins": [], "predicted": [], "observed": []}

    # --------------------------
    # Write metrics.json
    # --------------------------
    out = {
        "version": "0.1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "models": results,
        "calibration": calibration,
    }

    with open(os.path.join(args.out_dir, "model_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("Training complete.")
    print("Saved metrics, models, predictions, and PR curves.")


if __name__ == "__main__":
    main()
