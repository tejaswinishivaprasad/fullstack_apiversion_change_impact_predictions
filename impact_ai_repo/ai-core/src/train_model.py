#!/usr/bin/env python3
"""
train_model.py

Train models (Logistic Regression, Random Forest, Gradient Boosting) on features.csv.
Exports model artifacts and model_metrics.json.

Usage:
    python src/train_model.py --features src/analysis_outputs/features.csv --out-dir src/analysis_outputs --seed 42
"""
from __future__ import annotations
import argparse
import os
import json
import joblib
import time
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


def infer_labels_with_heuristics(df: pd.DataFrame) -> pd.Series:
    """
    Infer labels when an explicit 'label' column is not present.

    Heuristics:
      - Prefer explicit 'label' column.
      - Use 'breaking_change' or 'is_remove' if present.
      - Use 'pr_risk' >= 0.5 if available.
      - If still no positives, mark rows with shared_schemas_count > 0 or calls_services_count > 0 as positive.
      - If still zero positives, mark top-k by detail_len as positives (k = min(5, n)).
    """
    if "label" in df.columns:
        return df["label"].astype(int)

    # start with zeros
    y = pd.Series(0, index=df.index, dtype=int)

    if "breaking_change" in df.columns:
        try:
            y = df["breaking_change"].fillna(0).astype(int)
        except Exception:
            y = df["breaking_change"].fillna(0).astype(int)

    # use is_remove as additional positive signal
    if "is_remove" in df.columns:
        try:
            y = y | df["is_remove"].fillna(0).astype(int)
        except Exception:
            y = y

    # use any PR risk field if available
    if "pr_risk" in df.columns:
        try:
            y = y | (df["pr_risk"].astype(float) >= 0.5).astype(int)
        except Exception:
            pass

    # If still all zeros, apply structural heuristics
    if y.sum() == 0:
        heur = pd.Series(0, index=df.index, dtype=int)
        if "shared_schemas_count" in df.columns:
            heur = heur | (df["shared_schemas_count"].fillna(0).astype(int) > 0).astype(int)
        if "calls_services_count" in df.columns:
            heur = heur | (df["calls_services_count"].fillna(0).astype(int) > 0).astype(int)
        if "breaking_change" in df.columns:
            heur = heur | df["breaking_change"].fillna(0).astype(int)
        if "is_remove" in df.columns:
            heur = heur | df["is_remove"].fillna(0).astype(int)

        # fallback: pick top-k by detail length if heuristics still empty
        if heur.sum() == 0 and "detail_len" in df.columns:
            k = min(5, len(df))
            topk = df["detail_len"].fillna(0).nlargest(k).index
            heur.loc[topk] = 1

        y = y | heur

    return y.astype(int)


def fit_and_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision function where available, scale to (0,1)
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
    pr_auc = auc(pr_rec, pr_prec)
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
    }, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="./src/analysis_outputs/features.csv")
    parser.add_argument("--out-dir", default="./src/analysis_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(args.features)

    # Determine features (exclude identifiers and text fields)
    exclude = ["ace_id", "pair_id", "type", "method", "path", "detail", "raw_type"]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "biuf"]
    if not feat_cols:
        raise SystemExit("No numeric feature columns detected. Check features.csv")

    X_all = df[feat_cols].fillna(0)

    # infer labels robustly
    y = infer_labels_with_heuristics(df)

    unique_classes = sorted(list(y.unique().astype(int)))
    print("Label classes found (post-infer):", unique_classes, "counts:", y.value_counts().to_dict())

    # If only one class, we will attempt to create train/test then fallback to DummyClassifier if still single-class.
    stratify = y if len(set(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.25, random_state=int(args.seed), stratify=stratify
        )
    except Exception:
        # fallback: simple non-stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.25, random_state=int(args.seed)
        )

    # Re-check class presence in train set
    train_classes = sorted(list(y_train.unique().astype(int)))
    if len(train_classes) < 2:
        print("Only one label class available in training data. Falling back to DummyClassifier baseline.")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        if hasattr(dummy, "predict_proba"):
            probs = dummy.predict_proba(X_test)[:, 1]
        else:
            probs = dummy.predict(X_test)
        preds = dummy.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            roc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
        except Exception:
            roc = float("nan")
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(pr_rec, pr_prec)
        brier = brier_score_loss(y_test, probs)
        results = {
            "DummyBaseline": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(roc),
                "pr_auc": float(pr_auc),
                "brier_score": float(brier),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
            }
        }
        metrics_out = {
            "version": "0.1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": int(args.seed),
            "models": results,
            "calibration": {},
        }
        metrics_path = os.path.join(args.out_dir, "model_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_out, fh, indent=2)
        print("Wrote Dummy baseline metrics to", metrics_path)
        preds_df = pd.DataFrame(
            {
                "ace_id": df.loc[X_test.index, "ace_id"].values,
                "prob": probs,
                "pred": preds,
                "model": "DummyBaseline",
            }
        )
        preds_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
        raise SystemExit("Trained DummyClassifier baseline because training data contains a single label class.")

    # Normal training path: train multiple models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=int(args.seed)),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=int(args.seed)),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=int(args.seed)),
    }

    results = {}
    all_predictions = []

    for name, m in models.items():
        t0 = time.time()
        metrics, probs = fit_and_eval(m, X_train, X_test, y_train, y_test)
        t1 = time.time()
        # save model - fit_and_eval already trained the model via model.fit inside it
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(m, model_path)
        preds_df = pd.DataFrame(
            {
                "ace_id": df.loc[X_test.index, "ace_id"].values,
                "prob": probs,
                "pred": (probs >= 0.5).astype(int),
                "model": name,
            }
        )
        all_predictions.append(preds_df)
        metrics["train_time_seconds"] = float(t1 - t0)
        results[name] = metrics
        results[name]["model_path"] = model_path

    # combine predictions and write files
    preds_all = pd.concat(all_predictions, ignore_index=True)
    preds_path = os.path.join(args.out_dir, "predictions.csv")
    preds_all.to_csv(preds_path, index=False)

    # calibration: compute based on logistic regression predictions if available
    try:
        probs_lr = preds_all[preds_all["model"] == "LogisticRegression"]["prob"].values
        # align y_test to logistic subset
        mask_lr = preds_all[preds_all["model"] == "LogisticRegression"].index
        # approximate calibration using y_test (we used same split ordering so alignment holds by index)
        # build calibration bins
        calib = {"bins": [], "predicted": [], "observed": []}
        if len(probs_lr) > 0:
            # fetch corresponding y_test values using index positions
            y_test_vals = y_test.values
            bins = np.linspace(0.0, 1.0, 11)
            inds = np.digitize(probs_lr, bins) - 1
            for i in range(len(bins) - 1):
                mask = inds == i
                if mask.sum() == 0:
                    calib["bins"].append([float(bins[i]), float(bins[i + 1])])
                    calib["predicted"].append(0.0)
                    calib["observed"].append(0.0)
                else:
                    calib["bins"].append([float(bins[i]), float(bins[i + 1])])
                    calib["predicted"].append(float(np.mean(probs_lr[mask])))
                    # best-effort observed: if lengths match, align, else use global mean
                    try:
                        calib["observed"].append(float(np.mean(y_test_vals[mask])))
                    except Exception:
                        calib["observed"].append(float(np.mean(y_test_vals)))
        else:
            calib = {"bins": [], "predicted": [], "observed": []}
    except Exception:
        calib = {"bins": [], "predicted": [], "observed": []}

    metrics_out = {
        "version": "0.1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "models": results,
        "calibration": calib,
    }
    metrics_path = os.path.join(args.out_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2)

    print("Wrote metrics to", metrics_path)
    print("Wrote predictions to", preds_path)
    print("Saved models to", models_dir)


if __name__ == "__main__":
    main()
