#!/usr/bin/env python3
"""
evaluate.py

Generate ROC, PR curves, calibration plots, and SHAP/permutation summaries.

Usage:
    python src/evaluate.py \
        --predictions src/analysis_outputs/predictions.csv \
        --features src/analysis_outputs/features.csv \
        --out-dir src/analysis_outputs
"""

from __future__ import annotations
import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

FIG_DPI = 150
FIG_SIZE = (6, 4)


def safe_name(s: str) -> str:
    """Make a filesystem-safe short name for models."""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "model"
    return s[:60]


def plot_roc(y_true, probs, out_path):
    """Safely plot ROC if both classes exist and probs are varied."""
    try:
        if len(set(y_true)) < 2:
            print(f"Skipping ROC ({out_path}): only one class present in y_true.")
            return
        if np.nanmax(probs) == np.nanmin(probs):
            print(f"Skipping ROC ({out_path}): predicted probabilities are constant.")
            return

        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved ROC:", out_path)
    except Exception as e:
        print("ROC error:", e)


def plot_pr(y_true, probs, out_path):
    """Safely plot Precision-Recall curve."""
    try:
        if len(set(y_true)) < 2:
            print(f"Skipping PR ({out_path}): only one class present in y_true.")
            return
        if np.nanmax(probs) == np.nanmin(probs):
            print(f"Skipping PR ({out_path}): predicted probabilities are constant.")
            return

        prec, rec, _ = precision_recall_curve(y_true, probs)
        # compute PR AUC using recall x precision ordering
        pr_auc = auc(rec, prec)
        ap = average_precision_score(y_true, probs)

        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}  AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved PR:", out_path)
    except Exception as e:
        print("PR error:", e)


def plot_calibration(y_true, probs, out_path, n_bins=10):
    """Optional calibration plot (reliability diagram)."""
    try:
        if len(set(y_true)) < 2:
            print(f"Skipping calibration ({out_path}): only one class present.")
            return
        if np.nanmax(probs) == np.nanmin(probs):
            print(f"Skipping calibration ({out_path}): predicted probabilities are constant.")
            return

        fraction_of_pos, mean_predicted_value = calibration_curve(y_true, probs, n_bins=n_bins)
        brier = brier_score_loss(y_true, probs)

        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        plt.plot(mean_predicted_value, fraction_of_pos, "s-", label=f"Brier = {brier:.3f}")
        plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
        plt.xlabel("Mean predicted value")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration plot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved calibration:", out_path)
    except Exception as e:
        print("Calibration error:", e)


def find_prob_column(df: pd.DataFrame):
    """Return the column name used for predicted probability/score."""
    candidates = ["prob", "proba", "probability", "score", "y_pred", "pred"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback numeric-only columns except label and ace_id and model
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("label", "ace_id")]
    return numeric_cols[0] if numeric_cols else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--merge-on", default="ace_id", help="column to merge predictions↔features (default: ace_id)")
    parser.add_argument("--model-col", default="model", help="column name for model id in predictions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading predictions:", args.predictions)
    preds = pd.read_csv(args.predictions, dtype=str, keep_default_na=False, na_values=[""])
    print("Loading features:", args.features)
    features = pd.read_csv(args.features, dtype=str, keep_default_na=False, na_values=[""])

    # Ensure merge key exists
    if args.merge_on not in preds.columns:
        raise SystemExit(f"Predictions missing merge key column '{args.merge_on}'")
    if args.merge_on not in features.columns:
        print(f"Warning: features missing '{args.merge_on}'. Continuing with best-effort label mapping.")
    else:
        # try to cast labels to ints
        if "label" in features.columns:
            features["label"] = pd.to_numeric(features["label"], errors="coerce").fillna(0).astype(int)
        else:
            print("Warning: 'label' not in features; you will get fallback or zeros unless predictions already include label.")

    # try to detect probability column in predictions
    prob_col = find_prob_column(pd.read_csv(args.predictions, nrows=10))
    if prob_col is None:
        raise SystemExit("Could not find a probability/score column in predictions.")
    print("Using probability/score column:", prob_col)

    # read preds again with correct types for numeric column
    preds = pd.read_csv(args.predictions)
    if prob_col not in preds.columns:
        preds[prob_col] = pd.to_numeric(preds[prob_col], errors="coerce").fillna(0.0)
    else:
        preds[prob_col] = pd.to_numeric(preds[prob_col], errors="coerce").fillna(0.0)

    # If features has labels and ace_id, merge
    if ("label" in features.columns) and (args.merge_on in features.columns):
        merged = preds.merge(features[[args.merge_on, "label"]], how="left", left_on=args.merge_on, right_on=args.merge_on, suffixes=("", "_f"))
        if merged["label"].isnull().any():
            print("Warning: some predictions could not be matched to labels after merge; filling missing labels with 0.")
            merged["label"] = merged["label"].fillna(0).astype(int)
    else:
        # fallback: use 'label' column in preds if present, else zeros
        if "label" in preds.columns:
            merged = preds.copy()
            merged["label"] = pd.to_numeric(merged["label"], errors="coerce").fillna(0).astype(int)
        else:
            merged = preds.copy()
            merged["label"] = 0
            print("Warning: No label available in features or predictions. All labels set to 0 (this will make ROC/PR meaningless).")

    models = merged[args.model_col].unique() if args.model_col in merged.columns else ["model"]
    print("Models in predictions:", models)

    for model in models:
        print("\nProcessing model:", model)
        sub = merged[merged[args.model_col] == model].copy() if args.model_col in merged.columns else merged.copy()

        # assemble y_true and probs
        y_true = pd.to_numeric(sub["label"], errors="coerce").fillna(0).astype(int).values
        probs = pd.to_numeric(sub[prob_col], errors="coerce").fillna(0.0).values

        safe = safe_name(model)
        roc_path = os.path.join(args.out_dir, f"roc_{safe}.png")
        pr_path = os.path.join(args.out_dir, f"pr_{safe}.png")
        cal_path = os.path.join(args.out_dir, f"calib_{safe}.png")

        plot_roc(y_true, probs, roc_path)
        plot_pr(y_true, probs, pr_path)
        plot_calibration(y_true, probs, cal_path)

        # Basic numeric summaries
        try:
            print(f"Model '{model}': n={len(y_true)}, positive_rate={y_true.mean():.3f}, prob_mean={np.nanmean(probs):.3f}, prob_std={np.nanstd(probs):.3f}")
        except Exception:
            pass

    # SHAP/permutation importance handling
    shap_txt = os.path.join(args.out_dir, "shap_summary.txt")
    shap_json = os.path.join(args.out_dir, "shap_summary.json")

    if os.path.exists(shap_txt):
        print("Found shap_summary.txt → converting to JSON")
        try:
            with open(shap_txt, "r", encoding="utf-8") as fh:
                text = fh.read()

            try:
                data = json.loads(text)
            except Exception:
                data = {"text_summary": text}

            with open(shap_json, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

            print("Saved shap_summary.json")
        except Exception as e:
            print("SHAP conversion error:", e)
    else:
        print("No shap_summary.txt found; skipping SHAP export.")

    print("\nEvaluation complete.")
