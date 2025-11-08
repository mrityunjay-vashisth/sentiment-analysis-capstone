#!/usr/bin/env python3
"""Detailed evaluation script for sentiment analysis results."""
from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import classification_scores


def evaluate_predictions(scored_csv: str, model_name: str) -> None:
    """Evaluate and display detailed metrics for predictions."""
    print("=" * 70)
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print("=" * 70)

    # Load scored data
    df = pd.read_csv(scored_csv)

    print(f"\nDataset size: {len(df):,} records")

    # Get ground truth and predictions
    y_true = df['label']
    y_pred = df['sentiment_label']

    # Calculate metrics
    metrics = classification_scores(y_true, y_pred)

    print("\n" + "-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"Accuracy:        {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print(f"Macro Precision: {metrics.precision:.4f}")
    print(f"Macro Recall:    {metrics.recall:.4f}")
    print(f"Macro F1-Score:  {metrics.f1_macro:.4f} ({metrics.f1_macro*100:.2f}%)")

    # Display confusion matrix
    import numpy as np
    cm = np.array(metrics.confusion_matrix)

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    # Per-class analysis
    print("\n" + "-" * 70)
    print("PER-CLASS ANALYSIS")
    print("-" * 70)

    labels = ['negative', 'neutral', 'positive']
    for i, label in enumerate(labels):
        actual_count = (y_true == label).sum()
        predicted_count = (y_pred == label).sum()
        correct = cm[i, i]
        accuracy_class = correct / actual_count if actual_count > 0 else 0

        print(f"\n{label.upper()}:")
        print(f"  Actual count:    {actual_count:,}")
        print(f"  Predicted count: {predicted_count:,}")
        print(f"  Correctly classified: {correct:,} ({accuracy_class*100:.2f}%)")

    # Sentiment distribution comparison
    print("\n" + "-" * 70)
    print("SENTIMENT DISTRIBUTION")
    print("-" * 70)

    actual_dist = y_true.value_counts(normalize=True).sort_index()
    predicted_dist = y_pred.value_counts(normalize=True).sort_index()

    print("\n{:<15} {:<20} {:<20}".format("Sentiment", "Actual %", "Predicted %"))
    print("-" * 55)
    for label in labels:
        actual_pct = actual_dist.get(label, 0) * 100
        pred_pct = predicted_dist.get(label, 0) * 100
        diff = pred_pct - actual_pct
        diff_str = f"({diff:+.2f}%)"
        print(f"{label:<15} {actual_pct:>6.2f}%          {pred_pct:>6.2f}% {diff_str:>12}")

    # Goal assessment
    print("\n" + "=" * 70)
    print("PROJECT GOAL ASSESSMENT")
    print("=" * 70)
    target_f1 = 0.80
    current_f1 = metrics.f1_macro
    gap = target_f1 - current_f1

    if current_f1 >= target_f1:
        status = "✓ ACHIEVED"
        print(f"\n{status}: Target F1-score of {target_f1:.0%} has been met!")
    else:
        status = "✗ NOT MET"
        print(f"\n{status}: Target F1-score of {target_f1:.0%} not yet reached")
        print(f"Current F1: {current_f1:.4f} ({current_f1*100:.2f}%)")
        print(f"Gap:        {gap:.4f} ({gap*100:.2f} percentage points)")
        print(f"Progress:   {(current_f1/target_f1)*100:.1f}% of target")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis results")
    parser.add_argument("scored_csv", help="Path to scored CSV file")
    parser.add_argument("--model-name", default="Model", help="Name of the model")

    args = parser.parse_args()
    evaluate_predictions(args.scored_csv, args.model_name)
