"""Command-line pipeline for end-to-end sentiment scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .ingest import load_local_dataset, save_dataset
from .model_infer import ModelConfig, SentimentAnalyzer
from .preprocess import preprocess_dataframe
from .evaluate import aggregate_by_time, classification_scores


def run_pipeline(
    input_path: Path,
    output_path: Path,
    mode: str = "vader",
    text_column: str = "text",
    label_column: Optional[str] = None,
    timestamp_column: Optional[str] = "created_at",
) -> pd.DataFrame:
    raw = load_local_dataset(input_path)
    processed = preprocess_dataframe(raw, text_column=text_column)
    analyzer = SentimentAnalyzer(ModelConfig(mode=mode))
    scored = analyzer.predict_dataframe(processed, text_column="cleaned_text", output_prefix="sentiment")
    save_dataset(scored, output_path, fmt="csv")
    if label_column and label_column in scored.columns:
        metrics = classification_scores(scored[label_column], scored["sentiment_label"])
        print(f"Macro F1: {metrics.f1_macro:.3f}")
    if timestamp_column and timestamp_column in scored.columns:
        trend = aggregate_by_time(scored, timestamp_column=timestamp_column, label_column="sentiment_label")
        trend_path = output_path.with_name(output_path.stem + "_trend.csv")
        save_dataset(trend, trend_path, fmt="csv")
    return scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the social media sentiment pipeline")
    parser.add_argument("input", type=Path, help="Path to the input dataset (CSV or Parquet)")
    parser.add_argument("output", type=Path, help="Path to save the scored dataset")
    parser.add_argument("--mode", choices=["vader", "transformer"], default="vader", help="Sentiment analysis backend")
    parser.add_argument("--text-column", default="text", help="Column containing raw text")
    parser.add_argument("--label-column", help="Optional ground truth label column for evaluation")
    parser.add_argument("--timestamp-column", default="created_at", help="Column containing timestamps for trend aggregation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        text_column=args.text_column,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
    )


if __name__ == "__main__":
    main()
