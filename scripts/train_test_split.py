#!/usr/bin/env python3
"""
Create proper train/test splits for sentiment analysis project.

This script splits the dataset into:
- Training set (80%)
- Testing set (20%)

Ensures stratified split to maintain class balance.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys


def create_train_test_split(
    input_path: str,
    output_dir: str = "data/splits",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[str, str]:
    """Create stratified train/test split of the dataset.

    Args:
        input_path: Path to the full dataset CSV
        output_dir: Directory to save train/test splits
        test_size: Proportion of data for test set (default 0.2 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_path, test_path)
    """
    print("=" * 70)
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from: {input_path}")
    df = pd.read_parquet(input_path) if input_path.endswith('.parquet') else pd.read_csv(input_path)
    print(f"Total records: {len(df):,}")

    # Check label distribution
    print("\nOriginal label distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Create stratified split
    print(f"\nSplitting dataset (train={1-test_size:.0%}, test={test_size:.0%})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Maintain class balance
    )

    print(f"\nTrain set: {len(train_df):,} records")
    print(f"Test set:  {len(test_df):,} records")

    # Verify label distribution in splits
    print("\nTrain set label distribution:")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    print("\nTest set label distribution:")
    test_label_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        pct = (count / len(test_df)) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output filenames based on input
    input_name = Path(input_path).stem
    train_path = output_path / f"{input_name}_train.csv"
    test_path = output_path / f"{input_name}_test.csv"

    print("\n" + "=" * 70)
    print("SAVING SPLITS")
    print("=" * 70)

    train_df.to_csv(train_path, index=False)
    print(f"✓ Train set saved to: {train_path}")

    test_df.to_csv(test_path, index=False)
    print(f"✓ Test set saved to: {test_path}")

    # Also save as parquet for efficiency
    train_parquet = output_path / f"{input_name}_train.parquet"
    test_parquet = output_path / f"{input_name}_test.parquet"

    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)
    print(f"✓ Train parquet saved to: {train_parquet}")
    print(f"✓ Test parquet saved to: {test_parquet}")

    # Save metadata
    metadata = {
        'original_dataset': input_path,
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'test_size': test_size,
        'random_state': random_state,
        'stratified': True
    }

    metadata_path = output_path / f"{input_name}_split_info.txt"
    with open(metadata_path, 'w') as f:
        f.write("Train/Test Split Information\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print("✓ SPLIT COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Train model on: {train_path}")
    print(f"2. Evaluate on:    {test_path}")
    print("=" * 70)

    return str(train_path), str(test_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create train/test split")
    parser.add_argument("input_path", help="Path to input dataset")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion (0-1)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    create_train_test_split(
        args.input_path,
        args.output_dir,
        args.test_size,
        args.random_state
    )
