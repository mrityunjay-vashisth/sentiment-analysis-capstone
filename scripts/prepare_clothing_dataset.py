#!/usr/bin/env python3
"""
Download and transform Women's Clothing Reviews dataset from Kaggle.

Dataset: nicapotato/womens-ecommerce-clothing-reviews
Expected output schema: id, text, created_at, topic, label
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import subprocess
import sys


def download_clothing_dataset(output_dir: str = "data/raw") -> str:
    """Download the Women's Clothing Reviews dataset from Kaggle."""
    print("=" * 70)
    print("DOWNLOADING WOMEN'S CLOTHING REVIEWS DATASET")
    print("=" * 70)

    dataset_name = "nicapotato/womens-ecommerce-clothing-reviews"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {dataset_name}...")
    print(f"Output directory: {output_path}")

    try:
        # Download using kaggle CLI
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_path), "--unzip"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Download complete!")

        # Find the CSV file
        csv_files = list(output_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {output_path}")

        csv_path = csv_files[0]
        print(f"✓ Dataset file: {csv_path}")
        return str(csv_path)

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}")
        raise
    except FileNotFoundError:
        print("\nError: Kaggle CLI not found!")
        print("Please ensure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up credentials in ~/.kaggle/kaggle.json")
        raise


def transform_clothing_reviews(
    input_path: str,
    output_path: str
) -> None:
    """Transform the Women's Clothing Reviews dataset to match project schema."""
    print("\n" + "=" * 70)
    print("TRANSFORMING DATASET")
    print("=" * 70)

    # Load the dataset
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    print("\nOriginal columns:")
    for col in df.columns:
        print(f"  - {col}")

    # Display sample
    print("\nSample row:")
    print(df.head(1).T)

    # Map ratings to sentiment labels
    # 1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive
    def rating_to_sentiment(rating):
        if pd.isna(rating):
            return None
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:  # 4-5
            return 'positive'

    # Combine Title and Review Text for better context
    def combine_text(row):
        title = str(row.get('Title', '')) if pd.notna(row.get('Title')) else ''
        review = str(row.get('Review Text', '')) if pd.notna(row.get('Review Text')) else ''

        # If both exist, combine them
        if title and review:
            return f"{title}. {review}"
        # Otherwise return whichever exists
        return title or review or ''

    print("\n" + "=" * 70)
    print("Creating transformed dataset...")

    # Create transformed dataframe
    transformed = pd.DataFrame()
    transformed['id'] = ['clothing_' + str(i) for i in range(len(df))]
    transformed['text'] = df.apply(combine_text, axis=1)
    transformed['created_at'] = None  # Will generate timestamps

    # Use Department Name or Division Name as topic, fallback to 'clothing'
    if 'Department Name' in df.columns:
        transformed['topic'] = df['Department Name'].fillna('clothing')
    elif 'Division Name' in df.columns:
        transformed['topic'] = df['Division Name'].fillna('clothing')
    else:
        transformed['topic'] = 'clothing'

    # Map rating to sentiment label
    transformed['label'] = df['Rating'].apply(rating_to_sentiment)

    # Add additional metadata columns for richer analysis
    if 'Clothing ID' in df.columns:
        transformed['product_id'] = df['Clothing ID']
    if 'Age' in df.columns:
        transformed['reviewer_age'] = df['Age']
    if 'Recommended IND' in df.columns:
        transformed['recommended'] = df['Recommended IND']

    # Generate synthetic timestamps (spread over 2024)
    print("Generating timestamps...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    np.random.seed(42)  # For reproducibility
    random_days = np.random.randint(0, date_range, size=len(transformed))
    transformed['created_at'] = [start_date + timedelta(days=int(d)) for d in random_days]

    # Remove rows with empty text or missing labels
    before_drop = len(transformed)
    transformed = transformed[transformed['text'].str.strip() != '']
    transformed = transformed.dropna(subset=['label'])
    after_drop = len(transformed)

    if before_drop > after_drop:
        print(f"\nRemoved {before_drop - after_drop} rows with missing text/labels")

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(transformed):,}")
    print(f"Columns: {transformed.columns.tolist()}")

    print("\nSentiment distribution:")
    sentiment_counts = transformed['label'].value_counts().sort_index()
    for label, count in sentiment_counts.items():
        percentage = (count / len(transformed)) * 100
        print(f"  {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print("\nTop 5 departments/topics:")
    topic_counts = transformed['topic'].value_counts().head(5)
    for topic, count in topic_counts.items():
        percentage = (count / len(transformed)) * 100
        print(f"  {topic}: {count:,} ({percentage:.1f}%)")

    # Save transformed dataset
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Saving datasets...")

    # Save main schema (id, text, created_at, topic, label)
    main_cols = ['id', 'text', 'created_at', 'topic', 'label']
    transformed[main_cols].to_csv(output_path, index=False)
    print(f"✓ CSV saved to: {output_path}")

    # Also save as parquet for efficiency
    parquet_path = output_path.replace('.csv', '.parquet')
    transformed[main_cols].to_parquet(parquet_path, index=False)
    print(f"✓ Parquet saved to: {parquet_path}")

    # Save extended version with all metadata
    extended_path = output_path.replace('.csv', '_extended.csv')
    transformed.to_csv(extended_path, index=False)
    print(f"✓ Extended CSV (with metadata) saved to: {extended_path}")

    # Print sample rows
    print("\n" + "=" * 70)
    print("Sample rows:")
    print("=" * 70)
    print(transformed[main_cols].head(3).to_string(max_colwidth=60))

    print("\n" + "=" * 70)
    print("✓ Dataset preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run VADER analysis:")
    print(f"   python3 -m src.pipeline {output_path} outputs/clothing_vader_scored.csv --mode vader")
    print("\n2. Evaluate results:")
    print("   python3 scripts/evaluate_results.py outputs/clothing_vader_scored.csv --model-name 'VADER-Clothing'")
    print("\n3. Launch dashboard:")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    # Download the dataset
    try:
        raw_csv = download_clothing_dataset()
    except Exception as e:
        print(f"\nFailed to download dataset: {e}")
        print("\nAlternatively, you can manually download from:")
        print("https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews")
        print("And place it in data/raw/")
        sys.exit(1)

    # Transform the dataset
    output_file = "data/processed/clothing_reviews.csv"
    transform_clothing_reviews(raw_csv, output_file)
