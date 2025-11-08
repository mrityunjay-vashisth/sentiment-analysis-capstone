#!/usr/bin/env python3
"""
Transform the downloaded Kaggle dataset to match the project schema.

Expected schema: id, text, created_at, topic, label
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


def transform_twitter_reddit_dataset(
    twitter_path: str,
    reddit_path: str,
    output_path: str
) -> None:
    """Transform the Twitter and Reddit datasets to match project schema."""
    print("=" * 60)
    print("Loading Twitter dataset...")
    twitter_df = pd.read_csv(twitter_path)
    print(f"Twitter shape: {twitter_df.shape}")

    print("\nLoading Reddit dataset...")
    reddit_df = pd.read_csv(reddit_path)
    print(f"Reddit shape: {reddit_df.shape}")

    # Map sentiment labels: -1 (negative), 0 (neutral), 1 (positive)
    sentiment_map = {
        -1: 'negative',
        0: 'neutral',
        1: 'positive'
    }

    # Transform Twitter data
    print("\n" + "=" * 60)
    print("Transforming Twitter data...")
    twitter_transformed = pd.DataFrame({
        'id': ['twitter_' + str(i) for i in range(len(twitter_df))],
        'text': twitter_df['clean_text'].astype(str),
        'created_at': None,  # Will generate timestamps
        'topic': 'politics',  # Based on dataset content (Modi-related tweets)
        'label': twitter_df['category'].map(sentiment_map),
        'platform': 'twitter'
    })

    # Transform Reddit data
    print("Transforming Reddit data...")
    reddit_transformed = pd.DataFrame({
        'id': ['reddit_' + str(i) for i in range(len(reddit_df))],
        'text': reddit_df['clean_comment'].astype(str),
        'created_at': None,  # Will generate timestamps
        'topic': 'buddhism',  # Based on dataset content (Buddhism discussions)
        'label': reddit_df['category'].map(sentiment_map),
        'platform': 'reddit'
    })

    # Combine both datasets
    print("\nCombining datasets...")
    combined = pd.concat([twitter_transformed, reddit_transformed], ignore_index=True)

    # Generate synthetic timestamps (spread over 2023)
    print("Generating timestamps...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days

    np.random.seed(42)  # For reproducibility
    random_days = np.random.randint(0, date_range, size=len(combined))
    combined['created_at'] = [start_date + timedelta(days=int(d)) for d in random_days]

    # Remove any rows with missing labels
    before_drop = len(combined)
    combined = combined.dropna(subset=['label'])
    after_drop = len(combined)

    if before_drop > after_drop:
        print(f"\nRemoved {before_drop - after_drop} rows with missing labels")

    # Reorder columns to match project schema
    combined = combined[['id', 'text', 'created_at', 'topic', 'label']]

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(combined):,}")
    print(f"Columns: {combined.columns.tolist()}")

    print("\nPlatform distribution:")
    platform_counts = twitter_transformed.shape[0], reddit_transformed.shape[0]
    print(f"  Twitter: {platform_counts[0]:,}")
    print(f"  Reddit:  {platform_counts[1]:,}")

    print("\nSentiment distribution:")
    sentiment_counts = combined['label'].value_counts().sort_index()
    for label, count in sentiment_counts.items():
        percentage = (count / len(combined)) * 100
        print(f"  {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print("\nTopic distribution:")
    topic_counts = combined['topic'].value_counts()
    for topic, count in topic_counts.items():
        percentage = (count / len(combined)) * 100
        print(f"  {topic.capitalize()}: {count:,} ({percentage:.1f}%)")

    # Save transformed dataset
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saving datasets...")
    combined.to_csv(output_path, index=False)
    print(f"✓ CSV saved to: {output_path}")

    # Also save as parquet for efficiency
    parquet_path = output_path.replace('.csv', '.parquet')
    combined.to_parquet(parquet_path, index=False)
    print(f"✓ Parquet saved to: {parquet_path}")

    # Print sample rows
    print("\n" + "=" * 60)
    print("Sample rows:")
    print("=" * 60)
    print(combined.head(5).to_string(max_colwidth=50))

    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    twitter_file = "data/raw/Twitter_Data.csv"
    reddit_file = "data/raw/Reddit_Data.csv"
    output_file = "data/processed/sentiments_clean.csv"

    transform_twitter_reddit_dataset(twitter_file, reddit_file, output_file)
