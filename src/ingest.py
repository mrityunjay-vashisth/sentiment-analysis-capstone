"""Data ingestion utilities for social media sentiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd


@dataclass
class DataIngestionConfig:
    raw_data_dir: Path
    processed_data_dir: Path

    def ensure_directories(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


def load_local_dataset(path: str | Path, platform: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    if platform and "platform" not in frame.columns:
        frame["platform"] = platform
    if "id" not in frame.columns:
        frame["id"] = frame.index.astype(str)
    return frame


def merge_datasets(datasets: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in datasets if not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="id", keep="last")
    return merged


def save_dataset(frame: pd.DataFrame, output_path: str | Path, fmt: str = "csv") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_lower = fmt.lower()
    if format_lower == "csv":
        frame.to_csv(output_path, index=False)
    elif format_lower == "parquet":
        frame.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported save format {fmt}")
    return output_path


def read_from_directory(directory: str | Path, pattern: str = "*.csv") -> Iterator[pd.DataFrame]:
    directory = Path(directory)
    for path in sorted(directory.glob(pattern)):
        yield load_local_dataset(path)


def scrape_twitter_hashtag(hashtag: str, limit: int = 200) -> pd.DataFrame:
    try:
        from snscrape.modules import twitter
    except ImportError as exc:
        raise ImportError("snscrape is required for Twitter scraping") from exc

    tweets: list[dict[str, object]] = []
    query = f"#{hashtag}"
    scraper = twitter.TwitterHashtagScraper(query)
    for idx, tweet in enumerate(scraper.get_items()):
        if idx >= limit:
            break
        tweets.append(
            {
                "id": str(tweet.id),
                "platform": "twitter",
                "created_at": tweet.date,
                "user": str(tweet.user.username),
                "text": tweet.content,
            }
        )
    return pd.DataFrame(tweets)


def scrape_instagram_hashtag(hashtag: str, limit: int = 200) -> pd.DataFrame:
    try:
        import instaloader
    except ImportError as exc:
        raise ImportError("instaloader is required for Instagram scraping") from exc

    loader = instaloader.Instaloader()
    posts = instaloader.Hashtag.from_name(loader.context, hashtag).get_posts()
    records: list[dict[str, object]] = []
    for idx, post in enumerate(posts):
        if idx >= limit:
            break
        records.append(
            {
                "id": str(post.mediaid),
                "platform": "instagram",
                "created_at": post.date_utc,
                "user": hash(post.owner_username),
                "text": post.caption or "",
            }
        )
    return pd.DataFrame(records)
