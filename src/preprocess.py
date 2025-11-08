"""Text preprocessing utilities for sentiment analysis datasets."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as exc:
    raise ImportError("nltk is required for preprocessing") from exc


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")


def ensure_nltk_resources() -> None:
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab/english.pickle": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for locator, package in resources.items():
        try:
            nltk.data.find(locator)
        except LookupError:
            nltk.download(package, quiet=True)


def normalize_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = URL_PATTERN.sub("", text)
    cleaned = MENTION_PATTERN.sub("", cleaned)
    cleaned = HASHTAG_PATTERN.sub("", cleaned)
    cleaned = NON_ALPHANUMERIC_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def tokenize_and_lemmatize(text: str) -> str:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed)


def preprocess_series(texts: Iterable[str]) -> list[str]:
    ensure_nltk_resources()  # Call once for the entire series
    normalized = [normalize_text(text) for text in texts]
    lemmatized = [tokenize_and_lemmatize(text) for text in normalized]
    return lemmatized


def preprocess_dataframe(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    if text_column not in frame.columns:
        raise KeyError(f"Column {text_column} not found in dataframe")
    frame = frame.copy()
    frame["cleaned_text"] = preprocess_series(frame[text_column].fillna(""))
    return frame
