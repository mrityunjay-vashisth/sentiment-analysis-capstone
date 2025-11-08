"""Sentiment analysis model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


SentimentLabel = Literal["positive", "negative", "neutral"]


@dataclass
class ModelConfig:
    mode: Literal["vader", "transformer"] = "vader"
    transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentAnalyzer:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._analyzer = None

    def _load_vader(self):
        if self._analyzer is None:
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                import nltk
            except ImportError as exc:
                raise ImportError("nltk is required for VADER sentiment analysis") from exc
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def _load_transformer(self):
        if self._analyzer is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError("transformers is required for pipeline sentiment analysis") from exc
            self._analyzer = pipeline("sentiment-analysis", model=self.config.transformer_model)
        return self._analyzer

    def predict_texts(self, texts: Sequence[str], batch_size: int = 32) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["sentiment_label", "confidence"])
        if self.config.mode == "vader":
            analyzer = self._load_vader()
            scores = [analyzer.polarity_scores(text) for text in texts]
            labels = [self._scores_to_label(score) for score in scores]
            confidences = [self._scores_to_confidence(score) for score in scores]
        elif self.config.mode == "transformer":
            analyzer = self._load_transformer()
            # Process in batches to avoid memory issues
            labels = []
            confidences = []
            texts_list = list(texts)
            total_batches = (len(texts_list) + batch_size - 1) // batch_size

            print(f"Processing {len(texts_list):,} texts in {total_batches:,} batches of {batch_size}...")
            for i in range(0, len(texts_list), batch_size):
                batch = texts_list[i:i + batch_size]
                batch_num = i // batch_size + 1
                if batch_num % 100 == 0 or batch_num == total_batches:
                    print(f"  Batch {batch_num:,}/{total_batches:,} ({(batch_num/total_batches)*100:.1f}%)")
                outputs = analyzer(batch)
                labels.extend([self._normalize_transformer_label(item["label"]) for item in outputs])
                confidences.extend([float(item["score"]) for item in outputs])
        else:
            raise ValueError(f"Unsupported mode {self.config.mode}")
        return pd.DataFrame({"sentiment_label": labels, "confidence": confidences})

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        text_column: str = "cleaned_text",
        output_prefix: str = "sentiment",
    ) -> pd.DataFrame:
        if text_column not in frame.columns:
            raise KeyError(f"Column {text_column} not found in dataframe")
        predictions = self.predict_texts(frame[text_column].fillna("").tolist())
        result = frame.copy()
        result[f"{output_prefix}_label"] = predictions["sentiment_label"].values
        result[f"{output_prefix}_confidence"] = predictions["confidence"].values
        return result

    @staticmethod
    def _scores_to_label(score: dict[str, float]) -> SentimentLabel:
        compound = score.get("compound", 0.0)
        if compound >= 0.2:
            return "positive"
        if compound <= -0.2:
            return "negative"
        return "neutral"

    @staticmethod
    def _scores_to_confidence(score: dict[str, float]) -> float:
        positive = score.get("pos", 0.0)
        negative = score.get("neg", 0.0)
        neutral = score.get("neu", 0.0)
        return float(np.max([positive, negative, neutral]))

    @staticmethod
    def _normalize_transformer_label(label: str) -> SentimentLabel:
        label_lower = label.lower()
        if "pos" in label_lower:
            return "positive"
        if "neg" in label_lower:
            return "negative"
        return "neutral"
