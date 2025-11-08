"""Evaluation utilities for sentiment predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics


DEFAULT_LABELS = ["negative", "neutral", "positive"]


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    confusion_matrix: list[list[int]]


def classification_scores(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] | None = None,
) -> EvaluationResult:
    labels = list(labels or DEFAULT_LABELS)
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    recall = metrics.recall_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    f1_macro = metrics.f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    confusion = metrics.confusion_matrix(true_labels, predicted_labels, labels=labels)
    return EvaluationResult(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_macro=float(f1_macro),
        confusion_matrix=confusion.astype(int).tolist(),
    )


def classification_report(true_labels: Sequence[str], predicted_labels: Sequence[str], labels: Sequence[str] | None = None) -> pd.DataFrame:
    report = metrics.classification_report(true_labels, predicted_labels, labels=labels or DEFAULT_LABELS, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def aggregate_by_time(
    frame: pd.DataFrame,
    timestamp_column: str = "created_at",
    label_column: str = "sentiment_label",
    freq: str = "D",
) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise KeyError(f"Column {timestamp_column} not found in dataframe")
    if label_column not in frame.columns:
        raise KeyError(f"Column {label_column} not found in dataframe")
    data = frame.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
    data = data.dropna(subset=[timestamp_column])
    data.set_index(timestamp_column, inplace=True)
    grouped = data.groupby([pd.Grouper(freq=freq), label_column]).size().unstack(label_column, fill_value=0)
    totals = grouped.sum(axis=1)
    normalized = grouped.div(totals.replace(0, np.nan), axis=0)
    normalized = normalized.add_suffix("_share")
    result = grouped.join(normalized)
    result.reset_index(inplace=True)
    return result


def aggregate_by_topic(frame: pd.DataFrame, topic_column: str = "topic", label_column: str = "sentiment_label") -> pd.DataFrame:
    if topic_column not in frame.columns:
        raise KeyError(f"Column {topic_column} not found in dataframe")
    if label_column not in frame.columns:
        raise KeyError(f"Column {label_column} not found in dataframe")
    grouped = frame.groupby([topic_column, label_column]).size().reset_index(name="count")
    totals = grouped.groupby(topic_column)["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals.replace(0, np.nan)
    pivot = grouped.pivot(index=topic_column, columns=label_column, values="share").fillna(0)
    return pivot.reset_index()
