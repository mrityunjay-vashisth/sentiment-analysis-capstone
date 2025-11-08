"""Visualization utilities for sentiment analysis insights."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def sentiment_distribution(frame: pd.DataFrame, label_column: str = "sentiment_label") -> go.Figure:
    if label_column not in frame.columns:
        raise KeyError(f"Column {label_column} not found in dataframe")
    counts = frame[label_column].value_counts().reset_index()
    counts.columns = [label_column, "count"]
    figure = px.pie(counts, names=label_column, values="count", title="Sentiment Distribution")
    return figure


def sentiment_trend(frame: pd.DataFrame, date_column: str = "created_at", label_column: str = "sentiment_label", freq: str = "D") -> go.Figure:
    if date_column not in frame.columns:
        raise KeyError(f"Column {date_column} not found in dataframe")
    if label_column not in frame.columns:
        raise KeyError(f"Column {label_column} not found in dataframe")
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column])
    grouped = data.groupby([pd.Grouper(key=date_column, freq=freq), label_column]).size().reset_index(name="count")
    figure = px.line(grouped, x=date_column, y="count", color=label_column, title="Sentiment Trend Over Time")
    return figure


def topic_breakdown(frame: pd.DataFrame, topic_column: str = "topic", label_column: str = "sentiment_label") -> go.Figure:
    if topic_column not in frame.columns:
        raise KeyError(f"Column {topic_column} not found in dataframe")
    if label_column not in frame.columns:
        raise KeyError(f"Column {label_column} not found in dataframe")
    grouped = frame.groupby([topic_column, label_column]).size().reset_index(name="count")
    figure = px.bar(grouped, x=topic_column, y="count", color=label_column, title="Sentiment by Topic", barmode="stack")
    return figure


def export_figure(figure: go.Figure, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(str(output))
    return output
