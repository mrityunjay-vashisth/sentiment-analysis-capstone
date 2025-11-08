"""Streamlit dashboard for social media sentiment analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import preprocess
from src.evaluate import aggregate_by_time, aggregate_by_topic
from src.model_infer import ModelConfig, SentimentAnalyzer
from src.visualize import sentiment_distribution, sentiment_trend, topic_breakdown


st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

st.title("Social Media Sentiment Analysis")
st.markdown(
    "Upload a cleaned dataset or use the preprocessing tools to generate insights on product perception across social media platforms."
)


@st.cache_data
def load_sample_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Unsupported sample data format")


def run_pipeline(frame: pd.DataFrame, use_transformer: bool) -> pd.DataFrame:
    processed = preprocess.preprocess_dataframe(frame)
    config = ModelConfig(mode="transformer" if use_transformer else "vader")
    analyzer = SentimentAnalyzer(config)
    scored = analyzer.predict_dataframe(processed, text_column="cleaned_text", output_prefix="sentiment")
    return scored


sample_option = st.sidebar.checkbox("Load sample data", value=False)
transformer_mode = st.sidebar.checkbox("Use transformer model", value=False)

uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
dataframe: pd.DataFrame | None = None

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        dataframe = pd.read_csv(uploaded)
    else:
        dataframe = pd.read_parquet(uploaded)
elif sample_option:
    sample_path = Path("data/processed/sample.csv")
    if sample_path.exists():
        dataframe = load_sample_data(sample_path)
    else:
        st.info("No sample dataset found in data/processed/sample.csv")

if dataframe is not None:
    st.subheader("Input Snapshot")
    st.dataframe(dataframe.head())
    with st.spinner("Running preprocessing and sentiment analysis..."):
        results = run_pipeline(dataframe, transformer_mode)
    st.subheader("Sentiment Overview")
    dist_fig = sentiment_distribution(results, label_column="sentiment_label")
    st.plotly_chart(dist_fig, use_container_width=True)
    if "created_at" in results.columns:
        trend_fig = sentiment_trend(results, date_column="created_at", label_column="sentiment_label")
        st.plotly_chart(trend_fig, use_container_width=True)
        time_summary = aggregate_by_time(results, timestamp_column="created_at", label_column="sentiment_label")
        st.dataframe(time_summary.tail())
    if "topic" in results.columns:
        topic_fig = topic_breakdown(results, topic_column="topic", label_column="sentiment_label")
        st.plotly_chart(topic_fig, use_container_width=True)
        topic_summary = aggregate_by_topic(results, topic_column="topic", label_column="sentiment_label")
        st.dataframe(topic_summary)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored dataset", csv, file_name="sentiment_scored.csv")
else:
    st.info("Upload a dataset to get started or enable the sample data option")
