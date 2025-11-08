# Social Media Sentiment Analysis

End-to-end pipeline for measuring public sentiment toward a product using social media datasets. The project covers ingestion, preprocessing, modeling, analytics, and interactive visualization.

## Features

- Ingest local CSV or Parquet datasets and optional hashtag scraping via `snscrape` and `instaloader`.
- Clean and normalize posts with lemmatization, stopword removal, and tokenization.
- Run sentiment classification with VADER or transformer models from Hugging Face.
- Compute accuracy, precision, recall, macro-F1, and aggregated trend metrics.
- Visualize distributions, temporal trends, and topic breakdowns with Plotly and Streamlit.
- Launch a Streamlit dashboard for exploratory analysis and PDF-ready insights.

## Project Structure

```
project_root/
├── agent.md
├── app/
│   └── streamlit_app.py
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_visualization.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── src/
    ├── __init__.py
    ├── evaluate.py
    ├── ingest.py
    ├── model_infer.py
    ├── pipeline.py
    ├── preprocess.py
    └── visualize.py
```

## Setup

1. Create a virtual environment running Python 3.10 or newer.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK assets inside the environment if prompted (punkt, stopwords, wordnet, vader_lexicon).

## Data Expectations

Input datasets should include at minimum:

| Column       | Purpose                                |
|--------------|----------------------------------------|
| `id`         | Unique identifier for each post         |
| `text`       | Raw text content                        |
| `created_at` | ISO-formatted timestamp (optional)      |
| `topic`      | Product or cluster label (optional)     |
| `label`      | Ground truth sentiment (optional)       |

Additional metadata is preserved throughout the pipeline.

## Running the Pipeline

Execute the CLI pipeline to preprocess data, score sentiment, and export results:

```bash
python -m src.pipeline data/raw/posts.csv outputs/scored_posts.csv --mode vader --text-column text --label-column label --timestamp-column created_at
```

Outputs:

- `outputs/scored_posts.csv`: processed dataset with sentiment labels and confidence scores.
- `outputs/scored_posts_trend.csv`: optional daily aggregation when timestamps are provided.

## Streamlit Dashboard

Launch the interactive dashboard once dependencies are installed:

```bash
streamlit run app/streamlit_app.py
```

- Upload a CSV or Parquet dataset or toggle the sample loader.
- Review sentiment distribution, temporal trends, and topic breakdowns.
- Export the scored dataset directly from the interface.

## Notebooks

- `01_data_cleaning.ipynb`: exploratory data cleaning and augmentation.
- `02_model_training.ipynb`: baseline and transformer experiments.
- `03_visualization.ipynb`: prototype charting and narrative insights.

## Evaluation Metrics

Use `src.evaluate` to compute:

- Accuracy, Precision, Recall, Macro F1
- Confusion matrix
- Time-based sentiment share
- Topic-level sentiment share

## License

Project intended for academic capstone use. Ensure compliance with platform terms of service when collecting data.
