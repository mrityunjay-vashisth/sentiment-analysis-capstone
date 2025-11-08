# 🧠 Social Media Sentiment Analysis — Design Document

## 1️⃣ Project Overview

**Title:** Social Media Analysis to Gauge Users' Sentiment Towards a Company's Product
**Goal:** Build an NLP + ML system that analyzes public sentiment from social media (Twitter/X, Instagram, or Kaggle datasets) to visualize brand perception over time.

---

## 2️⃣ System Architecture

### 🔹 Logical Flow

```
 ┌────────────────────────────────────────────┐
 │          Data Sources (Kaggle / APIs)      │
 │  • Twitter Airline Sentiment              │
 │  • Sentiment140                           │
 │  • Public brand hashtags from X/Instagram │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │          Data Preprocessing Layer          │
 │  • Cleaning (URLs, mentions, hashtags)    │
 │  • Normalization & tokenization           │
 │  • Lemmatization                          │
 │  • Stopword removal                       │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │             Sentiment Analysis Layer        │
 │  • Baseline: TextBlob / VADER              │
 │  • Advanced: BERT / RoBERTa via HF API     │
 │  • Output: Positive / Negative / Neutral   │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │              Analytics Layer               │
 │  • Aggregation by date/product/topic       │
 │  • Sentiment trend visualization           │
 │  • Event overlay (product launches, etc.)  │
 └────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────┐
 │              Visualization Layer           │
 │  • Streamlit Dashboard                    │
 │  • Plotly Graphs                          │
 │  • Insights Summary PDF                   │
 └────────────────────────────────────────────┘
```

---

## 3️⃣ Module Breakdown

### 🧩 1. Data Collection Module

* Input: Kaggle CSV or scraped data.
* Output: `/data/raw/` folder with CSVs.
* Tools: `pandas`, `snscrape`, `instaloader`.

### 🧩 2. Preprocessing Module

* Cleans raw text data.
* Removes stopwords, URLs, special characters.
* Tokenizes and lemmatizes text.
* Output: `/data/processed/cleaned.csv`.

### 🧩 3. Sentiment Classification Module

* Uses pretrained model (Hugging Face pipeline).
* Outputs `sentiment_label` (positive/negative/neutral).
* Optional: Fine-tune transformer.

### 🧩 4. Trend Analysis Module

* Groups sentiment by time and topic.
* Generates line charts and summary stats.

### 🧩 5. Visualization Module

* Streamlit dashboard with tabs:

  * Overall sentiment distribution.
  * Time-based trends.
  * Product/topic breakdown.

---

## 4️⃣ Data Design

### Dataset Schema

| Column          | Type     | Description                   |
| :-------------- | :------- | :---------------------------- |
| id              | string   | Unique identifier             |
| platform        | string   | Twitter/Instagram/Kaggle      |
| created_at      | datetime | Post timestamp                |
| user            | string   | Hashed username               |
| text            | string   | Original post text            |
| cleaned_text    | string   | Processed text                |
| sentiment_label | string   | Positive / Negative / Neutral |
| confidence      | float    | Model confidence score        |
| topic           | string   | Cluster label / product       |

---

## 5️⃣ Technology Stack

| Layer         | Tools                                       |
| :------------ | :------------------------------------------ |
| Language      | Python 3.10+                                |
| Data          | pandas, numpy                               |
| NLP           | nltk, spacy, transformers                   |
| Modeling      | scikit-learn, TextBlob, VADER, BERT/RoBERTa |
| Visualization | matplotlib, seaborn, plotly                 |
| Dashboard     | Streamlit                                   |
| Storage       | CSV / Parquet                               |

---

## 6️⃣ Directory Structure

```
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── model_infer.py
│   ├── evaluate.py
│   └── visualize.py
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_visualization.ipynb
├── app/
│   └── streamlit_app.py
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── README.md
```

---

## 7️⃣ Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Time-based sentiment correlation

Target: **≥80% macro F1-score** on test data.

---

## 8️⃣ Risks & Mitigations

| Risk            | Mitigation                                 |
| :-------------- | :----------------------------------------- |
| API limits      | Use Kaggle datasets for reproducibility    |
| Imbalanced data | Apply oversampling or class weights        |
| Sarcasm / slang | Note as model limitation                   |
| Privacy issues  | Remove usernames, only analyze public data |

---

## 9️⃣ Timeline

| Week | Tasks                             |
| :--- | :-------------------------------- |
| 1    | Data acquisition + cleaning       |
| 2    | Baseline sentiment analysis       |
| 3    | Transformer model setup           |
| 4    | Visualization & analytics         |
| 5    | Streamlit dashboard + evaluation  |
| 6    | Final testing & report submission |

---

## 🔟 Deliverables

1. Codebase with documentation.
2. Cleaned dataset (non-sensitive).
3. Visual dashboards and insights.
4. Final report (≤1 MB PDF).
5. README and design document (this file).

---

## ✅ Success Criteria

* Reproducible pipeline with consistent output.
* Model achieves ≥80% F1 macro.
* Dashboard provides meaningful, interpretable visual insights.
* Ethical and privacy-compliant handling of social media data.

---

**© 2025 — Capstone Project: Social Media Sentiment Analysis**
