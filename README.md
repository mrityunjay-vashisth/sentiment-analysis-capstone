# Social Media Sentiment Analysis

[![View EDA Notebook](https://img.shields.io/badge/View-EDA%20Notebook-orange?logo=jupyter)](https://nbviewer.org/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/01_Exploratory_Data_Analysis_executed.ipynb)
[![View Results Notebook](https://img.shields.io/badge/View-Results%20Analysis-blue?logo=jupyter)](https://nbviewer.org/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/02_Model_Results_Analysis_executed.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/01_Exploratory_Data_Analysis.ipynb)

End-to-end pipeline for measuring public sentiment toward a product using social media datasets. The project covers ingestion, preprocessing, modeling, analytics, and interactive visualization.

## 📊 Interactive Notebooks

**✏️ Want to edit and run the notebooks?** See [EDITING_GUIDE.md](EDITING_GUIDE.md) for detailed instructions!

### Quick Access Options:

| Option | EDA Notebook | Results Notebook | Best For |
|--------|-------------|------------------|----------|
| **🚀 Run in Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/01_Exploratory_Data_Analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/02_Model_Results_Analysis.ipynb) | **Editing & running** |
| **📖 View (nbviewer)** | [View EDA](https://nbviewer.org/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/01_Exploratory_Data_Analysis_executed.ipynb) | [View Results](https://nbviewer.org/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/02_Model_Results_Analysis_executed.ipynb) | **Reading only** |
| **📁 GitHub** | [Open on GitHub](notebooks/01_Exploratory_Data_Analysis_executed.ipynb) | [Open on GitHub](notebooks/02_Model_Results_Analysis_executed.ipynb) | **Browsing code** |

### What's in the Notebooks:

- **01_Exploratory_Data_Analysis.ipynb**:
  - Dataset overview and statistics
  - Class distribution analysis (balanced vs imbalanced)
  - Text length analysis by sentiment
  - Word clouds for each sentiment class
  - Train/test split verification

- **02_Model_Results_Analysis.ipynb**:
  - VADER model performance evaluation
  - Enhanced confusion matrices with percentages
  - Per-class metrics (Precision, Recall, F1-Score)
  - Model comparison across datasets
  - Confidence score distributions
  - Error analysis and insights

## 📈 Project Highlights

- **200K+ Social Media Posts** analyzed with VADER sentiment analysis
- **22K Clothing Reviews** for domain comparison
- **15+ Publication-Quality Visualizations** (300 DPI)
- **Comprehensive EDA** with word clouds and distribution analysis
- **Proper Train/Test Splits** (80/20 stratified)
- **Multi-Dataset Evaluation** showing domain adaptation challenges

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

### For Editing Notebooks (Local Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone.git
   cd sentiment-analysis-capstone
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   # Navigate to notebooks/ folder and open any .ipynb file
   ```

5. **Download required NLTK assets if prompted:**
   - punkt, stopwords, wordnet, vader_lexicon

**📖 For detailed editing instructions with multiple options (Colab, VS Code, etc.), see [EDITING_GUIDE.md](EDITING_GUIDE.md)**

### For Running the Pipeline Only

If you just want to run the sentiment analysis pipeline without editing:

```bash
pip install -r requirements.txt
```

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

**Comprehensive Analysis Notebooks (With Full Outputs):**
- **`01_Exploratory_Data_Analysis_executed.ipynb`**: Complete EDA with 12+ visualizations including:
  - Class distribution comparison (balanced vs imbalanced datasets)
  - Text length analysis by sentiment
  - Word clouds for each sentiment class (negative, neutral, positive)
  - Train/test split verification charts

- **`02_Model_Results_Analysis_executed.ipynb`**: Full results analysis with:
  - Enhanced confusion matrices with percentages
  - Per-class performance metrics (Precision, Recall, F1-Score)
  - Model comparison across datasets
  - Progress visualization toward target metrics
  - Confidence score distributions
  - Detailed error analysis

**Legacy Notebooks:**
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
