# GitHub Codespaces Setup

This guide will help you run the sentiment analysis pipeline on GitHub Codespaces with better compute resources.

## Quick Start

### 1. Push to GitHub

```bash
# On your local machine (already done)
git add .
git commit -m "Initial commit"
gh repo create kirtiproject --public --source=. --remote=origin --push
```

### 2. Open in Codespaces

1. Go to your GitHub repository
2. Click the green "Code" button
3. Select "Codespaces" tab
4. Click "Create codespace on main"

**Recommended machine type:** 4-core (for faster processing)

### 3. Set Up Kaggle Credentials

Once in Codespaces:

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Upload your kaggle.json file to the Codespaces environment
# You can drag & drop it or use the file upload feature

# Move it to the right location
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download Dataset

```bash
python scripts/prepare_dataset.py
```

This will:
- Download 200K+ Twitter and Reddit posts from Kaggle
- Transform them to match the project schema
- Save to `data/processed/`

### 5. Run Transformer Analysis

Option A: **Using Jupyter Notebook** (Recommended)

1. Open `notebooks/run_transformer_analysis.ipynb`
2. Run all cells
3. See results inline with visualizations

Option B: **Using CLI**

```bash
python -m src.pipeline \
    data/processed/sentiments_clean.parquet \
    outputs/transformer_scored.csv \
    --mode transformer \
    --label-column label \
    --timestamp-column created_at
```

### 6. Evaluate Results

```bash
python scripts/evaluate_results.py outputs/transformer_scored.csv --model-name "Transformer"
```

### 7. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will be available at the forwarded port (8501).

## Expected Performance

- **VADER Model**: ~56% F1-score (baseline)
- **Transformer Model**: ~75-85% F1-score (expected)
- **Target**: ≥80% F1-score

## Processing Time

On Codespaces (4-core):
- VADER: ~1-2 minutes
- Transformer: ~3-5 minutes (200K records)

On local Mac:
- VADER: ~2-3 minutes
- Transformer: ~15-20 minutes (that's why we're using Codespaces!)

## Troubleshooting

### SSL Certificate Errors
These are warnings, not errors. The NLTK data is pre-downloaded in the postCreateCommand.

### Out of Memory
If you run out of memory, process the data in batches or upgrade to an 8-core Codespace machine.

### Kaggle API Not Found
Make sure `kaggle.json` is in `~/.kaggle/` and has 600 permissions.
