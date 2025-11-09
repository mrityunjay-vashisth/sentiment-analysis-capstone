# Sentiment Analysis Project - College Report

**Student Name:** [Your Name]
**Project Title:** Comparative Analysis of Sentiment Classification Models on Social Media and E-Commerce Data
**Date:** November 2025

---

## Executive Summary

This project implements and evaluates sentiment analysis models on two distinct datasets: social media posts (Twitter/Reddit) and e-commerce clothing reviews. The goal was to compare traditional lexicon-based approaches (VADER) with modern transformer-based models (DistilBERT) to achieve an F1-score of ≥80%.

### Key Findings

- **Social Media Dataset**: VADER achieved **56.2% F1-score** on 40,045 test samples
- **Clothing Reviews Dataset**: VADER achieved **38.8% F1-score** on 4,529 test samples
- **Transformer Model**: Underperformed due to binary classification limitation (29.9% F1)
- **Best Model**: VADER on social media data (70.3% of target goal)

---

## 1. Introduction

### 1.1 Problem Statement

Sentiment analysis is crucial for understanding public opinion, customer feedback, and brand perception. This project aims to:

1. Compare sentiment analysis models across different domains
2. Evaluate traditional vs. modern NLP approaches
3. Analyze performance on balanced vs. imbalanced datasets
4. Provide actionable insights for real-world applications

### 1.2 Objectives

- Implement VADER (rule-based) and Transformer (deep learning) models
- Properly split data into training (80%) and testing (20%) sets
- Evaluate using standard metrics: Accuracy, Precision, Recall, F1-Score
- Generate comprehensive reports with visualizations
- Compare performance across datasets and identify limitations

---

## 2. Datasets

### 2.1 Social Media Dataset

**Source:** Kaggle (Twitter + Reddit)
**Size:** 200,222 records
**Split:** 160,177 train / 40,045 test

**Class Distribution:**
- Negative: 21.9%
- Neutral: 34.1%
- Positive: 44.0%

**Characteristics:**
- Balanced distribution
- Short, informal text
- Clear sentiment expressions
- Mixed topics (politics, discussions)

### 2.2 Clothing Reviews Dataset

**Source:** Kaggle (Women's E-Commerce Reviews)
**Size:** 22,642 records
**Split:** 18,113 train / 4,529 test

**Class Distribution:**
- Negative: 10.5%
- Neutral: 12.5%
- Positive: 77.1%

**Characteristics:**
- Highly imbalanced (77% positive)
- Product-specific language
- Mixed sentiments common
- Nuanced opinions

---

## 3. Methodology

### 3.1 Text Preprocessing

1. **Normalization**
   - Remove URLs, mentions, hashtags
   - Remove special characters
   - Lowercase conversion

2. **Tokenization & Lemmatization**
   - Word tokenization using NLTK
   - Lemmatization to base forms
   - Stopword removal

3. **Optimization**
   - Multiprocessing (10 CPU cores)
   - Batch processing for memory efficiency

### 3.2 Models

#### 3.2.1 VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Type:** Rule-based lexicon approach
**Strengths:**
- No training required
- Fast inference
- Works well with social media text
- Handles emoticons, capitalization, punctuation

**Limitations:**
- Domain-dependent performance
- Struggles with nuanced language
- Fixed vocabulary

#### 3.2.2 DistilBERT Transformer

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`
**Type:** Pre-trained deep learning model
**Architecture:** 6-layer transformer, 66M parameters

**Critical Limitation Discovered:**
- Trained on SST-2 (binary sentiment: positive/negative only)
- Cannot handle 3-class classification (positive/negative/neutral)
- Results in 0% neutral recall

---

## 4. Experimental Setup

### 4.1 Train/Test Split

- **Strategy:** Stratified split (maintains class balance)
- **Ratio:** 80% train / 20% test
- **Random Seed:** 42 (for reproducibility)

### 4.2 Evaluation Metrics

1. **Accuracy:** Overall correctness
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1-Score (Macro):** Harmonic mean of precision and recall, averaged across classes

**Primary Metric:** Macro F1-Score (treats all classes equally, not biased by class imbalance)

---

## 5. Results

### 5.1 VADER - Social Media

| Metric | Score |
|--------|-------|
| Accuracy | 57.78% |
| Precision | 56.10% |
| Recall | 56.81% |
| **F1-Score** | **56.23%** |

**Per-Class Performance:**
- Negative: 54.2% recall
- Neutral: 52.2% recall
- Positive: 64.0% recall

**Analysis:**
- Balanced performance across classes
- Best on positive sentiments (most expressive)
- Achieves 70.3% of 80% target goal

### 5.2 VADER - Clothing Reviews

| Metric | Score |
|--------|-------|
| Accuracy | 78.12% |
| Precision | 56.42% |
| Recall | 38.58% |
| **F1-Score** | **38.82%** |

**Per-Class Performance:**
- Negative: 11.6% recall ❌
- Neutral: 4.4% recall ❌
- Positive: 99.0% recall ✅

**Analysis:**
- High accuracy misleading (due to imbalance)
- Strong bias toward positive predictions
- Poor performance on minority classes
- Domain-specific language challenges

### 5.3 Transformer (DistilBERT) - 10K Sample

| Metric | Score |
|--------|-------|
| Accuracy | 37.08% |
| Precision | 27.59% |
| Recall | 42.74% |
| **F1-Score** | **29.92%** |

**Critical Issue:**
- **0% neutral predictions** (model limitation)
- Heavily biased to negative (69.7% vs 23.6% actual)
- SST-2 pre-training incompatible with 3-class task

---

## 6. Visualizations

All visualizations are available in `reports/`:

1. **Confusion Matrices** - Show prediction vs. actual distribution
2. **Per-Class Performance** - Precision, Recall, F1 for each sentiment
3. **Model Comparison Chart** - Side-by-side metrics comparison

See `reports/project_report.html` for interactive visualizations.

---

## 7. Discussion

### 7.1 Key Findings

1. **Dataset Characteristics Matter**
   - Balanced data (social media) → better performance
   - Imbalanced data (clothing) → biased predictions

2. **Domain Adaptation Required**
   - VADER's general lexicon works well for social media
   - Clothing reviews need domain-specific models

3. **Model Selection Critical**
   - Pre-trained models must match task requirements
   - Binary models cannot handle 3-class problems

### 7.2 Why Target Not Achieved

**Social Media (56.2% vs 80% target):**
- VADER's fixed lexicon misses context
- Sarcasm and irony not detected
- Neutral sentiments ambiguous

**Clothing Reviews (38.8% vs 80% target):**
- Extreme class imbalance
- Mixed sentiments ("love fabric, hate fit")
- Product-specific terminology

**Transformer (29.9% vs 80% target):**
- Model architecture mismatch
- Binary pre-training incompatible
- Needs fine-tuning on 3-class data

### 7.3 Recommendations

1. **For Better Performance:**
   - Fine-tune transformer on target domain
   - Use 3-class pre-trained models (e.g., `cardiffnlp/twitter-roberta-base-sentiment-latest`)
   - Handle class imbalance (SMOTE, class weights)
   - Ensemble methods (combine VADER + Transformer)

2. **For Production:**
   - Use VADER for real-time, low-resource scenarios
   - Use fine-tuned transformers for high-accuracy needs
   - Implement domain-specific lexicons

---

## 8. Conclusion

This project successfully implemented and evaluated sentiment analysis models on two diverse datasets. While the 80% F1-score target was not achieved, the project provides valuable insights:

- **VADER remains competitive** for social media sentiment (56.2% F1)
- **Domain matters significantly** - same model, different performance
- **Pre-trained models need validation** for task compatibility
- **Proper evaluation is crucial** - train/test splits, multiple metrics

The infrastructure built (preprocessing, evaluation, reporting) provides a solid foundation for future improvements through fine-tuning and domain adaptation.

---

## 9. Project Structure

```
kirtiproject/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned datasets
│   └── splits/                 # Train/test splits
├── src/
│   ├── preprocess.py           # Text preprocessing
│   ├── model_infer.py          # VADER & Transformer models
│   ├── pipeline.py             # End-to-end pipeline
│   └── evaluate.py             # Metrics calculation
├── scripts/
│   ├── prepare_dataset.py      # Dataset download & transform
│   ├── train_test_split.py     # Create train/test splits
│   ├── evaluate_results.py     # Detailed evaluation
│   └── generate_report.py      # Report & visualizations
├── outputs/                     # Model predictions
├── reports/                     # Generated reports & charts
├── notebooks/                   # Jupyter notebooks
└── app/                        # Streamlit dashboard
```

---

## 10. References

1. Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
3. Kaggle Datasets:
   - Twitter/Reddit Sentiment Dataset
   - Women's E-Commerce Clothing Reviews (nicapotato)
4. NLTK: Natural Language Toolkit
5. Hugging Face Transformers Library

---

## 11. Appendix

### A. How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python3 scripts/prepare_dataset.py

# 3. Create train/test splits
python3 scripts/train_test_split.py data/processed/sentiments_clean.parquet

# 4. Run VADER on test set
python3 -m src.pipeline data/splits/sentiments_clean_test.csv outputs/test_results.csv --mode vader

# 5. Evaluate results
python3 scripts/evaluate_results.py outputs/test_results.csv --model-name "VADER"

# 6. Generate report
python3 scripts/generate_report.py
```

### B. Files for Submission

**Essential Files:**
1. `PROJECT_REPORT.md` (this file) - Main report
2. `reports/project_report.html` - Interactive report
3. `reports/results_summary.txt` - Quick results
4. `reports/*.png` - All visualizations
5. `data/splits/*_split_info.txt` - Dataset information

**Code Files:**
- All files in `src/` and `scripts/`
- `requirements.txt`

**Data Files:**
- Train/test splits in `data/splits/`
- Processed datasets in `data/processed/`

---

**End of Report**

*Generated: November 2025*
*Total Pages: 8*
