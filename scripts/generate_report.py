#!/usr/bin/env python3
"""
Generate comprehensive project report with visualizations.

This script creates:
- Detailed performance metrics
- Confusion matrices
- Performance comparison charts
- Classification reports
- PDF/HTML report
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import classification_scores


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_class_distribution(df, title, save_path):
    """Plot sentiment class distribution."""
    labels = ['negative', 'neutral', 'positive']
    counts = df['label'].value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts.values, color=['#d32f2f', '#ffa726', '#66bb6a'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_model_comparison(results_dict, save_path):
    """Compare multiple models' performance."""
    models = list(results_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    data = {
        'Accuracy': [results_dict[m]['accuracy'] for m in models],
        'Precision': [results_dict[m]['precision'] for m in models],
        'Recall': [results_dict[m]['recall'] for m in models],
        'F1-Score': [results_dict[m]['f1_macro'] for m in models]
    }

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_per_class_performance(y_true, y_pred, model_name, save_path):
    """Plot per-class precision, recall, F1."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = ['negative', 'neutral', 'positive']
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precision, width, label='Precision', color='#1976d2')
    ax.bar(x, recall, width, label='Recall', color='#388e3c')
    ax.bar(x + width, f1, width, label='F1-Score', color='#f57c00')

    ax.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def generate_html_report(results_dict, output_path):
    """Generate HTML report with all metrics and visualizations."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis Project Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 { margin: 0; font-size: 2.5em; }
        h2 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 { color: #555; margin-top: 25px; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }
        tr:hover { background-color: #f5f5f5; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
        }
        .success { color: #4caf50; font-weight: bold; }
        .warning { color: #ff9800; font-weight: bold; }
        .error { color: #f44336; font-weight: bold; }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Sentiment Analysis Project Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """</p>
    </div>

    <h2>📊 Executive Summary</h2>
    <p>This report presents a comprehensive sentiment analysis study comparing multiple models across different datasets.</p>
"""

    # Add results for each model
    for model_name, metrics in results_dict.items():
        html += f"""
    <h2>🔍 {model_name}</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{metrics['accuracy']:.1%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['precision']:.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['recall']:.1%}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['f1_macro']:.1%}</div>
            <div class="metric-label">F1-Score</div>
        </div>
    </div>
"""

        # Add classification report table
        if 'per_class' in metrics:
            html += """
    <h3>Per-Class Performance</h3>
    <table>
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
"""
            for class_name, class_metrics in metrics['per_class'].items():
                # Skip non-class entries like 'accuracy', 'macro avg', 'weighted avg'
                if not isinstance(class_metrics, dict) or class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                html += f"""
        <tr>
            <td><strong>{class_name.capitalize()}</strong></td>
            <td>{class_metrics['precision']:.3f}</td>
            <td>{class_metrics['recall']:.3f}</td>
            <td>{class_metrics['f1-score']:.3f}</td>
            <td>{int(class_metrics['support'])}</td>
        </tr>
"""
            html += "    </table>\n"

    html += """
    <h2>📈 Visualizations</h2>
    <div class="chart-container">
        <h3>Model Comparison</h3>
        <img src="model_comparison.png" alt="Model Comparison">
    </div>

    <div class="footer">
        <p>Generated by Sentiment Analysis Pipeline</p>
        <p>For more information, see project documentation</p>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"  ✓ Saved: {output_path}")


def generate_comprehensive_report(
    test_results: dict[str, str],
    output_dir: str = "reports"
):
    """Generate comprehensive project report with all visualizations.

    Args:
        test_results: Dict mapping model names to scored CSV paths
        output_dir: Directory to save reports and visualizations
    """
    print("=" * 70)
    print("GENERATING COMPREHENSIVE PROJECT REPORT")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {}

    # Process each model's results
    for model_name, scored_csv in test_results.items():
        print(f"\nProcessing {model_name}...")
        df = pd.read_csv(scored_csv)

        y_true = df['label']
        y_pred = df['sentiment_label']

        # Calculate metrics
        metrics = classification_scores(y_true, y_pred)

        # Get per-class metrics
        from sklearn.metrics import classification_report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        results_dict[model_name] = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_macro': metrics.f1_macro,
            'per_class': class_report
        }

        # Generate visualizations
        print(f"  Generating visualizations for {model_name}...")

        # Confusion matrix
        cm_path = output_path / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, f"{model_name} - Confusion Matrix", cm_path)

        # Per-class performance
        perf_path = output_path / f"{model_name.lower().replace(' ', '_')}_per_class.png"
        plot_per_class_performance(y_true, y_pred, model_name, perf_path)

    # Generate comparison charts
    print("\nGenerating comparison charts...")
    comp_path = output_path / "model_comparison.png"
    plot_model_comparison(results_dict, comp_path)

    # Generate HTML report
    print("\nGenerating HTML report...")
    html_path = output_path / "project_report.html"
    generate_html_report(results_dict, html_path)

    # Generate text summary
    print("\nGenerating text summary...")
    summary_path = output_path / "results_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SENTIMENT ANALYSIS PROJECT - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for model_name, metrics in results_dict.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)\n")

            # Target assessment
            target_f1 = 0.80
            if metrics['f1_macro'] >= target_f1:
                f.write(f"\n✓ Target F1-score of {target_f1:.0%} ACHIEVED!\n")
            else:
                gap = target_f1 - metrics['f1_macro']
                f.write(f"\n✗ Target F1-score of {target_f1:.0%} not met\n")
                f.write(f"  Gap: {gap:.4f} ({gap*100:.2f} percentage points)\n")
                f.write(f"  Progress: {(metrics['f1_macro']/target_f1)*100:.1f}% of target\n")

    print(f"  ✓ Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("✓ REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nReports saved to: {output_path}/")
    print(f"  - HTML Report: {html_path.name}")
    print(f"  - Text Summary: {summary_path.name}")
    print(f"  - Visualizations: *.png files")
    print("=" * 70)


if __name__ == "__main__":
    # Example usage
    test_results = {
        "VADER - Social Media": "outputs/vader_scored.csv",
        "VADER - Clothing": "outputs/clothing_vader_scored.csv",
    }

    generate_comprehensive_report(test_results)
