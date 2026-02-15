#!/usr/bin/env python3
"""
Visualize hyperparameter search results

Usage:
    python visualize_search_results.py random_search_results_*/random_search_results.csv
    python visualize_search_results.py grid_search_results_*/grid_search_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results(csv_path):
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(df)} results from {csv_path}")
    return df


def plot_hyperparameter_importance(df, output_dir):
    """Plot how each hyperparameter affects performance."""

    # Get hyperparameter columns
    hp_cols = [col for col in df.columns if col.startswith('hp_')]

    # Numeric hyperparameters
    numeric_hps = []
    categorical_hps = []

    for col in hp_cols:
        if df[col].dtype in ['float64', 'int64']:
            # Check if it's actually categorical (few unique values)
            if df[col].nunique() <= 10:
                categorical_hps.append(col)
            else:
                numeric_hps.append(col)
        else:
            categorical_hps.append(col)

    # Plot 1: Numeric hyperparameters vs F1
    if numeric_hps:
        n_plots = len(numeric_hps)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, hp in enumerate(numeric_hps):
            ax = axes[i]
            ax.scatter(df[hp], df['test_f1'], alpha=0.6, s=100)
            ax.set_xlabel(hp.replace('hp_', ''), fontsize=12)
            ax.set_ylabel('Test F1 Score', fontsize=12)
            ax.set_title(f'Impact of {hp.replace("hp_", "")} on F1', fontsize=14)

            # Add trend line
            z = np.polyfit(df[hp], df['test_f1'], 1)
            p = np.poly1d(z)
            ax.plot(df[hp], p(df[hp]), "r--", alpha=0.5, linewidth=2)

        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'numeric_hyperparameters.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_path}")
        plt.close()

    # Plot 2: Categorical hyperparameters vs F1
    if categorical_hps:
        n_plots = len(categorical_hps)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, hp in enumerate(categorical_hps):
            ax = axes[i]

            # Box plot
            df.boxplot(column='test_f1', by=hp, ax=ax)
            ax.set_xlabel(hp.replace('hp_', ''), fontsize=12)
            ax.set_ylabel('Test F1 Score', fontsize=12)
            ax.set_title(f'Impact of {hp.replace("hp_", "")} on F1', fontsize=14)
            plt.sca(ax)
            plt.xticks(rotation=45)

        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'categorical_hyperparameters.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_path}")
        plt.close()


def plot_performance_distribution(df, output_dir):
    """Plot distribution of performance metrics."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics = ['test_f1', 'test_auc', 'test_aupr', 'test_recall', 'test_precision', 'test_acc']
    metric_names = ['F1 Score', 'AUC', 'AUPR', 'Recall', 'Precision', 'Accuracy']

    for ax, metric, name in zip(axes.flatten(), metrics, metric_names):
        if metric in df.columns:
            ax.hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[metric].mean():.3f}')
            ax.axvline(df[metric].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[metric].median():.3f}')
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of Test {name}', fontsize=14)
            ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    """Plot correlation between hyperparameters and metrics."""

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    hp_cols = [col for col in numeric_cols if col.startswith('hp_')]
    metric_cols = ['test_f1', 'test_auc', 'test_aupr', 'test_recall', 'test_precision', 'test_acc']

    # Create correlation matrix
    cols_to_include = hp_cols + [m for m in metric_cols if m in df.columns]
    corr = df[cols_to_include].corr()

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation between Hyperparameters and Metrics', fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()


def plot_top_configs_comparison(df, output_dir, top_n=10):
    """Compare top N configurations."""

    top_df = df.head(top_n)

    metrics = ['test_f1', 'test_auc', 'test_recall', 'test_precision']
    metric_names = ['F1', 'AUC', 'Recall', 'Precision']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(top_df))
    width = 0.2

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if metric in top_df.columns:
            offset = width * (i - 1.5)
            ax.bar([xi + offset for xi in x], top_df[metric], width, label=name, alpha=0.8)

    ax.set_xlabel('Configuration Rank', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Top {top_n} Configurations Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in x])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'top_configs_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_search_results.py <path_to_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load results
    df = load_results(csv_path)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(csv_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📊 Generating visualizations...")
    print(f"Output directory: {output_dir}\n")

    # Generate plots
    plot_performance_distribution(df, output_dir)
    plot_hyperparameter_importance(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_top_configs_comparison(df, output_dir)

    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
