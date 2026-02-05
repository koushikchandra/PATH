#!/usr/bin/env python3
"""
Standalone Embedding Visualization Tool

This script loads patient embeddings from CSV files and generates
UMAP and t-SNE visualizations to explore learned representations.

Usage:
    python visualize_embeddings.py --input embeddings.csv --output_dir ./plots
    python visualize_embeddings.py --input embeddings.csv --perplexity 40 --n_neighbors 20
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pathlib import Path


def load_embeddings_from_csv(csv_path):
    """
    Load patient embeddings from CSV file

    Expected format:
        - First column: Sample ID
        - Second column: label (0 or 1)
        - Remaining columns: embedding dimensions (emb_000, emb_001, ...)

    Args:
        csv_path: Path to embeddings CSV file

    Returns:
        Tuple of (embeddings_array, labels_array, sample_ids)
    """
    df = pd.read_csv(csv_path)

    # Identify embedding columns (should start with 'emb_')
    emb_cols = [col for col in df.columns if col.startswith('emb_')]

    if not emb_cols:
        raise ValueError(f"No embedding columns found (expected columns starting with 'emb_')")

    embeddings = df[emb_cols].values
    labels = df['label'].values if 'label' in df.columns else None
    sample_ids = df['Sample'].values if 'Sample' in df.columns else None

    print(f"Loaded embeddings: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {int(label)}: {count} samples ({count/len(labels)*100:.1f}%)")

    return embeddings, labels, sample_ids


def plot_umap(embeddings, labels=None, sample_ids=None, save_path=None,
              n_neighbors=15, min_dist=0.1, random_state=42,
              title="UMAP Projection", figsize=(10, 8), dpi=300):
    """
    Create UMAP visualization of embeddings

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional class labels for coloring points
        sample_ids: Optional sample IDs for annotations
        save_path: Path to save plot
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
        title: Plot title
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
    """
    try:
        import umap.umap_ as umap_module
    except ImportError:
        print("Error: UMAP not installed. Install with: pip install umap-learn")
        return None

    if embeddings.shape[0] < 3:
        print("Warning: Not enough samples for UMAP (need at least 3)")
        return None

    # Standardize embeddings
    X = StandardScaler().fit_transform(embeddings)

    # Adjust n_neighbors if needed
    n_neighbors = max(2, min(n_neighbors, X.shape[0] - 1))

    print(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    umap_model = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state
    )

    xy = umap_model.fit_transform(X)

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Color by class labels
        palette = {0: "#1f77b4", 1: "#d62728"}  # Blue for Primary, Red for Metastatic
        label_names = {0: "Primary (0)", 1: "Metastatic (1)"}

        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=50, alpha=0.7,
                c=palette.get(int(cls), "#555555"),
                label=label_names.get(int(cls), f"Class {int(cls)}"),
                edgecolors='white',
                linewidths=0.5
            )
    else:
        # No labels - single color
        ax.scatter(xy[:, 0], xy[:, 1], s=50, alpha=0.7, c="#1f77b4", edgecolors='white', linewidths=0.5)

    ax.set_xlabel("UMAP-1", fontsize=14, fontweight='bold')
    ax.set_ylabel("UMAP-2", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    if labels is not None:
        ax.legend(loc="best", fontsize=12, framealpha=0.9)

    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"UMAP plot saved to: {save_path}")

    plt.close()

    return xy


def plot_tsne(embeddings, labels=None, sample_ids=None, save_path=None,
              perplexity=30, learning_rate=200, n_iter=1000, random_state=42,
              title="t-SNE Projection", figsize=(10, 8), dpi=300):
    """
    Create t-SNE visualization of embeddings

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional class labels for coloring points
        sample_ids: Optional sample IDs for annotations
        save_path: Path to save plot
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of iterations
        random_state: Random seed
        title: Plot title
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
    """
    if embeddings.shape[0] < 3:
        print("Warning: Not enough samples for t-SNE (need at least 3)")
        return None

    # Standardize embeddings
    X = StandardScaler().fit_transform(embeddings)

    # Adjust perplexity if needed
    perplexity = min(perplexity, max(5, X.shape[0] // 3))
    if perplexity >= X.shape[0]:
        perplexity = max(1, X.shape[0] - 1)

    print(f"Computing t-SNE projection (perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        verbose=1
    )

    xy = tsne.fit_transform(X)

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Color by class labels
        palette = {0: "#1f77b4", 1: "#d62728"}  # Blue for Primary, Red for Metastatic
        label_names = {0: "Primary (0)", 1: "Metastatic (1)"}

        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=50, alpha=0.7,
                c=palette.get(int(cls), "#555555"),
                label=label_names.get(int(cls), f"Class {int(cls)}"),
                edgecolors='white',
                linewidths=0.5
            )
    else:
        # No labels - single color
        ax.scatter(xy[:, 0], xy[:, 1], s=50, alpha=0.7, c="#1f77b4", edgecolors='white', linewidths=0.5)

    ax.set_xlabel("t-SNE-1", fontsize=14, fontweight='bold')
    ax.set_ylabel("t-SNE-2", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    if labels is not None:
        ax.legend(loc="best", fontsize=12, framealpha=0.9)

    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")

    plt.close()

    return xy


def plot_combined_visualization(embeddings, labels=None, sample_ids=None,
                                output_dir="./plots", prefix="embeddings",
                                n_neighbors=15, min_dist=0.1,
                                perplexity=30, learning_rate=200,
                                random_state=42, dpi=300):
    """
    Create both UMAP and t-SNE visualizations side-by-side

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional class labels
        sample_ids: Optional sample IDs
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        perplexity: t-SNE perplexity
        learning_rate: t-SNE learning rate
        random_state: Random seed
        dpi: Resolution for saved figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual plots
    umap_path = os.path.join(output_dir, f"{prefix}_umap.png")
    tsne_path = os.path.join(output_dir, f"{prefix}_tsne.png")

    umap_coords = plot_umap(
        embeddings, labels, sample_ids, umap_path,
        n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=random_state, dpi=dpi
    )

    tsne_coords = plot_tsne(
        embeddings, labels, sample_ids, tsne_path,
        perplexity=perplexity, learning_rate=learning_rate,
        random_state=random_state, dpi=dpi
    )

    # Create combined side-by-side plot
    if umap_coords is not None and tsne_coords is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        if labels is not None:
            palette = {0: "#1f77b4", 1: "#d62728"}
            label_names = {0: "Primary (0)", 1: "Metastatic (1)"}

            for cls in np.unique(labels):
                mask = labels == cls

                # UMAP
                ax1.scatter(
                    umap_coords[mask, 0], umap_coords[mask, 1],
                    s=50, alpha=0.7,
                    c=palette.get(int(cls), "#555555"),
                    label=label_names.get(int(cls), f"Class {int(cls)}"),
                    edgecolors='white', linewidths=0.5
                )

                # t-SNE
                ax2.scatter(
                    tsne_coords[mask, 0], tsne_coords[mask, 1],
                    s=50, alpha=0.7,
                    c=palette.get(int(cls), "#555555"),
                    label=label_names.get(int(cls), f"Class {int(cls)}"),
                    edgecolors='white', linewidths=0.5
                )
        else:
            ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], s=50, alpha=0.7, c="#1f77b4")
            ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=50, alpha=0.7, c="#1f77b4")

        ax1.set_xlabel("UMAP-1", fontsize=14, fontweight='bold')
        ax1.set_ylabel("UMAP-2", fontsize=14, fontweight='bold')
        ax1.set_title("UMAP Projection", fontsize=16, fontweight='bold', pad=15)
        ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        if labels is not None:
            ax1.legend(loc="best", fontsize=11, framealpha=0.9)

        ax2.set_xlabel("t-SNE-1", fontsize=14, fontweight='bold')
        ax2.set_ylabel("t-SNE-2", fontsize=14, fontweight='bold')
        ax2.set_title("t-SNE Projection", fontsize=16, fontweight='bold', pad=15)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        if labels is not None:
            ax2.legend(loc="best", fontsize=11, framealpha=0.9)

        plt.suptitle("Patient Embedding Visualizations", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        combined_path = os.path.join(output_dir, f"{prefix}_combined.png")
        plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved to: {combined_path}")

    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize patient embeddings using UMAP and t-SNE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Path to embeddings CSV file")
    parser.add_argument("--output_dir", type=str, default="./embedding_plots",
                        help="Directory to save visualization plots")
    parser.add_argument("--prefix", type=str, default="patient_embeddings",
                        help="Prefix for output filenames")

    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter (2-100)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter (0.0-0.99)")

    # t-SNE parameters
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter (5-50)")
    parser.add_argument("--learning_rate", type=float, default=200,
                        help="t-SNE learning rate (10-1000)")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="t-SNE number of iterations")

    # Other parameters
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution (DPI) for saved plots")
    parser.add_argument("--skip_umap", action='store_true',
                        help="Skip UMAP visualization")
    parser.add_argument("--skip_tsne", action='store_true',
                        help="Skip t-SNE visualization")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"\n{'='*60}")
    print("PATIENT EMBEDDING VISUALIZATION")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load embeddings
    embeddings, labels, sample_ids = load_embeddings_from_csv(args.input)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    if not args.skip_umap:
        umap_path = os.path.join(args.output_dir, f"{args.prefix}_umap.png")
        plot_umap(
            embeddings, labels, sample_ids, umap_path,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
            dpi=args.dpi
        )

    if not args.skip_tsne:
        tsne_path = os.path.join(args.output_dir, f"{args.prefix}_tsne.png")
        plot_tsne(
            embeddings, labels, sample_ids, tsne_path,
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
            random_state=args.random_state,
            dpi=args.dpi
        )

    print(f"\n{'='*60}")
    print("All visualizations generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
