#!/usr/bin/env python3
"""
Align embeddings across folds using Procrustes analysis

This corrects the 5-cluster artifact by aligning all fold embeddings
to a common reference space.
"""

import numpy as np
import pandas as pd
import argparse
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler


def procrustes_align(source, target):
    """
    Align source embeddings to target embeddings using Procrustes analysis

    Args:
        source: Source embeddings (N x D)
        target: Target embeddings (M x D)

    Returns:
        Aligned source embeddings
    """
    # Center both matrices
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Find optimal rotation matrix
    R, _ = orthogonal_procrustes(source_centered, target_centered)

    # Apply transformation
    aligned = source_centered @ R

    return aligned


def align_fold_embeddings(input_csv, output_csv, n_folds=5):
    """
    Align embeddings from different folds to a common space

    Args:
        input_csv: Path to combined 5-fold embeddings CSV
        output_csv: Output path for aligned embeddings
        n_folds: Number of folds (default: 5)
    """
    print(f"\n{'='*70}")
    print("ALIGNING EMBEDDINGS ACROSS FOLDS")
    print(f"{'='*70}\n")

    # Load full embeddings
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")

    # Extract embedding columns
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    print(f"Embedding dimensions: {len(emb_cols)}")

    embeddings = df[emb_cols].values
    samples_per_fold = len(df) // n_folds

    # Split into folds
    fold_embeddings = []
    fold_indices = []

    for fold in range(n_folds):
        start_idx = fold * samples_per_fold
        end_idx = start_idx + samples_per_fold if fold < n_folds - 1 else len(df)

        fold_emb = embeddings[start_idx:end_idx]
        fold_embeddings.append(fold_emb)
        fold_indices.append((start_idx, end_idx))

        print(f"Fold {fold+1}: {len(fold_emb)} samples")

    # Use Fold 1 as reference
    reference = fold_embeddings[0]
    print(f"\nUsing Fold 1 as reference space")

    # Align all other folds to Fold 1
    aligned_embeddings = [reference]  # Fold 1 stays as-is

    for fold_idx in range(1, n_folds):
        print(f"Aligning Fold {fold_idx+1} to reference...")
        aligned = procrustes_align(fold_embeddings[fold_idx], reference)
        aligned_embeddings.append(aligned)

    # Combine all aligned embeddings
    all_aligned = np.vstack(aligned_embeddings)

    # Create output DataFrame
    output_df = df.copy()
    for i, col in enumerate(emb_cols):
        output_df[col] = all_aligned[:, i]

    # Save aligned embeddings
    output_df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print("ALIGNMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Aligned embeddings saved to: {output_csv}")
    print(f"\nVisualize aligned embeddings:")
    print(f"python representation/visualize_embeddings.py \\")
    print(f"    --input {output_csv} \\")
    print(f"    --output_dir ./aligned_plots \\")
    print(f"    --prefix aligned_embeddings")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align fold embeddings to remove 5-cluster artifact"
    )
    parser.add_argument("--input", required=True,
                        help="Input combined embeddings CSV")
    parser.add_argument("--output", help="Output CSV path (auto-generated if not specified)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds (default: 5)")

    args = parser.parse_args()

    # Auto-generate output name if not provided
    if args.output is None:
        base_name = args.input.replace('.csv', '')
        args.output = f"{base_name}_aligned.csv"

    align_fold_embeddings(args.input, args.output, args.n_folds)
