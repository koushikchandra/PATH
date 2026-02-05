#!/usr/bin/env python3
"""
Extract embeddings from a single fold for visualization

This solves the clustering artifact caused by combining embeddings
from 5 different models trained on different folds.
"""

import pandas as pd
import argparse
import os


def extract_single_fold_embeddings(input_csv, fold_number, output_csv):
    """
    Extract embeddings from a single fold based on sample count

    Args:
        input_csv: Path to combined 5-fold embeddings CSV
        fold_number: Which fold to extract (1-5)
        output_csv: Output path for single-fold embeddings
    """
    # Load full embeddings
    df = pd.read_csv(input_csv)

    print(f"Total samples in combined file: {len(df)}")

    # Assuming roughly equal splits (20% test per fold)
    samples_per_fold = len(df) // 5

    # Calculate start and end indices for the requested fold
    start_idx = (fold_number - 1) * samples_per_fold
    end_idx = start_idx + samples_per_fold if fold_number < 5 else len(df)

    # Extract the fold
    fold_df = df.iloc[start_idx:end_idx].copy()

    print(f"\nExtracted Fold {fold_number}:")
    print(f"  Samples: {len(fold_df)}")

    if 'label' in fold_df.columns:
        label_counts = fold_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  Class {int(label)}: {count} samples ({count/len(fold_df)*100:.1f}%)")

    # Save to CSV
    fold_df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
    print(f"\nVisualize with:")
    print(f"python representation/visualize_embeddings.py \\")
    print(f"    --input {output_csv} \\")
    print(f"    --output_dir ./single_fold_plots \\")
    print(f"    --prefix fold{fold_number}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract single fold embeddings to avoid multi-cluster artifact"
    )
    parser.add_argument("--input", required=True, help="Input combined embeddings CSV")
    parser.add_argument("--fold", type=int, choices=[1,2,3,4,5], default=1,
                        help="Which fold to extract (1-5)")
    parser.add_argument("--output", help="Output CSV path (auto-generated if not specified)")

    args = parser.parse_args()

    # Auto-generate output name if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_fold{args.fold}_only.csv"

    extract_single_fold_embeddings(args.input, args.fold, args.output)
