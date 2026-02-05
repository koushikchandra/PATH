#!/usr/bin/env python3
"""
Create Sample Embeddings for Testing Visualization

This script generates synthetic patient embeddings for testing the
visualize_embeddings.py tool without needing to run the full training pipeline.
"""

import numpy as np
import pandas as pd
import os


def create_sample_embeddings(n_samples=100, n_dims=64, output_file="sample_embeddings.csv", random_state=42):
    """
    Create synthetic patient embeddings with two classes

    Args:
        n_samples: Number of samples to generate
        n_dims: Number of embedding dimensions
        output_file: Output CSV filename
        random_state: Random seed
    """
    np.random.seed(random_state)

    # Generate embeddings for two classes with different distributions
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    print(f"Generating {n_samples} synthetic patient embeddings...")
    print(f"  Class 0 (Primary): {n_class0} samples")
    print(f"  Class 1 (Metastatic): {n_class1} samples")
    print(f"  Embedding dimensions: {n_dims}")

    # Class 0: Centered around origin with some spread
    embeddings_class0 = np.random.randn(n_class0, n_dims) * 0.5

    # Class 1: Centered away from origin with different spread
    embeddings_class1 = np.random.randn(n_class1, n_dims) * 0.7 + 1.5

    # Combine
    embeddings = np.vstack([embeddings_class0, embeddings_class1])
    labels = np.array([0] * n_class0 + [1] * n_class1)

    # Create sample IDs
    sample_ids = [f"SAMPLE_{i:04d}" for i in range(n_samples)]

    # Create DataFrame
    data = {'Sample': sample_ids, 'label': labels}

    # Add embedding columns
    for i in range(n_dims):
        data[f'emb_{i:03d}'] = embeddings[:, i]

    df = pd.DataFrame(data)

    # Save to CSV
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\nSample embeddings saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nYou can now visualize this file using:")
    print(f"python representation/visualize_embeddings.py --input {output_file} --output_dir ./test_plots")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample embeddings for testing")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n_dims", type=int, default=64, help="Number of embedding dimensions")
    parser.add_argument("--output", type=str, default="sample_embeddings.csv", help="Output CSV file")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    create_sample_embeddings(
        n_samples=args.n_samples,
        n_dims=args.n_dims,
        output_file=args.output,
        random_state=args.random_state
    )
