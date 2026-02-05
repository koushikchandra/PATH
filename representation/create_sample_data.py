#!/usr/bin/env python3
"""
Create Sample Data for Testing the Training Pipeline

This script generates synthetic genomic data files that match the required format
for representation.py. Use this to test the pipeline before using real data.
"""

import numpy as np
import pandas as pd
import os


def create_sample_data(output_dir="data", n_patients=150, n_genes=500, n_pathways=50, random_state=42):
    """
    Generate synthetic genomic data for testing

    Args:
        output_dir: Directory to save data files
        n_patients: Number of patients to generate
        n_genes: Number of genes
        n_pathways: Number of pathways
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("GENERATING SAMPLE GENOMIC DATA")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Patients: {n_patients}")
    print(f"Genes: {n_genes}")
    print(f"Pathways: {n_pathways}")
    print(f"{'='*70}\n")

    # Generate gene names
    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    patient_ids = [f"PATIENT_{i:04d}" for i in range(n_patients)]

    # 1. Generate mutation_data.csv
    print("1. Generating mutation_data.csv...")
    # Binary mutations: 0 or 1, sparse (most genes not mutated)
    mutation_prob = 0.1  # 10% mutation rate
    mutation_data = np.random.binomial(1, mutation_prob, size=(n_patients, n_genes))

    mut_df = pd.DataFrame(mutation_data, columns=gene_names)
    mut_df.insert(0, 'Patient_ID', patient_ids)
    mut_file = os.path.join(output_dir, 'mutation_data.csv')
    mut_df.to_csv(mut_file, index=False)
    print(f"   ✓ Saved: {mut_file}")
    print(f"   - Shape: {n_patients} patients × {n_genes} genes")
    print(f"   - Mutation rate: {mutation_data.mean()*100:.1f}%")

    # 2. Generate cnv_data.csv
    print("\n2. Generating cnv_data.csv...")
    # Continuous CNV values: log2 ratios, mostly around 0
    cnv_data = np.random.randn(n_patients, n_genes) * 0.5  # std=0.5

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)
    cnv_df.insert(0, 'Patient_ID', patient_ids)
    cnv_file = os.path.join(output_dir, 'cnv_data.csv')
    cnv_df.to_csv(cnv_file, index=False)
    print(f"   ✓ Saved: {cnv_file}")
    print(f"   - Shape: {n_patients} patients × {n_genes} genes")
    print(f"   - Mean: {cnv_data.mean():.3f}, Std: {cnv_data.std():.3f}")

    # 3. Generate patient_labels.csv
    print("\n3. Generating patient_labels.csv...")
    # Balanced classes: 50% Primary, 50% Metastatic
    n_primary = n_patients // 2
    n_metastatic = n_patients - n_primary
    labels = ['Primary'] * n_primary + ['Metastatic'] * n_metastatic
    np.random.shuffle(labels)

    label_df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Label': labels
    })
    label_file = os.path.join(output_dir, 'patient_labels.csv')
    label_df.to_csv(label_file, index=False)
    print(f"   ✓ Saved: {label_file}")
    print(f"   - Primary: {labels.count('Primary')} ({labels.count('Primary')/len(labels)*100:.1f}%)")
    print(f"   - Metastatic: {labels.count('Metastatic')} ({labels.count('Metastatic')/len(labels)*100:.1f}%)")

    # 4. Generate filtered_pathways.csv
    print("\n4. Generating filtered_pathways.csv...")
    pathway_data = []
    genes_per_pathway = n_genes // n_pathways  # Average genes per pathway

    for i in range(n_pathways):
        pathway_id = f"PW{i:04d}"
        pathway_name = f"Synthetic Pathway {i+1}"

        # Randomly select 5-20 genes for this pathway
        n_genes_in_pathway = np.random.randint(5, min(21, genes_per_pathway + 5))
        pathway_genes = np.random.choice(gene_names, size=n_genes_in_pathway, replace=False)
        genes_str = ','.join(pathway_genes)

        pathway_data.append({
            'Pathway_ID': pathway_id,
            'Pathway_Name': pathway_name,
            'Genes': genes_str
        })

    pathway_df = pd.DataFrame(pathway_data)
    pathway_file = os.path.join(output_dir, 'filtered_pathways.csv')
    pathway_df.to_csv(pathway_file, index=False)
    print(f"   ✓ Saved: {pathway_file}")
    print(f"   - Number of pathways: {n_pathways}")
    print(f"   - Genes per pathway: 5-20 (random)")

    # 5. Generate adjacency_matrix.csv
    print("\n5. Generating adjacency_matrix.csv...")
    pathway_ids = [f"PW{i:04d}" for i in range(n_pathways)]

    # Create sparse adjacency matrix (10% connectivity)
    adjacency = np.random.rand(n_pathways, n_pathways)
    adjacency = (adjacency > 0.9).astype(float) * np.random.rand(n_pathways, n_pathways)

    # Make symmetric
    adjacency = (adjacency + adjacency.T) / 2

    # Zero diagonal
    np.fill_diagonal(adjacency, 0)

    adj_df = pd.DataFrame(adjacency, index=pathway_ids, columns=pathway_ids)
    adj_df.insert(0, 'Pathway_ID', pathway_ids)
    adj_file = os.path.join(output_dir, 'adjacency_matrix.csv')
    adj_df.to_csv(adj_file, index=False)
    print(f"   ✓ Saved: {adj_file}")
    print(f"   - Shape: {n_pathways} × {n_pathways}")
    num_edges = (adjacency > 0).sum()
    max_edges = n_pathways * (n_pathways - 1)
    sparsity = 100 * (1 - num_edges / max_edges)
    print(f"   - Sparsity: {sparsity:.1f}% ({num_edges} edges)")

    print(f"\n{'='*70}")
    print("SAMPLE DATA GENERATION COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Review the generated files in the 'data/' directory")
    print("2. Run the training pipeline:")
    print(f"   python representation/representation.py --data_dir {output_dir} --epochs 50")
    print("\n3. This will generate embeddings in:")
    print(f"   outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv")
    print("\n4. Visualize the embeddings:")
    print("   python representation/visualize_embeddings.py \\")
    print("       --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \\")
    print("       --output_dir ./plots")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic genomic data for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for data files")
    parser.add_argument("--n_patients", type=int, default=150,
                        help="Number of patients to generate")
    parser.add_argument("--n_genes", type=int, default=500,
                        help="Number of genes")
    parser.add_argument("--n_pathways", type=int, default=50,
                        help="Number of pathways")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    create_sample_data(
        output_dir=args.output_dir,
        n_patients=args.n_patients,
        n_genes=args.n_genes,
        n_pathways=args.n_pathways,
        random_state=args.random_state
    )
