# Quick Start Guide - Embedding Visualization

This is a quick 5-minute guide to get started with embedding visualization.

## Step 1: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

## Step 2: Generate Sample Data (Optional - for testing)

If you don't have embeddings yet, create sample data:

```bash
cd /home/user/PATH
python representation/create_sample_embeddings.py --output sample_embeddings.csv
```

This creates a CSV file with 100 synthetic patient embeddings.

## Step 3: Visualize the Embeddings

```bash
python representation/visualize_embeddings.py \
    --input sample_embeddings.csv \
    --output_dir ./test_plots
```

## Step 4: View the Results

Check the `./test_plots/` directory for:
- `patient_embeddings_umap.png` - UMAP visualization
- `patient_embeddings_tsne.png` - t-SNE visualization
- `patient_embeddings_combined.png` - Side-by-side comparison

## Common Use Cases

### Visualize Real Embeddings from Training

```bash
# After running representation.py, visualize the test embeddings
python representation/visualize_embeddings.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \
    --output_dir ./my_plots \
    --prefix my_experiment
```

### Try Different UMAP Parameters

```bash
# Tighter clusters
python representation/visualize_embeddings.py \
    --input sample_embeddings.csv \
    --output_dir ./umap_tight \
    --n_neighbors 5 \
    --min_dist 0.01

# More global structure
python representation/visualize_embeddings.py \
    --input sample_embeddings.csv \
    --output_dir ./umap_global \
    --n_neighbors 30 \
    --min_dist 0.3
```

### Try Different t-SNE Parameters

```bash
# For small datasets
python representation/visualize_embeddings.py \
    --input sample_embeddings.csv \
    --output_dir ./tsne_small \
    --perplexity 10

# For larger datasets
python representation/visualize_embeddings.py \
    --input sample_embeddings.csv \
    --output_dir ./tsne_large \
    --perplexity 50
```

## Troubleshooting

### Missing Dependencies?
```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

### Want to see all options?
```bash
python representation/visualize_embeddings.py --help
```

### Need more detailed help?
Check `README_VISUALIZATION.md` for comprehensive documentation.

## Full Workflow Example

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn

# 2. Create sample data (for testing)
python representation/create_sample_embeddings.py

# 3. Visualize with default settings
python representation/visualize_embeddings.py --input sample_embeddings.csv

# 4. View results
ls ./embedding_plots/
```

That's it! You now have beautiful UMAP and t-SNE visualizations of your embeddings.
