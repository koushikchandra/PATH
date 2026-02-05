# Patient Embedding Visualization Guide

This guide explains how to use `visualize_embeddings.py` to create UMAP and t-SNE visualizations of patient embeddings.

## Prerequisites

### 1. Install Required Dependencies

First, ensure you have the necessary Python packages installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

Or if you have a requirements file:

```bash
pip install -r requirements.txt
```

### Required packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scikit-learn` - Machine learning utilities (includes t-SNE)
- `umap-learn` - UMAP dimensionality reduction

## Input File Format

The script expects a CSV file with the following format:

| Sample | label | emb_000 | emb_001 | emb_002 | ... |
|--------|-------|---------|---------|---------|-----|
| P001   | 0     | 0.123   | -0.456  | 0.789   | ... |
| P002   | 1     | -0.234  | 0.567   | -0.890  | ... |
| P003   | 0     | 0.345   | -0.678  | 0.901   | ... |

**Column Requirements:**
- `Sample` - Patient/sample ID (optional but recommended)
- `label` - Class label (0 = Primary, 1 = Metastatic)
- `emb_000`, `emb_001`, ... - Embedding dimensions (any number of dimensions)

## Usage Examples

### 1. Basic Usage

```bash
python representation/visualize_embeddings.py \
    --input path/to/embeddings.csv \
    --output_dir ./plots
```

This will create:
- `patient_embeddings_umap.png`
- `patient_embeddings_tsne.png`
- `patient_embeddings_combined.png`

### 2. With Custom Output Prefix

```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./my_visualizations \
    --prefix fold1_test
```

Creates:
- `fold1_test_umap.png`
- `fold1_test_tsne.png`
- `fold1_test_combined.png`

### 3. Custom UMAP Parameters

```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./plots \
    --n_neighbors 20 \
    --min_dist 0.05
```

**UMAP Parameters:**
- `--n_neighbors` (default: 15) - Controls local vs global structure
  - Lower values (5-10): Focus on local structure
  - Higher values (20-50): Preserve more global structure
- `--min_dist` (default: 0.1) - Minimum distance between points
  - Lower values (0.0-0.1): Tight clusters
  - Higher values (0.2-0.5): More spread out

### 4. Custom t-SNE Parameters

```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./plots \
    --perplexity 40 \
    --learning_rate 300 \
    --n_iter 1500
```

**t-SNE Parameters:**
- `--perplexity` (default: 30) - Balance between local and global aspects
  - Small datasets (< 100 samples): 5-15
  - Medium datasets (100-500): 20-40
  - Large datasets (> 500): 30-50
- `--learning_rate` (default: 200) - Step size during optimization
  - Too low: Slow convergence
  - Too high: Unstable results
  - Typical range: 100-500
- `--n_iter` (default: 1000) - Number of optimization iterations
  - Minimum: 500
  - Recommended: 1000-2000 for better convergence

### 5. Skip One Visualization Method

```bash
# Only generate UMAP (skip t-SNE)
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./plots \
    --skip_tsne

# Only generate t-SNE (skip UMAP)
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./plots \
    --skip_umap
```

### 6. High-Resolution Output for Publication

```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./publication_figures \
    --dpi 600
```

### 7. Reproducible Results

```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir ./plots \
    --random_state 42
```

## Complete Example Workflow

### Step 1: Train model and generate embeddings
```bash
python representation/representation.py \
    --data_dir ./data \
    --outdir ./outputs_5fold_cv \
    --epochs 200
```

This creates embeddings in:
```
outputs_5fold_cv/embeddings/5fold_test_embeddings_sparse_20240205_120000.csv
```

### Step 2: Visualize the embeddings
```bash
python representation/visualize_embeddings.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_sparse_20240205_120000.csv \
    --output_dir ./embedding_visualizations \
    --prefix experiment1 \
    --n_neighbors 15 \
    --perplexity 30 \
    --dpi 300
```

### Step 3: View the results
Plots will be saved in `./embedding_visualizations/`:
- `experiment1_umap.png`
- `experiment1_tsne.png`
- `experiment1_combined.png`

## Command-Line Options Reference

### Required Arguments
- `--input` - Path to embeddings CSV file

### Optional Arguments

#### Output Settings
- `--output_dir` (default: `./embedding_plots`) - Output directory
- `--prefix` (default: `patient_embeddings`) - Filename prefix
- `--dpi` (default: `300`) - Image resolution

#### UMAP Settings
- `--n_neighbors` (default: `15`) - Number of neighbors (2-100)
- `--min_dist` (default: `0.1`) - Minimum distance (0.0-0.99)

#### t-SNE Settings
- `--perplexity` (default: `30`) - Perplexity value (5-50)
- `--learning_rate` (default: `200`) - Learning rate (10-1000)
- `--n_iter` (default: `1000`) - Number of iterations

#### Other Settings
- `--random_state` (default: `42`) - Random seed
- `--skip_umap` - Skip UMAP visualization
- `--skip_tsne` - Skip t-SNE visualization

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'umap'"
**Solution:** Install UMAP
```bash
pip install umap-learn
```

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install all dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

### Issue: "No embedding columns found"
**Solution:** Ensure your CSV has columns starting with `emb_` (e.g., `emb_000`, `emb_001`, etc.)

### Issue: t-SNE is very slow
**Solution:**
- Reduce `--n_iter` to 500
- Use `--skip_tsne` if only UMAP is needed
- UMAP is generally faster for large datasets

### Issue: Plots look crowded
**Solution:** Adjust visualization parameters:
```bash
# For UMAP: increase min_dist
--min_dist 0.3

# For t-SNE: adjust perplexity
--perplexity 20
```

## Tips for Best Results

1. **Start with defaults** - The default parameters work well for most datasets
2. **Experiment with n_neighbors (UMAP)** - Try values between 10-30
3. **Adjust perplexity (t-SNE)** - Should be roughly 1/3 of your sample size
4. **Use consistent random_state** - For reproducible visualizations
5. **Compare both methods** - UMAP and t-SNE may reveal different patterns
6. **Check sample size** - Small datasets (< 50 samples) may not separate well

## Examples for Different Dataset Sizes

### Small Dataset (< 100 samples)
```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --n_neighbors 8 \
    --perplexity 10 \
    --output_dir ./plots
```

### Medium Dataset (100-500 samples)
```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --n_neighbors 15 \
    --perplexity 30 \
    --output_dir ./plots
```

### Large Dataset (> 500 samples)
```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --n_neighbors 30 \
    --perplexity 50 \
    --output_dir ./plots
```

## Understanding the Output

### Color Coding
- **Blue points** - Class 0 (Primary)
- **Red points** - Class 1 (Metastatic)

### Interpretation
- **Tight clusters** - Similar patients group together
- **Separation** - Good separation suggests the model learned discriminative features
- **Overlap** - May indicate challenging cases or ambiguous samples
- **Outliers** - Points far from clusters may be unusual samples

## Additional Resources

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [t-SNE FAQ](https://lvdmaaten.github.io/tsne/)
- [Visualization Best Practices](https://distill.pub/2016/misread-tsne/)
