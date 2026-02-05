# Patient Embedding Visualization - Complete Guide

This directory contains tools for training a Graph Transformer model on genomic data and visualizing patient embeddings with UMAP and t-SNE.

## 🚀 Quick Start (Choose One)

### Option 1: One-Command Demo (Recommended for First Time)
```bash
bash representation/RUN_DEMO.sh
```
This runs the complete pipeline automatically (5-15 minutes).

### Option 2: Step-by-Step
```bash
# 1. Generate sample data
python representation/create_sample_data.py

# 2. Train model
python representation/representation.py --data_dir data --epochs 50

# 3. Visualize embeddings
python representation/visualize_embeddings.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \
    --output_dir ./plots
```

## 📁 Files in This Directory

### Main Scripts
| File | Purpose |
|------|---------|
| `representation.py` | Main training script (5-fold CV, generates embeddings) |
| `visualize_embeddings.py` | Standalone UMAP/t-SNE visualization tool |
| `create_sample_data.py` | Generate synthetic genomic data for testing |
| `create_sample_embeddings.py` | Generate sample embeddings (skip training) |
| `RUN_DEMO.sh` | One-command automated demo script |

### Documentation
| File | Content |
|------|---------|
| `README.md` | This file - overview and quick start |
| `HOW_TO_GET_EMBEDDINGS.md` | Complete guide to generating embeddings |
| `QUICKSTART.md` | 5-minute visualization quick start |
| `README_VISUALIZATION.md` | Detailed visualization documentation |
| `DATA_REQUIREMENTS.md` | Input data format specifications |

## 🎯 Common Tasks

### Task 1: Get the Embeddings File
**Question:** "How do I get `5fold_test_embeddings_*.csv`?"

**Answer:** Run the training pipeline:
```bash
python representation/representation.py --data_dir data --epochs 50
```

Embeddings will be saved to:
```
outputs_5fold_cv/embeddings/5fold_test_embeddings_sparse_YYYYMMDD_HHMMSS.csv
```

**Don't have data?** Generate sample data first:
```bash
python representation/create_sample_data.py
```

📖 See: `HOW_TO_GET_EMBEDDINGS.md` for detailed instructions

### Task 2: Visualize Existing Embeddings
**Question:** "I have an embeddings CSV, how do I visualize it?"

**Answer:**
```bash
python representation/visualize_embeddings.py \
    --input path/to/embeddings.csv \
    --output_dir ./plots
```

📖 See: `QUICKSTART.md` or `README_VISUALIZATION.md`

### Task 3: Test Without Real Data
**Question:** "Can I test this without real genomic data?"

**Answer:** Yes! Use the demo script:
```bash
bash representation/RUN_DEMO.sh
```

Or generate sample data manually:
```bash
python representation/create_sample_data.py
```

📖 See: `HOW_TO_GET_EMBEDDINGS.md`

### Task 4: Prepare My Own Data
**Question:** "What data format do I need?"

**Answer:** You need 5 CSV files in a `data/` directory:
- `mutation_data.csv` - Binary mutation matrix
- `cnv_data.csv` - Copy number variation matrix
- `patient_labels.csv` - Primary/Metastatic labels
- `filtered_pathways.csv` - Pathway gene sets
- `adjacency_matrix.csv` - Pathway interactions (optional)

📖 See: `DATA_REQUIREMENTS.md` for complete specifications

## 📊 What You Get

### From Training (`representation.py`)
```
outputs_5fold_cv/
├── embeddings/
│   └── 5fold_test_embeddings_*.csv        ← Patient embeddings (for visualization)
├── visualizations/
│   ├── 5fold_confusion_matrix_*.png
│   ├── 5fold_roc_curve_*.png
│   └── 5fold_pr_curve_*.png
├── pathway_importance/
│   └── aggregate_pathway_importance.csv
└── 5fold_results_*.json                    ← Performance metrics
```

### From Visualization (`visualize_embeddings.py`)
```
plots/
├── patient_embeddings_umap.png        ← UMAP projection
├── patient_embeddings_tsne.png        ← t-SNE projection
└── patient_embeddings_combined.png    ← Side-by-side comparison
```

## 🔧 Installation

### Required Dependencies
```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn umap-learn
```

### Optional (for faster training)
```bash
# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 💡 Usage Examples

### Example 1: Quick Demo
```bash
# Complete pipeline in one command
bash representation/RUN_DEMO.sh
```

### Example 2: Custom Training
```bash
# Generate data
python representation/create_sample_data.py --n_patients 200 --n_genes 1000

# Train with custom parameters
python representation/representation.py \
    --data_dir data \
    --outdir my_experiment \
    --epochs 100 \
    --batch 32 \
    --lr 1e-4 \
    --use_full_graph
```

### Example 3: Custom Visualization
```bash
# UMAP with custom parameters
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --output_dir plots \
    --n_neighbors 20 \
    --min_dist 0.05 \
    --dpi 600
```

### Example 4: Only t-SNE
```bash
python representation/visualize_embeddings.py \
    --input embeddings.csv \
    --skip_umap \
    --perplexity 40
```

## 📈 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     COMPLETE WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. PREPARE DATA
   ├─ Option A: Use sample data
   │  └─ python create_sample_data.py
   └─ Option B: Prepare real data (see DATA_REQUIREMENTS.md)
      └─ Create data/*.csv files

2. TRAIN MODEL
   └─ python representation.py --data_dir data --epochs 200
      │
      ├─ 5-Fold Cross-Validation
      ├─ Trains 5 models
      ├─ Evaluates performance
      └─ Generates embeddings
         └─ outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv

3. VISUALIZE EMBEDDINGS
   └─ python visualize_embeddings.py --input embeddings.csv
      │
      ├─ Computes UMAP projection
      ├─ Computes t-SNE projection
      └─ Creates visualizations
         └─ plots/*.png

4. ANALYZE RESULTS
   ├─ View performance metrics (5fold_results_*.json)
   ├─ Examine pathway importance (pathway_importance/*.csv)
   ├─ Explore embeddings (UMAP/t-SNE plots)
   └─ Review confusion matrix and ROC curves
```

## 🎓 Learning Resources

### For Beginners
1. Start with `RUN_DEMO.sh` to see everything in action
2. Read `QUICKSTART.md` for basic visualization
3. Try `create_sample_embeddings.py` to test visualization only

### For Intermediate Users
1. Read `HOW_TO_GET_EMBEDDINGS.md` for the complete process
2. Explore `README_VISUALIZATION.md` for all visualization options
3. Experiment with different model parameters

### For Advanced Users
1. Review `DATA_REQUIREMENTS.md` for data preparation
2. Modify `representation.py` for custom architectures
3. Extend `visualize_embeddings.py` for additional analyses

## 🔍 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn umap-learn
```

### "FileNotFoundError: data/mutation_data.csv"
```bash
python representation/create_sample_data.py
```

### "No embeddings file found"
First complete training:
```bash
python representation/representation.py --data_dir data --epochs 50
```

### Training is too slow
Reduce epochs or batch size:
```bash
python representation/representation.py --epochs 30 --batch 8
```

### More help?
- Check individual documentation files
- Look at error messages carefully
- Try the demo script first

## 📊 Model Architecture

The pipeline uses an **Edge-Aware Graph Transformer** that:
- Encodes gene mutations and CNV data using FiLM-style modulation
- Pools genes into pathway representations using attention
- Processes pathways through graph transformer layers
- Generates patient embeddings for classification
- Learns which pathways are most important

## 🎨 Visualization Features

The visualization tool provides:
- **UMAP**: Preserves local and global structure
- **t-SNE**: Emphasizes local neighborhood structure
- **Combined view**: Side-by-side comparison
- **Customizable**: Adjust all projection parameters
- **High-quality**: Publication-ready figures (configurable DPI)

## 📚 Citation

If you use this code, please consider citing:
- UMAP: McInnes et al., arXiv:1802.03426
- t-SNE: van der Maaten & Hinton, JMLR 2008
- Graph Transformers: Dwivedi & Bresson, arXiv:2012.09699

## 📞 Support

For issues or questions:
1. Check the relevant documentation file
2. Review error messages and troubleshooting section
3. Ensure all dependencies are installed
4. Try the demo script to verify setup

## 🎯 Summary

| If you want to... | Use this... |
|-------------------|-------------|
| Run complete demo | `bash RUN_DEMO.sh` |
| Generate embeddings | `python representation.py` |
| Visualize embeddings | `python visualize_embeddings.py` |
| Create test data | `python create_sample_data.py` |
| Quick test (no training) | `python create_sample_embeddings.py` |
| Learn about data format | Read `DATA_REQUIREMENTS.md` |
| Learn about visualization | Read `README_VISUALIZATION.md` |
| Get embeddings | Read `HOW_TO_GET_EMBEDDINGS.md` |
| 5-min quick start | Read `QUICKSTART.md` |

---

**Last Updated:** 2026-02-05

**Version:** 1.0

**License:** MIT (or your preferred license)
