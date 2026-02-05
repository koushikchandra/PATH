# How to Get the Embeddings File

This guide explains how to generate the `5fold_test_embeddings_*.csv` file.

## 🚀 Quick Answer (5 minutes)

Run the complete demo with one command:

```bash
cd /home/user/PATH
bash representation/RUN_DEMO.sh
```

This will:
1. ✅ Generate sample genomic data
2. ✅ Train the model (5-fold CV)
3. ✅ Create embeddings automatically
4. ✅ Generate UMAP/t-SNE visualizations

**Result:** Embeddings will be in `outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv`

---

## 📋 Step-by-Step Manual Process

### Option A: Use Sample Data (For Testing)

**Step 1:** Generate sample data
```bash
python representation/create_sample_data.py
```

**Step 2:** Train the model
```bash
python representation/representation.py \
    --data_dir data \
    --outdir outputs_5fold_cv \
    --epochs 50
```

**Step 3:** Find your embeddings
```bash
ls outputs_5fold_cv/embeddings/
```

You'll see: `5fold_test_embeddings_sparse_YYYYMMDD_HHMMSS.csv`

### Option B: Use Your Own Data

**Step 1:** Prepare your data files in `data/` directory:
```
data/
├── mutation_data.csv        # Binary mutation matrix
├── cnv_data.csv            # Copy number variation matrix
├── patient_labels.csv      # Primary/Metastatic labels
├── filtered_pathways.csv   # Pathway gene sets
└── adjacency_matrix.csv    # Pathway interactions (optional with --use_full_graph)
```

See `DATA_REQUIREMENTS.md` for detailed format specifications.

**Step 2:** Train the model
```bash
python representation/representation.py \
    --data_dir data \
    --outdir outputs_5fold_cv \
    --epochs 200 \
    --batch 16
```

**Step 3:** Get embeddings
```bash
# Embeddings are automatically saved here:
ls outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv
```

---

## 🎯 Understanding the Embeddings File

The generated file `5fold_test_embeddings_sparse_20240205_123456.csv` contains:

### Filename Components:
- `5fold` - Generated from 5-fold cross-validation
- `test` - Test set embeddings (not training set)
- `sparse` or `fullgraph` - Graph mode used
- `20240205_123456` - Timestamp of generation

### File Contents:

| Sample       | label | emb_000 | emb_001 | emb_002 | ... | emb_063 |
|--------------|-------|---------|---------|---------|-----|---------|
| PATIENT_0001 | 0     | 0.234   | -0.567  | 0.891   | ... | -0.123  |
| PATIENT_0002 | 1     | -0.456  | 0.789   | -0.234  | ... | 0.567   |
| ...          | ...   | ...     | ...     | ...     | ... | ...     |

- **Sample**: Patient ID
- **label**: 0 = Primary, 1 = Metastatic
- **emb_000 to emb_063**: 64-dimensional embedding (default)

---

## 📊 What Happens During Training?

When you run `representation.py`, it:

1. **Loads your data** from the `data/` directory
2. **Creates 5-fold splits** (stratified by class)
3. **Trains 5 separate models** (one per fold)
4. **For each fold:**
   - Trains on 72% of data
   - Validates on 8% of data
   - Tests on 20% of data
   - Extracts embeddings from test samples
5. **Combines all test embeddings** into one CSV file
6. **Saves multiple outputs:**
   - Embeddings: `embeddings/5fold_test_embeddings_*.csv`
   - Results: `5fold_results_*.json`
   - Visualizations: `visualizations/*.png`
   - Pathway importance: `pathway_importance/*.csv`

---

## ⚙️ Customization Options

### Faster Training (For Testing)
```bash
python representation/representation.py \
    --data_dir data \
    --epochs 50 \
    --patience 15 \
    --batch 32
```

### Better Performance (For Publication)
```bash
python representation/representation.py \
    --data_dir data \
    --epochs 300 \
    --patience 50 \
    --batch 8 \
    --lr 5e-5
```

### Full Graph Mode (No Adjacency Matrix Required)
```bash
python representation/representation.py \
    --data_dir data \
    --use_full_graph \
    --epochs 200
```

### Larger Embeddings
```bash
python representation/representation.py \
    --data_dir data \
    --d_model 128 \
    --epochs 200
```
This creates 128-dimensional embeddings instead of 64.

---

## 🔍 Troubleshooting

### Problem: "FileNotFoundError: data/mutation_data.csv"
**Solution:** You need to create data files first:
```bash
python representation/create_sample_data.py
```

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn umap-learn
```

### Problem: Training is too slow
**Solution:** Reduce epochs or use GPU:
```bash
python representation/representation.py --epochs 50
```

### Problem: Out of memory
**Solution:** Reduce batch size:
```bash
python representation/representation.py --batch 4
```

### Problem: No embeddings file generated
**Solution:** Check the output directory:
```bash
# Look for embeddings
find outputs_5fold_cv -name "*embeddings*.csv"

# Check if training completed
ls outputs_5fold_cv/5fold_results_*.json
```

---

## 📁 Expected Output Structure

After running `representation.py`, you'll have:

```
outputs_5fold_cv/
├── 5fold_results_sparse_20240205_123456.json
├── 5fold_test_predictions_sparse_20240205_123456.csv
├── 5fold_splits_rs42.json
│
├── embeddings/
│   ├── 5fold_test_embeddings_sparse_20240205_123456.csv  ← THIS IS YOUR FILE!
│   ├── 5fold_test_embeddings_sparse_20240205_123456_umap.png
│   └── 5fold_test_embeddings_sparse_20240205_123456_tsne.png
│
├── visualizations/
│   ├── 5fold_confusion_matrix_sparse_20240205_123456.png
│   ├── 5fold_roc_curve_sparse_20240205_123456.png
│   └── 5fold_pr_curve_sparse_20240205_123456.png
│
└── pathway_importance/
    ├── pathway_importance_fold1.csv
    ├── ...
    └── aggregate_pathway_importance.csv
```

---

## 🎨 Visualize Your Embeddings

Once you have the embeddings file:

```bash
python representation/visualize_embeddings.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \
    --output_dir ./my_plots
```

---

## 💡 Tips

1. **Start with sample data** - Test the pipeline before using real data
2. **Use fewer epochs for testing** - 50 epochs is enough to verify everything works
3. **Monitor training** - Watch the terminal output for validation AUC
4. **Check embeddings** - Look at the CSV file to ensure it has the right format
5. **Experiment with parameters** - Try different n_neighbors and perplexity values

---

## 📚 Related Documentation

- `QUICKSTART.md` - 5-minute quick start
- `README_VISUALIZATION.md` - Complete visualization guide
- `DATA_REQUIREMENTS.md` - Data format specifications
- `RUN_DEMO.sh` - Automated demo script

---

## ⏱️ Time Estimates

| Task | Small Dataset | Medium Dataset | Large Dataset |
|------|--------------|----------------|---------------|
| Sample data generation | < 1 min | < 1 min | < 1 min |
| Training (50 epochs) | 5-10 min | 15-30 min | 30-60 min |
| Training (200 epochs) | 15-30 min | 45-90 min | 2-4 hours |
| Visualization | < 1 min | 1-2 min | 2-5 min |

*Small: <100 patients, Medium: 100-500, Large: >500*

---

## 🆘 Need Help?

If you're still having trouble:
1. Check all documentation files in `representation/`
2. Verify your data files match the required format
3. Try the demo script first: `bash representation/RUN_DEMO.sh`
4. Look at error messages carefully - they usually indicate the problem
