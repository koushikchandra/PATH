#!/bin/bash
# One-Command Demo: Generate Sample Data, Train Model, Create Visualizations
#
# This script demonstrates the complete workflow from data generation to visualization
# Perfect for testing the pipeline without real genomic data

set -e  # Exit on error

echo "========================================================================"
echo "PATHWAY GRAPH TRANSFORMER - COMPLETE DEMO WORKFLOW"
echo "========================================================================"
echo ""
echo "This demo will:"
echo "  1. Generate synthetic genomic data (150 patients, 500 genes, 50 pathways)"
echo "  2. Train the model with 5-fold cross-validation (50 epochs for speed)"
echo "  3. Generate patient embeddings"
echo "  4. Create UMAP and t-SNE visualizations"
echo ""
echo "Estimated time: 5-15 minutes (depending on your hardware)"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

# Check/install dependencies
echo "Checking dependencies..."
python -c "import numpy, pandas, torch, sklearn" 2>/dev/null || {
    echo "Some dependencies are missing. Install with:"
    echo "  pip install numpy pandas torch scikit-learn matplotlib seaborn umap-learn"
    exit 1
}

echo "✓ All dependencies found"
echo ""

# Step 1: Generate sample data
echo "========================================================================"
echo "STEP 1: Generating Sample Data"
echo "========================================================================"
python representation/create_sample_data.py \
    --output_dir data \
    --n_patients 150 \
    --n_genes 500 \
    --n_pathways 50

# Step 2: Train model (using fewer epochs for faster demo)
echo ""
echo "========================================================================"
echo "STEP 2: Training Model (5-Fold Cross-Validation)"
echo "========================================================================"
echo "Running with 50 epochs for faster demo (use --epochs 200 for real analysis)"
echo ""

python representation/representation.py \
    --data_dir data \
    --outdir outputs_5fold_cv \
    --epochs 50 \
    --batch 16 \
    --patience 15 \
    --use_full_graph

# Step 3: Find and visualize embeddings
echo ""
echo "========================================================================"
echo "STEP 3: Creating Embedding Visualizations"
echo "========================================================================"

# Find the most recent embeddings file
EMBEDDINGS_FILE=$(ls -t outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv 2>/dev/null | head -1)

if [ -z "$EMBEDDINGS_FILE" ]; then
    echo "Error: No embeddings file found!"
    exit 1
fi

echo "Found embeddings: $EMBEDDINGS_FILE"
echo ""

python representation/visualize_embeddings.py \
    --input "$EMBEDDINGS_FILE" \
    --output_dir outputs_5fold_cv/visualizations \
    --prefix demo_embeddings \
    --dpi 300

echo ""
echo "========================================================================"
echo "DEMO COMPLETE! 🎉"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  📁 Data files: data/"
echo "  📊 Results: outputs_5fold_cv/5fold_results_*.json"
echo "  📈 Performance plots: outputs_5fold_cv/visualizations/"
echo "  🧬 Embeddings: outputs_5fold_cv/embeddings/"
echo "  🎨 UMAP/t-SNE: outputs_5fold_cv/visualizations/demo_embeddings_*.png"
echo ""
echo "View your visualizations:"
echo "  outputs_5fold_cv/visualizations/demo_embeddings_umap.png"
echo "  outputs_5fold_cv/visualizations/demo_embeddings_tsne.png"
echo "  outputs_5fold_cv/visualizations/demo_embeddings_combined.png"
echo ""
echo "========================================================================"
