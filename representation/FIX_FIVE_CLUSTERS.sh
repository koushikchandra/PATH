#!/bin/bash
# Quick script to fix the 5-cluster visualization artifact
#
# This script takes your 5-fold embeddings and creates properly aligned
# visualizations that show the true biological structure.

set -e

echo "========================================================================"
echo "FIXING 5-CLUSTER VISUALIZATION ARTIFACT"
echo "========================================================================"
echo ""

# Find the embeddings file
EMBEDDINGS_FILE=$(ls outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv 2>/dev/null | head -1)

if [ -z "$EMBEDDINGS_FILE" ]; then
    echo "❌ Error: No embeddings file found!"
    echo ""
    echo "Expected location: outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv"
    echo ""
    echo "Generate embeddings first by running:"
    echo "  bash representation/RUN_DEMO.sh"
    echo "  OR"
    echo "  python representation/representation.py --data_dir data --epochs 50"
    exit 1
fi

echo "Found embeddings file: $EMBEDDINGS_FILE"
echo ""

# Create output directories
mkdir -p single_fold_plots aligned_plots comparison_plots

echo "========================================================================"
echo "SOLUTION 1: Extracting Single Fold (Fold 1)"
echo "========================================================================"
python representation/extract_single_fold.py \
    --input "$EMBEDDINGS_FILE" \
    --fold 1

FOLD1_FILE="${EMBEDDINGS_FILE%.csv}_fold1_only.csv"

echo ""
echo "Creating visualizations for Fold 1 only..."
python representation/visualize_embeddings.py \
    --input "$FOLD1_FILE" \
    --output_dir single_fold_plots \
    --prefix fold1_corrected

echo ""
echo "========================================================================"
echo "SOLUTION 2: Aligning All Folds"
echo "========================================================================"
python representation/align_fold_embeddings.py \
    --input "$EMBEDDINGS_FILE"

ALIGNED_FILE="${EMBEDDINGS_FILE%.csv}_aligned.csv"

echo ""
echo "Creating visualizations for aligned embeddings..."
python representation/visualize_embeddings.py \
    --input "$ALIGNED_FILE" \
    --output_dir aligned_plots \
    --prefix aligned_corrected

echo ""
echo "========================================================================"
echo "Creating Comparison Plots (Before vs After)"
echo "========================================================================"

# Original (with 5 clusters)
echo "Generating original visualization (with 5-cluster artifact)..."
python representation/visualize_embeddings.py \
    --input "$EMBEDDINGS_FILE" \
    --output_dir comparison_plots \
    --prefix original_5clusters \
    --skip_tsne  # Just UMAP for comparison

echo ""
echo "========================================================================"
echo "✅ COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo ""
echo "  📁 single_fold_plots/"
echo "     └─ fold1_corrected_*.png        (20% of data, correct structure)"
echo ""
echo "  📁 aligned_plots/"
echo "     └─ aligned_corrected_*.png      (100% of data, aligned)"
echo ""
echo "  📁 comparison_plots/"
echo "     └─ original_5clusters_*.png     (original with 5-cluster artifact)"
echo ""
echo "========================================================================"
echo "INTERPRETING RESULTS"
echo "========================================================================"
echo ""
echo "Compare the plots:"
echo ""
echo "  original_5clusters_umap.png  → Shows 5 artificial clusters (WRONG)"
echo "  aligned_corrected_umap.png   → Shows true biological structure (RIGHT)"
echo ""
echo "What to look for in the corrected plots:"
echo ""
echo "  ✓ Clear separation of Primary (blue) vs Metastatic (red)"
echo "  ✓ Biological subtypes (2-3 clusters with mixed colors)"
echo "  ✓ Continuous gradient (no discrete clusters)"
echo ""
echo "All of these are valid biological patterns!"
echo ""
echo "========================================================================"
echo ""
echo "For detailed explanation, see:"
echo "  representation/UNDERSTANDING_FIVE_CLUSTERS.md"
echo ""
echo "========================================================================"
