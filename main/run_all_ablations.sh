#!/usr/bin/env bash
# =============================================================================
# run_all_ablations.sh
# Runs baseline + all 4 ablation scripts and saves a summary of
# ACC, AUC, F1, Precision, Recall to ablation_results_summary.txt
# =============================================================================

set -euo pipefail

# ---------- Configurable args (edit here or override via env) ----------
DATA_DIR="${DATA_DIR:-data}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
D_MODEL="${D_MODEL:-64}"
LAYERS="${LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
PYTHON="${PYTHON:-python}"
# -----------------------------------------------------------------------

RESULTS_FILE="ablation_results_summary.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

COMMON_ARGS="--data_dir ${DATA_DIR} \
             --epochs ${EPOCHS} \
             --batch ${BATCH} \
             --lr ${LR} \
             --d_model ${D_MODEL} \
             --layers ${LAYERS} \
             --num_heads ${NUM_HEADS} \
             --dropout ${DROPOUT} \
             --random_state ${RANDOM_STATE}"

declare -a SCRIPTS=(
    "main.py|outputs_baseline|Baseline (full model)"
    "ablation1_no_graph_transformer.py|outputs_ablation1_no_graph_transformer|Ablation 1: No Graph Transformer"
    "ablation2_hard_mask.py|outputs_ablation2_hard_mask|Ablation 2: Hard Mask"
    "ablation3_no_edge_features.py|outputs_ablation3_no_edge_features|Ablation 3: No Edge Features"
    "ablation4_full_graph.py|outputs_ablation4_full_graph|Ablation 4: Full Graph"
)

# -----------------------------------------------------------------------
# Helper: parse mean ± std from a results JSON file
# -----------------------------------------------------------------------
parse_metric() {
    local json_file="$1"
    local metric="$2"
    # Extract mean and std from test_summary block
    python -c "
import json, sys
with open('${json_file}') as f:
    d = json.load(f)
ts = d.get('test_summary', {})
m = ts.get('${metric}', {})
mean = m.get('mean', float('nan'))
std  = m.get('std',  float('nan'))
print(f'{mean:.4f} +/- {std:.4f}')
" 2>/dev/null || echo "N/A"
}

# -----------------------------------------------------------------------
# Write header
# -----------------------------------------------------------------------
{
    echo "============================================================"
    echo "  ABLATION STUDY — TEST SET RESULTS SUMMARY"
    echo "  Generated : ${TIMESTAMP}"
    echo "  Data dir  : ${DATA_DIR}"
    echo "  Epochs    : ${EPOCHS} | Batch: ${BATCH} | LR: ${LR}"
    echo "  d_model   : ${D_MODEL} | Layers: ${LAYERS} | Heads: ${NUM_HEADS}"
    echo "  Dropout   : ${DROPOUT} | Seed: ${RANDOM_STATE}"
    echo "============================================================"
    echo ""
    printf "%-45s | %-20s | %-20s | %-20s | %-20s | %-20s\n" \
        "Experiment" "ACC" "AUC" "F1" "Precision" "Recall"
    printf '%s\n' "$(printf '%.0s-' {1..155})"
} > "${RESULTS_FILE}"

# -----------------------------------------------------------------------
# Run each script
# -----------------------------------------------------------------------
for entry in "${SCRIPTS[@]}"; do
    IFS='|' read -r script outdir label <<< "${entry}"
    script_path="${SCRIPT_DIR}/${script}"

    echo ""
    echo "============================================================"
    echo " Running: ${label}"
    echo " Script : ${script_path}"
    echo " Outdir : ${outdir}"
    echo "============================================================"

    if [ ! -f "${script_path}" ]; then
        echo "  [SKIP] ${script_path} not found."
        printf "%-45s | %-20s | %-20s | %-20s | %-20s | %-20s\n" \
            "${label}" "SKIP" "SKIP" "SKIP" "SKIP" "SKIP" >> "${RESULTS_FILE}"
        continue
    fi

    log_file="${SCRIPT_DIR}/${outdir}_run.log"

    if ${PYTHON} "${script_path}" ${COMMON_ARGS} --outdir "${outdir}" \
            > "${log_file}" 2>&1; then
        echo "  [OK] Completed. Log: ${log_file}"

        # Find the latest results JSON in the output dir
        json_file=$(ls -t "${SCRIPT_DIR}/${outdir}"/5fold_results_*.json 2>/dev/null | head -1 || true)

        if [ -z "${json_file}" ]; then
            echo "  [WARN] No results JSON found in ${outdir}."
            printf "%-45s | %-20s | %-20s | %-20s | %-20s | %-20s\n" \
                "${label}" "NO JSON" "NO JSON" "NO JSON" "NO JSON" "NO JSON" >> "${RESULTS_FILE}"
        else
            acc=$(parse_metric "${json_file}" "acc")
            auc=$(parse_metric "${json_file}" "auc")
            f1=$(parse_metric "${json_file}" "f1_binary")
            prec=$(parse_metric "${json_file}" "precision")
            rec=$(parse_metric "${json_file}" "recall")

            printf "%-45s | %-20s | %-20s | %-20s | %-20s | %-20s\n" \
                "${label}" "${acc}" "${auc}" "${f1}" "${prec}" "${rec}" >> "${RESULTS_FILE}"

            echo "  ACC=${acc}  AUC=${auc}  F1=${f1}  Prec=${prec}  Rec=${rec}"
        fi
    else
        echo "  [FAIL] Script exited with error. Check log: ${log_file}"
        printf "%-45s | %-20s | %-20s | %-20s | %-20s | %-20s\n" \
            "${label}" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "${RESULTS_FILE}"
    fi
done

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
{
    printf '%s\n' "$(printf '%.0s-' {1..155})"
    echo ""
    echo "Metrics reported as:  mean +/- std  across 5 folds (test set)"
    echo "Individual run logs : <outdir>_run.log"
    echo "Full JSON results   : <outdir>/5fold_results_*.json"
    echo "============================================================"
} >> "${RESULTS_FILE}"

echo ""
echo "============================================================"
echo " All runs complete."
echo " Summary saved to: ${RESULTS_FILE}"
echo "============================================================"
