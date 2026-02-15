"""
Helper functions to parse results from soft_masking.py experiments
"""

import json
import os
import glob


def find_results_file(outdir):
    """Find the results JSON file in the output directory."""
    # Look for files matching the pattern
    pattern = os.path.join(outdir, "5fold_results_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Return the most recent file (in case multiple exist)
    return max(files, key=os.path.getctime)


def parse_experiment_results(outdir):
    """Parse results from an experiment output directory."""
    results_file = find_results_file(outdir)

    if not results_file:
        print(f"⚠️ Warning: No results file found in {outdir}")
        return None

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract summary metrics
        test_summary = results.get("test_summary", {})

        metrics = {
            "test_auc": test_summary.get("auc", {}).get("mean", 0.0),
            "test_auc_std": test_summary.get("auc", {}).get("std", 0.0),

            "test_f1": test_summary.get("f1_binary", {}).get("mean", 0.0),
            "test_f1_std": test_summary.get("f1_binary", {}).get("std", 0.0),

            "test_acc": test_summary.get("acc", {}).get("mean", 0.0),
            "test_acc_std": test_summary.get("acc", {}).get("std", 0.0),

            "test_precision": test_summary.get("precision", {}).get("mean", 0.0),
            "test_precision_std": test_summary.get("precision", {}).get("std", 0.0),

            "test_recall": test_summary.get("recall", {}).get("mean", 0.0),
            "test_recall_std": test_summary.get("recall", {}).get("std", 0.0),
        }

        # Also try to get AUPR if available in fold_details
        fold_details = results.get("fold_details", [])
        if fold_details and len(fold_details) > 0:
            # Check if test_aupr exists in the first fold
            if "test_aupr" in fold_details[0]:
                aupr_values = [fd.get("test_aupr", 0.0) for fd in fold_details]
                import numpy as np
                metrics["test_aupr"] = np.mean(aupr_values)
                metrics["test_aupr_std"] = np.std(aupr_values)
            else:
                metrics["test_aupr"] = 0.0
                metrics["test_aupr_std"] = 0.0
        else:
            metrics["test_aupr"] = 0.0
            metrics["test_aupr_std"] = 0.0

        return metrics

    except Exception as e:
        print(f"❌ Error parsing results from {results_file}: {e}")
        return None


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        outdir = sys.argv[1]
        metrics = parse_experiment_results(outdir)

        if metrics:
            print("\n📊 Parsed Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("❌ Failed to parse results")
    else:
        print("Usage: python parse_results.py <output_directory>")
