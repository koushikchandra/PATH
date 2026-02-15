#!/usr/bin/env python3
"""
Grid Search for Soft Masking Hyperparameters

Usage:
    python grid_search.py --mode quick     # Fast search (2-3 hours)
    python grid_search.py --mode medium    # Balanced search (6-8 hours)
    python grid_search.py --mode extensive # Full search (12-24 hours)
"""

import subprocess
import itertools
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from parse_results import parse_experiment_results

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

SEARCH_SPACES = {
    "quick": {
        # Focus on most impactful parameters
        "lr": [5e-5, 1e-4, 5e-4],
        "dropout": [0.2, 0.3],
        "batch": [16, 32],
        "use_focal_loss": [False, True],
    },

    "medium": {
        # More comprehensive search
        "lr": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.2, 0.3],
        "batch": [16, 32],
        "weight_decay": [1e-4, 5e-4, 1e-3],
        "use_focal_loss": [False, True],
    },

    "extensive": {
        # Full grid search (WARNING: many combinations!)
        "lr": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "batch": [8, 16, 32],
        "weight_decay": [1e-4, 5e-4, 1e-3],
        "d_model": [32, 64, 128],
        "num_heads": [2, 4, 8],
        "layers": [1, 2, 3],
        "use_focal_loss": [False, True],
        "lr_scheduler_factor": [0.3, 0.5, 0.7],
    }
}


def get_hyperparameter_combinations(mode="quick"):
    """Generate all hyperparameter combinations for grid search."""
    search_space = SEARCH_SPACES[mode]

    # Get all parameter names and their values
    param_names = list(search_space.keys())
    param_values = list(search_space.values())

    # Generate all combinations
    combinations = list(itertools.product(*param_values))

    # Convert to list of dicts
    hyperparameter_configs = []
    for combo in combinations:
        config = dict(zip(param_names, combo))
        hyperparameter_configs.append(config)

    return hyperparameter_configs


def run_experiment(config, experiment_id, results_dir):
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_id}")
    print(f"{'='*80}")
    print(f"Hyperparameters: {json.dumps(config, indent=2)}")
    print(f"{'='*80}\n")

    # Create unique output directory for this experiment
    outdir = os.path.join(results_dir, f"exp_{experiment_id:03d}")

    # Build command
    cmd = ["python", "soft_masking.py", "--outdir", outdir]

    # Add hyperparameters
    for param, value in config.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{param}")
        else:
            cmd.extend([f"--{param}", str(value)])

    # Run the experiment
    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse results from the output
        metrics = parse_results(outdir)

        return {
            "experiment_id": experiment_id,
            "config": config,
            "metrics": metrics,
            "status": "success",
            "outdir": outdir,
        }

    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {experiment_id} FAILED!")
        print(f"Error: {e}")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "metrics": None,
            "status": "failed",
            "error": str(e),
            "outdir": outdir,
        }


def parse_results(outdir):
    """Parse results from experiment output directory."""
    return parse_experiment_results(outdir)


def save_results_table(all_results, results_dir):
    """Save results as CSV and JSON for easy analysis."""

    # Flatten results for DataFrame
    rows = []
    for result in all_results:
        if result["status"] == "success" and result["metrics"] is not None:
            row = {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "outdir": result["outdir"],
            }

            # Add hyperparameters
            row.update({f"hp_{k}": v for k, v in result["config"].items()})

            # Add metrics
            row.update(result["metrics"])

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by test F1 score (descending)
    df = df.sort_values("test_f1", ascending=False)

    # Save to CSV
    csv_path = os.path.join(results_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")

    # Save full results as JSON
    json_path = os.path.join(results_dir, "grid_search_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Full results saved to: {json_path}")

    return df


def print_summary(df):
    """Print summary of grid search results."""
    print("\n" + "="*80)
    print("🏆 GRID SEARCH RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal experiments: {len(df)}")

    print("\n📊 TOP 5 CONFIGURATIONS (by Test F1):")
    print("-" * 80)

    # Select columns to display
    display_cols = [
        "experiment_id",
        "test_f1", "test_auc", "test_aupr", "test_recall", "test_precision",
    ]

    # Add hyperparameter columns
    hp_cols = [col for col in df.columns if col.startswith("hp_")]
    display_cols.extend(hp_cols)

    # Print top 5
    print(df[display_cols].head(5).to_string(index=False))

    print("\n" + "="*80)
    print("🎯 BEST CONFIGURATION:")
    print("="*80)

    best_row = df.iloc[0]

    print(f"\nExperiment ID: {best_row['experiment_id']}")
    print(f"Output Directory: {best_row['outdir']}")

    print("\n📈 Metrics:")
    print(f"  - Test F1:        {best_row['test_f1']:.4f} ± {best_row['test_f1_std']:.4f}")
    print(f"  - Test AUC:       {best_row['test_auc']:.4f} ± {best_row['test_auc_std']:.4f}")
    print(f"  - Test AUPR:      {best_row['test_aupr']:.4f}")
    print(f"  - Test Recall:    {best_row['test_recall']:.4f}")
    print(f"  - Test Precision: {best_row['test_precision']:.4f}")

    print("\n⚙️ Hyperparameters:")
    for col in hp_cols:
        param_name = col.replace("hp_", "")
        print(f"  --{param_name} {best_row[col]}")

    print("\n" + "="*80)

    # Generate command to re-run best config
    best_cmd = "python soft_masking.py"
    for col in hp_cols:
        param_name = col.replace("hp_", "")
        value = best_row[col]
        if isinstance(value, bool):
            if value:
                best_cmd += f" --{param_name}"
        else:
            best_cmd += f" --{param_name} {value}"

    print("\n🚀 To re-run best configuration:")
    print(f"\n{best_cmd}\n")

    # Save best config to file
    best_config_file = os.path.join(os.path.dirname(best_row['outdir']), "best_config.txt")
    with open(best_config_file, 'w') as f:
        f.write(f"# Best configuration found by grid search\n")
        f.write(f"# Test F1: {best_row['test_f1']:.4f}\n")
        f.write(f"# Test AUC: {best_row['test_auc']:.4f}\n\n")
        f.write(best_cmd + "\n")

    print(f"✅ Best config command saved to: {best_config_file}")
    print("="*80 + "\n")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Grid search for soft masking hyperparameters")
    ap.add_argument("--mode", choices=["quick", "medium", "extensive"], default="quick",
                    help="Search mode: quick (12 combos), medium (48 combos), extensive (100+ combos)")
    ap.add_argument("--results_dir", default="grid_search_results",
                    help="Directory to save all experiment results")
    ap.add_argument("--start_from", type=int, default=0,
                    help="Resume from experiment number (useful if interrupted)")

    args = ap.parse_args()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.results_dir}_{args.mode}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"GRID SEARCH - MODE: {args.mode.upper()}")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")

    # Generate hyperparameter combinations
    configs = get_hyperparameter_combinations(args.mode)

    print(f"\nTotal configurations to test: {len(configs)}")

    # Estimate time
    estimated_time_per_exp = 20  # minutes (rough estimate for 5-fold CV)
    estimated_total_hours = (len(configs) * estimated_time_per_exp) / 60
    print(f"Estimated total time: {estimated_total_hours:.1f} hours")

    # Confirm before starting
    if len(configs) > 20:
        response = input(f"\n⚠️ This will run {len(configs)} experiments (~{estimated_total_hours:.1f} hours). Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run all experiments
    all_results = []

    for i, config in enumerate(configs):
        if i < args.start_from:
            print(f"Skipping experiment {i} (already completed)")
            continue

        result = run_experiment(config, i, results_dir)
        all_results.append(result)

        # Save intermediate results after each experiment
        df = save_results_table(all_results, results_dir)

        print(f"\n✅ Completed {i+1}/{len(configs)} experiments")

        # Print current best
        if len(df) > 0:
            best_f1 = df.iloc[0]["test_f1"]
            print(f"🏆 Current best F1: {best_f1:.4f}")

    # Final summary
    df = save_results_table(all_results, results_dir)
    print_summary(df)


if __name__ == "__main__":
    main()
