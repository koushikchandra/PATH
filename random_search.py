#!/usr/bin/env python3
"""
Random Search for Soft Masking Hyperparameters

Faster than grid search - samples N random combinations instead of testing all.
Often finds good hyperparameters in 10-20% of the time of full grid search.

Usage:
    python random_search.py --n_trials 20    # Quick search (4-6 hours)
    python random_search.py --n_trials 50    # Medium search (10-15 hours)
    python random_search.py --n_trials 100   # Extensive search (20-30 hours)
"""

import subprocess
import json
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from parse_results import parse_experiment_results


# ============================================================================
# HYPERPARAMETER DISTRIBUTIONS
# ============================================================================

SEARCH_SPACE = {
    # Continuous parameters (sample from distribution)
    "lr": {
        "type": "log_uniform",
        "low": 1e-5,
        "high": 1e-3,
    },
    "dropout": {
        "type": "uniform",
        "low": 0.1,
        "high": 0.5,
    },
    "weight_decay": {
        "type": "log_uniform",
        "low": 1e-5,
        "high": 1e-2,
    },
    "lr_scheduler_factor": {
        "type": "uniform",
        "low": 0.3,
        "high": 0.8,
    },

    # Discrete parameters (sample from list)
    "batch": {
        "type": "choice",
        "choices": [8, 16, 32, 64],
    },
    "d_model": {
        "type": "choice",
        "choices": [32, 64, 128, 256],
    },
    "num_heads": {
        "type": "choice",
        "choices": [2, 4, 8],
    },
    "layers": {
        "type": "choice",
        "choices": [1, 2, 3, 4],
    },
    "use_focal_loss": {
        "type": "choice",
        "choices": [True, False],
    },
}


def sample_hyperparameters(search_space):
    """Sample a random hyperparameter configuration."""
    config = {}

    for param_name, param_config in search_space.items():
        if param_config["type"] == "choice":
            # Sample from discrete choices
            config[param_name] = random.choice(param_config["choices"])

        elif param_config["type"] == "uniform":
            # Sample from uniform distribution
            config[param_name] = random.uniform(
                param_config["low"],
                param_config["high"]
            )

        elif param_config["type"] == "log_uniform":
            # Sample from log-uniform distribution (better for learning rates)
            log_low = np.log(param_config["low"])
            log_high = np.log(param_config["high"])
            config[param_name] = np.exp(random.uniform(log_low, log_high))

    return config


def run_experiment(config, trial_id, results_dir):
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_id}")
    print(f"{'='*80}")
    print(f"Hyperparameters: {json.dumps(config, indent=2)}")
    print(f"{'='*80}\n")

    # Create unique output directory for this trial
    outdir = os.path.join(results_dir, f"trial_{trial_id:03d}")

    # Build command
    cmd = ["python", "soft_masking.py", "--outdir", outdir]

    # Add hyperparameters
    for param, value in config.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{param}")
        elif isinstance(value, float):
            cmd.extend([f"--{param}", f"{value:.6f}"])
        else:
            cmd.extend([f"--{param}", str(value)])

    # Run the experiment
    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse results from the output
        metrics = parse_results(outdir)

        return {
            "trial_id": trial_id,
            "config": config,
            "metrics": metrics,
            "status": "success",
            "outdir": outdir,
        }

    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial_id} FAILED!")
        print(f"Error: {e}")
        return {
            "trial_id": trial_id,
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
    rows = []
    for result in all_results:
        if result["status"] == "success" and result["metrics"] is not None:
            row = {
                "trial_id": result["trial_id"],
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
    csv_path = os.path.join(results_dir, "random_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")

    # Save full results as JSON
    json_path = os.path.join(results_dir, "random_search_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Full results saved to: {json_path}")

    return df


def print_summary(df, results_dir):
    """Print summary of random search results."""
    print("\n" + "="*80)
    print("🎲 RANDOM SEARCH RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal trials: {len(df)}")

    print("\n📊 TOP 5 CONFIGURATIONS (by Test F1):")
    print("-" * 80)

    # Select columns to display
    display_cols = [
        "trial_id",
        "test_f1", "test_auc", "test_aupr", "test_recall", "test_precision",
    ]

    # Print top 5
    print(df[display_cols].head(5).to_string(index=False))

    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION:")
    print("="*80)

    best_row = df.iloc[0]

    print(f"\nTrial ID: {best_row['trial_id']}")
    print(f"Output Directory: {best_row['outdir']}")

    print("\n📈 Metrics:")
    print(f"  - Test F1:        {best_row['test_f1']:.4f} ± {best_row['test_f1_std']:.4f}")
    print(f"  - Test AUC:       {best_row['test_auc']:.4f} ± {best_row['test_auc_std']:.4f}")
    print(f"  - Test AUPR:      {best_row['test_aupr']:.4f}")
    print(f"  - Test Recall:    {best_row['test_recall']:.4f}")
    print(f"  - Test Precision: {best_row['test_precision']:.4f}")

    print("\n⚙️ Hyperparameters:")
    hp_cols = [col for col in df.columns if col.startswith("hp_")]
    for col in hp_cols:
        param_name = col.replace("hp_", "")
        value = best_row[col]
        if isinstance(value, float):
            print(f"  --{param_name} {value:.6f}")
        else:
            print(f"  --{param_name} {value}")

    print("\n" + "="*80)

    # Generate command to re-run best config
    best_cmd = "python soft_masking.py"
    for col in hp_cols:
        param_name = col.replace("hp_", "")
        value = best_row[col]
        if isinstance(value, bool):
            if value:
                best_cmd += f" --{param_name}"
        elif isinstance(value, float):
            best_cmd += f" --{param_name} {value:.6f}"
        else:
            best_cmd += f" --{param_name} {value}"

    print("\n🚀 To re-run best configuration:")
    print(f"\n{best_cmd}\n")

    # Save best config to file
    best_config_file = os.path.join(results_dir, "best_config.sh")
    with open(best_config_file, 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# Best configuration found by random search\n")
        f.write(f"# Test F1: {best_row['test_f1']:.4f}\n")
        f.write(f"# Test AUC: {best_row['test_auc']:.4f}\n\n")
        f.write(best_cmd + "\n")

    os.chmod(best_config_file, 0o755)  # Make executable

    print(f"✅ Best config saved to: {best_config_file}")
    print("="*80 + "\n")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Random search for soft masking hyperparameters")
    ap.add_argument("--n_trials", type=int, default=20,
                    help="Number of random trials to run")
    ap.add_argument("--results_dir", default="random_search_results",
                    help="Directory to save all trial results")
    ap.add_argument("--start_from", type=int, default=0,
                    help="Resume from trial number (useful if interrupted)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    args = ap.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RANDOM SEARCH - {args.n_trials} TRIALS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Random seed: {args.seed}")

    # Estimate time
    estimated_time_per_trial = 20  # minutes (rough estimate for 5-fold CV)
    estimated_total_hours = (args.n_trials * estimated_time_per_trial) / 60
    print(f"\nEstimated total time: {estimated_total_hours:.1f} hours")

    # Run all trials
    all_results = []

    for trial_id in range(args.n_trials):
        if trial_id < args.start_from:
            print(f"Skipping trial {trial_id} (already completed)")
            continue

        # Sample random hyperparameters
        config = sample_hyperparameters(SEARCH_SPACE)

        # Run experiment
        result = run_experiment(config, trial_id, results_dir)
        all_results.append(result)

        # Save intermediate results after each trial
        df = save_results_table(all_results, results_dir)

        print(f"\n✅ Completed {trial_id+1}/{args.n_trials} trials")

        # Print current best
        if len(df) > 0:
            best_f1 = df.iloc[0]["test_f1"]
            print(f"🏆 Current best F1: {best_f1:.4f}")

    # Final summary
    df = save_results_table(all_results, results_dir)
    print_summary(df, results_dir)


if __name__ == "__main__":
    main()
