#!/usr/bin/env python3

import os, re, json, math, argparse, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import copy
from statistics import NormalDist

try:
    from scipy.stats import ttest_ind as scipy_ttest_ind
    HAS_SCIPY = True
except Exception:
    scipy_ttest_ind = None
    HAS_SCIPY = False




def aggregate_pathway_rankings_across_folds(outdir, n_folds=5):
    """
    Aggregate pathway importance rankings across all folds
    
    Args:
        outdir: Output directory containing fold results
        n_folds: Number of folds
    
    Returns:
        Dictionary with aggregated pathway statistics
    """
    pathway_dir = os.path.join(outdir, "pathway_importance")
    
    if not os.path.exists(pathway_dir):
        print(f"Warning: Pathway importance directory not found: {pathway_dir}")
        return None
    
    print(f"\n{'='*70}")
    print("AGGREGATING PATHWAY IMPORTANCE ACROSS ALL FOLDS")
    print(f"{'='*70}")
    
    # Collect data from all folds
    all_fold_data = []
    for fold in range(1, n_folds + 1):
        csv_file = os.path.join(pathway_dir, f"pathway_importance_fold{fold}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_fold_data.append(df)
            print(f"Loaded fold {fold} pathway data: {len(df)} pathways")
        else:
            print(f"Warning: Missing fold {fold} data: {csv_file}")
    
    if not all_fold_data:
        print("Error: No fold data found!")
        return None
    
    # Combine all fold data
    combined_df = pd.concat(all_fold_data, ignore_index=True)
    
    # Calculate aggregate statistics for each pathway
    pathway_stats = combined_df.groupby('pathway_idx').agg({
        'pathway_name': 'first',
        'importance_score': ['mean', 'std', 'min', 'max'],
        'rank': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    pathway_stats.columns = [
        'pathway_idx', 'pathway_name',
        'mean_importance', 'std_importance', 'min_importance', 'max_importance',
        'mean_rank', 'std_rank', 'min_rank', 'max_rank'
    ]
    
    # Sort by mean importance (descending)
    pathway_stats = pathway_stats.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    pathway_stats['overall_rank'] = range(1, len(pathway_stats) + 1)
    
    # Display top 20 pathways
    print(f"\nTop 20 Most Consistently Important Pathways (Across {n_folds} Folds):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Pathway':<15} {'Name':<35} {'Mean Importance':<18} {'Mean Rank':<12}")
    print("-" * 100)
    
    for _, row in pathway_stats.head(20).iterrows():
        display_name = row['pathway_name'][:33] + "..." if len(row['pathway_name']) > 35 else row['pathway_name']
        importance_str = f"{row['mean_importance']:.6f} ± {row['std_importance']:.6f}"
        rank_str = f"{row['mean_rank']:.1f} ± {row['std_rank']:.1f}"
        print(f"{int(row['overall_rank']):<6} {int(row['pathway_idx']):<15} {display_name:<35} "
              f"{importance_str:<18} {rank_str:<12}")
    
    # Save aggregate results
    aggregate_csv = os.path.join(pathway_dir, "aggregate_pathway_importance.csv")
    pathway_stats.to_csv(aggregate_csv, index=False)
    print(f"\nAggregate pathway importance saved to: {aggregate_csv}")
    
    # Save top pathways to JSON
    top_pathways = []
    for _, row in pathway_stats.head(20).iterrows():
        top_pathways.append({
            'overall_rank': int(row['overall_rank']),
            'pathway_idx': int(row['pathway_idx']),
            'pathway_name': row['pathway_name'],
            'mean_importance': float(row['mean_importance']),
            'std_importance': float(row['std_importance']),
            'mean_rank': float(row['mean_rank']),
            'std_rank': float(row['std_rank'])
        })
    
    aggregate_json = os.path.join(pathway_dir, "aggregate_top_pathways.json")
    with open(aggregate_json, 'w') as f:
        json.dump({
            'n_folds': n_folds,
            'top_pathways': top_pathways,
            'total_pathways_analyzed': len(pathway_stats),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"Top aggregate pathways saved to: {aggregate_json}")
    
    print(f"{'='*70}\n")
    
    return {
        'pathway_stats': pathway_stats,
        'top_pathways': top_pathways,
        'n_folds': n_folds
    }

# +++++++++++++++++++++++++++
# AUROC and AUPR Visualization
# +++++++++++++++++++++++++++

def plot_aggregate_confusion_matrix(all_y_true, all_y_pred, save_path, n_folds, title="Aggregate Confusion Matrix"):
    """Plot confusion matrix with actual counts aggregated across all folds"""
    cms = []
    for yt, yp in zip(all_y_true, all_y_pred):
        cms.append(confusion_matrix(yt, yp))
    cms = np.array(cms)
    
    cm_mean = cms.mean(axis=0)
    cm_std = cms.std(axis=0)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_mean, annot=False, fmt='', cmap='Blues', 
                     xticklabels=['Primary', 'Metastatic'],
                     yticklabels=['Primary', 'Metastatic'],
                     cbar_kws={'label': 'Count'})
    
    for i in range(2):
        for j in range(2):
            count_text = f'{cm_mean[i, j]:.1f}\n±{cm_std[i, j]:.1f}'
            ax.text(j + 0.5, i + 0.5, count_text,
                   ha='center', va='center',
                   color='white' if cm_mean[i, j] > cm_mean.max()/2 else 'black',
                   fontsize=14, weight='bold')
    
    plt.title(f'{title}\n({n_folds} folds)', fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Aggregate confusion matrix saved to: {save_path}")

def plot_aggregate_roc_curve(all_y_true, all_y_probs, save_path, n_folds, title="Aggregate ROC Curve"):
    """Plot ROC curve with mean and confidence bands across folds"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))
    
    for yt, yp in zip(all_y_true, all_y_probs):
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2.5,
             label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='darkorange', alpha=0.2,
                     label='± 1 std. dev.')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title}\n({n_folds} folds)', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Aggregate ROC curve saved to: {save_path}")

def plot_aggregate_pr_curve(all_y_true, all_y_probs, save_path, n_folds, title="Aggregate Precision-Recall Curve"):
    """Plot PR curve with mean and confidence bands across folds"""
    precisions = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))
    
    for yt, yp in zip(all_y_true, all_y_probs):
        precision, recall, _ = precision_recall_curve(yt, yp)
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]
        interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
        precisions.append(interp_precision)
    
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_recall, mean_precision, color='blue', lw=2.5,
             label=f'Mean PR (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, 
                     color='blue', alpha=0.2, label='± 1 std. dev.')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{title}\n({n_folds} folds)', fontsize=14, weight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Aggregate PR curve saved to: {save_path}")

# ===============================
# Single Stratified Split Creation (70/10/20)
# ===============================

def create_train_val_test_split(y_arr, random_state=42, save_path="split_70_10_20.json"):
    """
    Create one stratified split: 70% train, 10% validation, 20% test.
    """
    all_idx = np.arange(len(y_arr))

    # First split: train+val (80%) vs test (20%)
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=y_arr,
        random_state=random_state
    )

    # Second split within train+val: train (70 total) vs val (10 total)
    y_train_val = y_arr[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,  # 10 / 80 = 0.125
        stratify=y_train_val,
        random_state=random_state
    )

    total_samples = len(y_arr)
    train_prop = len(train_idx) / total_samples
    val_prop = len(val_idx) / total_samples
    test_prop = len(test_idx) / total_samples

    train_classes = np.bincount(y_arr[train_idx])
    val_classes = np.bincount(y_arr[val_idx])
    test_classes = np.bincount(y_arr[test_idx])

    split_info = {
        'split_name': 'single_split_70_10_20',
        'random_state': random_state,
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist(),
        'proportions': {
            'train': train_prop,
            'val': val_prop,
            'test': test_prop
        },
        'sample_counts': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx),
            'total': total_samples
        },
        'class_distribution': {
            'train': train_classes.tolist(),
            'val': val_classes.tolist(),
            'test': test_classes.tolist()
        }
    }

    print("\nCreated stratified split (70/10/20):")
    print(f"  Train: {len(train_idx)} ({train_prop:.1%}) - Class dist: {train_classes}")
    print(f"  Val:   {len(val_idx)} ({val_prop:.1%}) - Class dist: {val_classes}")
    print(f"  Test:  {len(test_idx)} ({test_prop:.1%}) - Class dist: {test_classes}")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit file saved to {save_path}")
    return split_info

def load_train_val_test_split(load_path="split_70_10_20.json"):
    """Load previously saved train/val/test split."""
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r') as f:
        split_info = json.load(f)
    print(f"Train/val/test split loaded from {load_path}")
    return split_info

# ===============================
# Utility Functions (kept from original)
# ===============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def zscore_train_only(train_vals: np.ndarray, other_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_vals.mean(axis=0, keepdims=True)
    sd = train_vals.std(axis=0, keepdims=True) + 1e-8
    return (train_vals - mu) / sd, (other_vals - mu) / sd

def binarize_labels(labels: pd.Series) -> pd.Series:
    m = labels.astype(str).str.lower().str.strip()
    mapping = {"primary": 0, "metastatic": 1, "p": 0, "m": 1, "0": 0, "1": 1}
    y = m.map(mapping)
    if y.isna().any():
        y = pd.to_numeric(labels, errors="coerce")
    if y.isna().any():
        bad = labels[y.isna()]
        raise ValueError(f"Unrecognized labels (use Primary/Metastatic or 0/1). Offenders: {bad.unique()[:5]}")
    return y.astype(int)

# def load_tables(data_dir: str):
def load_tables(data_dir: str, use_full_graph: bool = False):
    """Load all required data tables"""
    mut_path = os.path.join(data_dir, "mutation_data.csv")
    cnv_path = os.path.join(data_dir, "cnv_data.csv")
    lab_path = os.path.join(data_dir, "patient_labels.csv")
    pw_path  = os.path.join(data_dir, "filtered_pathways.csv")
    adj_path = os.path.join(data_dir, "adjacency_matrix.csv")

    mut_df = pd.read_csv(mut_path)
    cnv_df = pd.read_csv(cnv_path)
    if mut_df.shape[1] < 2 or cnv_df.shape[1] < 2:
        raise ValueError("mutation_data.csv / cnv_data.csv must have patient id in col 1 and genes afterward.")
    mut_df = mut_df.set_index(mut_df.columns[0])
    cnv_df = cnv_df.set_index(cnv_df.columns[0])

    common_genes = [g for g in mut_df.columns if g in cnv_df.columns]
    if len(common_genes) == 0:
        raise ValueError("No overlapping gene columns between mutation_data.csv and cnv_data.csv.")
    mut_df = mut_df[common_genes]
    cnv_df = cnv_df[common_genes]

    lab_df = pd.read_csv(lab_path)
    if lab_df.shape[1] < 2:
        raise ValueError("patient_labels.csv must have two columns: patient_id,label")
    lab_df = lab_df.set_index(lab_df.columns[0])
    lab_df = lab_df.iloc[:, :1]

    mut_df.index = mut_df.index.astype(str)
    cnv_df.index = cnv_df.index.astype(str)
    lab_df.index = lab_df.index.astype(str)
    common_ids = mut_df.index.intersection(cnv_df.index).intersection(lab_df.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping patient IDs across mutation/cnv/labels.")
    mut_df = mut_df.loc[common_ids].sort_index()
    cnv_df = cnv_df.loc[common_ids].sort_index()
    lab_df = lab_df.loc[common_ids].sort_index()

    pw_df = pd.read_csv(pw_path)
    needed_cols = {"Pathway_ID", "Pathway_Name", "Genes"}
    if not needed_cols.issubset(set(pw_df.columns)):
        raise ValueError(f"filtered_pathways.csv must contain columns: {needed_cols}")
    
    gene_cols = list(common_genes)
    col_to_idx = {g: i for i, g in enumerate(gene_cols)}

    def parse_genes(s: str) -> List[str]:
        if pd.isna(s):
            return []
        toks = re.split(r'[,\s;|]+', str(s).strip())
        return [t for t in toks if t]

    pathway_gene_lists = []
    pathway_ids = []
    pathway_names = []
    for _, row in pw_df.iterrows():
        pid = str(row["Pathway_ID"])
        pname = str(row["Pathway_Name"])
        genes = parse_genes(row["Genes"])
        idxs = [col_to_idx[g] for g in genes if g in col_to_idx]
        if len(idxs) > 0:
            pathway_ids.append(pid)
            pathway_names.append(pname)
            pathway_gene_lists.append(sorted(set(idxs)))

    if len(pathway_gene_lists) == 0:
        raise ValueError("After mapping, no pathways contain genes present in your MUT/CNV tables.")

#     adj_df = pd.read_csv(adj_path)
#     adj_df = adj_df.set_index(adj_df.columns[0])
#     adj_df.index = adj_df.index.astype(str)
#     adj_df.columns = adj_df.columns.astype(str)
#     adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
#     adj_df = adj_df.reindex(index=pathway_ids, columns=pathway_ids).fillna(0)
#     A = adj_df.values.astype(np.float32)
    
#     A = 0.5 * (A + A.T)
#     A = A / (A.max() + 1e-8)
# Create adjacency matrix based on mode
    if use_full_graph:
        # FULL GRAPH MODE: Create fully connected adjacency matrix
        num_pathways = len(pathway_ids)
        A = np.ones((num_pathways, num_pathways), dtype=np.float32)
        np.fill_diagonal(A, 0.0)  # No self-loops
        print(f"Using FULL GRAPH mode: {num_pathways}x{num_pathways} fully connected adjacency matrix")
    else:
        # SPARSE GRAPH MODE: Load from file
        adj_df = pd.read_csv(adj_path)
        adj_df = adj_df.set_index(adj_df.columns[0])
        adj_df.index = adj_df.index.astype(str)
        adj_df.columns = adj_df.columns.astype(str)
        adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        adj_df = adj_df.reindex(index=pathway_ids, columns=pathway_ids).fillna(0)
        A = adj_df.values.astype(np.float32)

        A = 0.5 * (A + A.T)  # Symmetrize
        A = A / (A.max() + 1e-8)  # Normalize

        # Calculate sparsity for information
        num_edges = (A > 0).sum() - len(pathway_ids)  # exclude diagonal
        max_edges = len(pathway_ids) * (len(pathway_ids) - 1)
        sparsity = 100 * (1 - num_edges / max_edges)
        print(f"Using SPARSE GRAPH mode: {num_edges} edges, {sparsity:.1f}% sparsity")

    return mut_df, cnv_df, lab_df.iloc[:,0], pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols

def prepare_data_from_splits(mut_df, cnv_df, y, train_idx, val_idx, test_idx):
    """Prepare train/val/test data from predefined indices"""
    mut_arr = mut_df.values.astype(np.float32)
    cnv_arr = cnv_df.values.astype(np.float32)
    y_vec = binarize_labels(y).values.astype(np.int64)
    
    mut_tr, mut_va, mut_te = mut_arr[train_idx], mut_arr[val_idx], mut_arr[test_idx]
    cnv_tr, cnv_va, cnv_te = cnv_arr[train_idx], cnv_arr[val_idx], cnv_arr[test_idx]
    y_tr, y_va, y_te = y_vec[train_idx], y_vec[val_idx], y_vec[test_idx]
    
    cnv_tr_norm, cnv_va_norm = zscore_train_only(cnv_tr, cnv_va)
    _, cnv_te_norm = zscore_train_only(cnv_tr, cnv_te)
    
    return (torch.from_numpy(mut_tr), torch.from_numpy(mut_va), torch.from_numpy(mut_te),
            torch.from_numpy(cnv_tr_norm), torch.from_numpy(cnv_va_norm), torch.from_numpy(cnv_te_norm),
            torch.from_numpy(y_tr), torch.from_numpy(y_va), torch.from_numpy(y_te),
            train_idx, val_idx, test_idx)

# ===============================
# Model Components (kept from original - truncated for brevity)
# ===============================

class GeneEncoderB(nn.Module):
    def __init__(self, num_genes, d=128, hidden=64, positive_gamma=True):
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, d)
        self.to_gammabeta = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2*d)
        )
        self.positive_gamma = positive_gamma
        self._init_identity()

    def _init_identity(self):
        last = self.to_gammabeta[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            d = last.bias.numel() // 2
            last.bias[:d].fill_(0.0)
            last.bias[d:].zero_()
        nn.init.normal_(self.gene_emb.weight, std=0.02)

    def forward(self, gene_ids, mut, cnv):
        e = self.gene_emb(gene_ids)
        x = torch.stack([mut, cnv], dim=-1)
        gb = self.to_gammabeta(x)
        gamma_raw, beta = gb.chunk(2, dim=-1)
        if self.positive_gamma:
            gamma = F.softplus(gamma_raw) + 1e-3
        else:
            gamma = 1.0 + 0.1 * gamma_raw
        h = gamma * e + beta
        return h

class PathwayAttentionPool(nn.Module):
    def __init__(self, d: int, num_pathways: int, max_pathway_genes: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)
        self.u = nn.Parameter(torch.randn(d) * 0.02)
        self.b_p = nn.Parameter(torch.zeros(num_pathways, d))
        self.max_pathway_genes = max_pathway_genes
    
    def forward(self, H: torch.Tensor, pw_gene_idx: torch.Tensor, pw_gene_mask: torch.Tensor):
        B, G, d = H.shape
        P, M = pw_gene_idx.shape
        safe_idx = pw_gene_idx.clamp(min=0)
        H_exp = H.unsqueeze(1).expand(B, P, G, d)
        idx = safe_idx.view(1, P, M, 1).expand(B, P, M, d)
        Hp = torch.gather(H_exp, dim=2, index=idx)
        scores_in = torch.tanh(self.W(Hp) + self.b_p.view(1, P, 1, d))
        scores = torch.einsum('bpmd,d->bpm', scores_in, self.u)
        gene_mask = pw_gene_mask.view(1, P, M).expand(B, P, M)
        scores = scores.masked_fill(~gene_mask, float('-inf'))
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~gene_mask, 0.0)
        Z = torch.einsum('bpm,bpmd->bpd', alpha, Hp)
        return Z, alpha

class EdgeAwareGraphTransformerBlock(nn.Module):
    def __init__(self, d: int, num_heads: int = 4, ff_dim: Optional[int] = None, 
                 dropout: float = 0.1, edge_dim: int = 1, use_batch_norm: bool = True):
        super().__init__()
        if ff_dim is None: 
            ff_dim = 4 * d
        
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_dim), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d), 
            nn.Dropout(dropout)
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(d)
            self.norm2 = nn.BatchNorm1d(d)
        else:
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)

        self.edge_proj = nn.Linear(edge_dim, num_heads, bias=True)
        self.edge_ff = nn.Sequential(
            nn.Linear(edge_dim, edge_dim), 
            nn.GELU(), 
            nn.Linear(edge_dim, edge_dim)
        )
        self.edge_bn = nn.BatchNorm1d(edge_dim)

    def _apply_norm(self, x: torch.Tensor, norm_layer) -> torch.Tensor:
        if self.use_batch_norm:
            B, P, d = x.shape
            return norm_layer(x.reshape(B * P, d)).reshape(B, P, d)
        else:
            return norm_layer(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                edge_feat: Optional[torch.Tensor] = None):
        B, P, d = x.shape
        H = self.attn.num_heads

        gain = None
        if edge_feat is not None:
            if edge_feat.dim() == 3:
                edge_feat = edge_feat.unsqueeze(0).expand(B, -1, -1, -1)
            
            head_gains = F.softplus(self.edge_proj(edge_feat))
            log_gain = torch.log(head_gains + 1e-8)
            gain = log_gain.mean(dim=-1)

        final_mask = None
        if attn_mask is not None or gain is not None:
            if attn_mask is not None:
                base_mask = attn_mask
                if base_mask.dim() == 2:
                    base_mask = base_mask.unsqueeze(0).expand(B * H, -1, -1)
                elif base_mask.dim() == 3 and base_mask.size(0) == B:
                    base_mask = base_mask.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, P, P)
            else:
                base_mask = torch.zeros(B * H, P, P, device=x.device)
            
            if gain is not None:
                gain_expanded = gain.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, P, P)
                final_mask = base_mask + gain_expanded
            else:
                final_mask = base_mask

        attn_out, attn_w = self.attn(x, x, x, attn_mask=final_mask)
        x = self._apply_norm(x + attn_out, self.norm1)
        x = self._apply_norm(x + self.ff(x), self.norm2)

        updated_edge_feat = edge_feat
        if edge_feat is not None:
            B_e, P_e, _, E = edge_feat.shape
            ef = edge_feat.reshape(-1, E)
            ef2 = self.edge_ff(ef)
            ef2 = self.edge_bn(ef2)
            updated_edge_feat = ef2.reshape(B_e, P_e, P_e, E)

        return x, attn_w, updated_edge_feat

class PathwayGraphTransformer(nn.Module):
    def __init__(self, num_genes, num_pathways, max_pathway_genes,
                 d=64, layers=2, num_heads=4, dropout=0.2, use_film=True,
                 use_edge_mask=True, use_edge_bias=True, pe_dim=16,
                 use_batch_norm=True, use_edge_aware_blocks=True, 
                 full_graph_attention=False):
        super().__init__()
        self.num_genes = num_genes
        self.num_pathways = num_pathways
        self.max_pathway_genes = max_pathway_genes
        self.use_edge_mask = use_edge_mask
        self.use_edge_bias = use_edge_bias
        self.pe_dim = pe_dim
        self.use_batch_norm = use_batch_norm
        self.use_edge_aware_blocks = use_edge_aware_blocks
        self.full_graph_attention = full_graph_attention

        self.gene_enc = GeneEncoderB(num_genes, d=d, hidden=64)
        self.pw_pool = PathwayAttentionPool(d=d, num_pathways=num_pathways, max_pathway_genes=max_pathway_genes)

        in_pe = pe_dim + 1
        self.pe_proj = nn.Linear(in_pe, d)

        if use_edge_aware_blocks:
            self.layers = nn.ModuleList([
                EdgeAwareGraphTransformerBlock(d, num_heads=num_heads, dropout=dropout, 
                                             edge_dim=1, use_batch_norm=use_batch_norm) 
                for _ in range(layers)
            ])
        else:
            raise NotImplementedError("Regular blocks not included for brevity")

        self.pathway_attention = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        self.head = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 2))

        self.register_buffer("pw_gene_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("pw_gene_mask", torch.empty(0, dtype=torch.bool))
        self.register_buffer("A", torch.empty(0))
        self.register_buffer("attn_mask", torch.empty(0))
        self.register_buffer("PE", torch.empty(0))

    @staticmethod
    def build_pathway_index_and_mask(pathway_gene_lists: List[List[int]], num_genes: int):
        P = len(pathway_gene_lists)
        M = max(len(lst) for lst in pathway_gene_lists)
        idx = torch.full((P, M), -1, dtype=torch.long)
        mask = torch.zeros((P, M), dtype=torch.bool)
        for p, lst in enumerate(pathway_gene_lists):
            if len(lst) == 0: continue
            idx[p, :len(lst)] = torch.tensor(lst, dtype=torch.long).clamp(0, num_genes-1)
            mask[p, :len(lst)] = True
        return idx, mask

    @staticmethod
    def _laplacian_pe(A: torch.Tensor, k: int) -> torch.Tensor:
        deg = A.sum(dim=1).clamp(min=1e-8)
        Dm12 = torch.diag(torch.pow(deg, -0.5))
        L = torch.eye(A.size(0), device=A.device) - Dm12 @ A @ Dm12
        vals, vecs = torch.linalg.eigh(L)
        k = min(k, vecs.size(1))
        pe = vecs[:, :k]
        return pe

    def set_structures(self, pathway_gene_lists: List[List[int]], adjacency: torch.Tensor):
        idx, msk = self.build_pathway_index_and_mask(pathway_gene_lists, self.num_genes)
        dev = next(self.parameters()).device
        self.pw_gene_idx = idx.to(dev)
        self.pw_gene_mask = msk.to(dev)

        A = 0.5 * (adjacency + adjacency.T)
        A = A / (A.max() + 1e-8)
        A = A.to(dev)
        self.A = A

        deg = A.sum(1, keepdim=True) / (A.size(1) + 1e-8)
        pe = self._laplacian_pe(A, self.pe_dim)
        self.PE = torch.cat([pe.detach(), deg.detach()], dim=1)

        mask = torch.zeros_like(A)
        if self.full_graph_attention:
            mask = torch.zeros_like(A)
        else:
            if self.use_edge_mask:
                nonedge = (A <= 0)
                # CHANGED: Use soft penalty instead of -inf
                mask = mask.masked_fill(nonedge, -10.0)  # ← ONLY THIS LINE CHANGED!
                mask.fill_diagonal_(0.0)

        self.attn_mask = mask.detach()

    def forward(self, mut: torch.Tensor, cnv: torch.Tensor, return_extras: bool = False, 
                gene_ids: Optional[torch.Tensor] = None, return_attn: bool = False,
                return_edge: bool = False):
        assert self.pw_gene_idx.numel() > 0, "Call set_structures(...) first."
        B = mut.size(0)

        if gene_ids is None:
            gene_ids = torch.arange(self.num_genes, device=mut.device).unsqueeze(0).expand(B, -1)
        H = self.gene_enc(gene_ids, mut, cnv)

        Z, gene_alpha = self.pw_pool(H, self.pw_gene_idx, self.pw_gene_mask)

        pe = self.PE
        if self.training:
            flips = (torch.rand(self.pe_dim, device=pe.device) < 0.5).float() * 2 - 1
            pe_flipped = pe.clone()
            pe_flipped[:, :self.pe_dim] = pe[:, :self.pe_dim] * flips.unsqueeze(0)
            pe_projected = self.pe_proj(pe_flipped)
        else:
            pe_projected = self.pe_proj(pe)
        
        X = Z + pe_projected.unsqueeze(0)

        edge_feat = self.A.unsqueeze(-1) if self.use_edge_aware_blocks else None

        attn_weights = []
        edge_features = []
        for blk in self.layers:
            X, attn_w, edge_feat = blk(X, attn_mask=self.attn_mask, edge_feat=edge_feat)
            if return_attn:
                attn_weights.append(attn_w.detach())
            if return_edge:
                edge_features.append(edge_feat.detach() if edge_feat is not None else None)

        pw_scores = self.pathway_attention(X)
        pw_weights = F.softmax(pw_scores, dim=1)
        global_repr = torch.sum(X * pw_weights, dim=1)
        logits = self.head(global_repr)

        if return_extras or return_attn or return_edge:
            extras = {'gene_alpha': gene_alpha, 'pathway_weights': pw_weights.squeeze(-1)}
            if return_attn:
                extras['attn_weights'] = attn_weights
            if return_edge:
                extras['edge_features'] = edge_features
            return logits, extras
        return logits

# ===============================
# Training and Evaluation (kept from original)
# ===============================

class GenePatientDataset(Dataset):
    def __init__(self, mut: torch.Tensor, cnv: torch.Tensor, y: torch.Tensor):
        self.mut, self.cnv, self.y = mut, cnv, y
    def __len__(self): return self.mut.size(0)
    def __getitem__(self, i): return self.mut[i], self.cnv[i], self.y[i]

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False, ncols=80)
    for mut_b, cnv_b, y_b in pbar:
        mut_b, cnv_b, y_b = mut_b.to(device), cnv_b.to(device), y_b.to(device)
        opt.zero_grad()
        logits = model(mut_b, cnv_b)
        loss = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * y_b.size(0)
        pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    all_y, all_p1 = [], []
    pbar = tqdm(loader, desc="Validating", leave=False, ncols=80)
    for mut_b, cnv_b, y_b in pbar:
        mut_b, cnv_b, y_b = mut_b.to(device), cnv_b.to(device), y_b.to(device)
        logits = model(mut_b, cnv_b)
        probs = logits.softmax(dim=-1)[:,1]
        all_y.append(y_b.cpu().numpy())
        all_p1.append(probs.cpu().numpy())
    
    y_true = np.concatenate(all_y) if all_y else np.array([])
    p1 = np.concatenate(all_p1) if all_p1 else np.array([])
    
    prec, rec, thr = precision_recall_curve(y_true, p1)
    if thr.size:
        f1s = (2*prec[:-1]*rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_thr = float(thr[int(np.nanargmax(f1s))])
    else:
        best_thr = 0.5
    pred = (p1 >= best_thr).astype(int)
    
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, p1) if y_true.size>0 and len(np.unique(y_true))>1 else float("nan")
    aupr = average_precision_score(y_true, p1) if y_true.size>0 and len(np.unique(y_true))>1 else float("nan")
    
    f1_binary = f1_score(y_true, pred, average='binary', zero_division=0)
    precision = precision_score(y_true, pred, average='binary', zero_division=0)
    recall = recall_score(y_true, pred, average='binary', zero_division=0)
    
    return {
        "acc": acc,
        "auc": auc,
        "aupr": aupr,
        "f1_binary": f1_binary,
        "precision": precision,
        "recall": recall,
        "y_true": y_true,
        "y_pred": pred,
        "y_probs": p1
    }

@torch.no_grad()
def extract_pathway_attention_matrix(model, loader, device):
    """Aggregate pathway-pathway attention into a single matrix."""
    model.eval()
    total = None
    count = 0
    for mut_b, cnv_b, _ in loader:
        mut_b, cnv_b = mut_b.to(device), cnv_b.to(device)
        logits, extras = model(mut_b, cnv_b, return_attn=True)
        attn_list = extras.get("attn_weights", [])
        if not attn_list:
            continue
        layer_mean = torch.stack(attn_list, dim=0).mean(dim=0)
        batch_sum = layer_mean.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += layer_mean.size(0)
    if total is None or count == 0:
        return None
    return (total / count).detach().cpu().numpy()

def save_pathway_matrix_csv(matrix, pathway_ids, pathway_names, save_path):
    df = pd.DataFrame(matrix, index=pathway_ids, columns=pathway_ids)
    df.insert(0, "Pathway_Name", pathway_names)
    df.index.name = "Pathway_ID"
    df.to_csv(save_path)

def save_named_matrix_csv(matrix, row_ids, row_names, col_ids, save_path):
    df = pd.DataFrame(matrix, index=row_ids, columns=col_ids)
    df.insert(0, "Pathway_Name", row_names)
    df.index.name = "Pathway_ID"
    df.to_csv(save_path)

def minmax_normalize_matrix(matrix, min_val=None, max_val=None, eps=1e-12):
    """Min-max normalize matrix to [0, 1]."""
    mat = np.asarray(matrix, dtype=np.float32)
    if min_val is None:
        min_val = float(np.min(mat))
    if max_val is None:
        max_val = float(np.max(mat))
    denom = max(max_val - min_val, eps)
    norm = (mat - min_val) / denom
    return np.clip(norm, 0.0, 1.0), min_val, max_val

def normalize_primary_metastatic_pair(primary_mat, metastatic_mat):
    """Normalize primary and metastatic matrices using a shared min/max scale."""
    global_min = float(min(np.min(primary_mat), np.min(metastatic_mat)))
    global_max = float(max(np.max(primary_mat), np.max(metastatic_mat)))
    primary_norm, _, _ = minmax_normalize_matrix(primary_mat, min_val=global_min, max_val=global_max)
    metastatic_norm, _, _ = minmax_normalize_matrix(metastatic_mat, min_val=global_min, max_val=global_max)
    return primary_norm, metastatic_norm, global_min, global_max

def rank_pathways_by_interaction_change(primary_mat, metastatic_mat):
    """
    Rank pathways by total absolute change in their incoming and outgoing interactions.
    Higher score means the pathway's crosstalk pattern changes more between states.
    """
    delta = np.abs(np.asarray(metastatic_mat) - np.asarray(primary_mat))
    np.fill_diagonal(delta, 0.0)
    change_scores = delta.sum(axis=0) + delta.sum(axis=1)
    ranking = np.argsort(-change_scores)
    return ranking, change_scores

def _short_clean_pathway_label(name: str, max_len: int = 10) -> str:
    """Create very short labels and remove numbers from pathway names."""
    cleaned = re.sub(r"\d+", "", str(name))
    cleaned = re.sub(r"[_\-/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Pathway"

    words = cleaned.split()
    if len(words) == 1:
        label = words[0]
    else:
        # Keep first 3 words, each shortened for compact axis labels
        label = "".join(w[:3] for w in words[:3])
    return label[:max_len]

def plot_crosstalk_heatmap(matrix, labels, title, save_path, cmap="RdBu_r", center=0.0, vmin=None, vmax=None):
    plt.figure(figsize=(3.5, 3.5))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Attention Weight"}
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches='tight')
    plt.close()

@torch.no_grad()
def extract_pathway_attention_by_class(model, loader, device):
    """Aggregate pathway-pathway attention separately by class label (0=Primary, 1=Metastatic)."""
    model.eval()
    totals = {0: None, 1: None}
    counts = {0: 0, 1: 0}

    for mut_b, cnv_b, y_b in loader:
        mut_b = mut_b.to(device)
        cnv_b = cnv_b.to(device)
        logits, extras = model(mut_b, cnv_b, return_attn=True)
        attn_list = extras.get("attn_weights", [])
        if not attn_list:
            continue

        layer_mean = torch.stack(attn_list, dim=0).mean(dim=0)  # [B, P, P]
        y_np = y_b.detach().cpu().numpy()

        for cls in (0, 1):
            cls_mask = (y_np == cls)
            if not np.any(cls_mask):
                continue
            cls_tensor_mask = torch.from_numpy(cls_mask).to(layer_mean.device)
            cls_batch = layer_mean[cls_tensor_mask]  # [n_cls, P, P]
            cls_sum = cls_batch.sum(dim=0)

            if totals[cls] is None:
                totals[cls] = cls_sum
            else:
                totals[cls] = totals[cls] + cls_sum
            counts[cls] += int(cls_batch.size(0))

    out = {}
    for cls in (0, 1):
        if totals[cls] is not None and counts[cls] > 0:
            out[cls] = (totals[cls] / counts[cls]).detach().cpu().numpy()
        else:
            out[cls] = None
    return out, counts

@torch.no_grad()
def extract_pathway_attention_samples(model, loader, device):
    """Return per-sample pathway attention matrices and corresponding labels."""
    model.eval()
    all_attn = []
    all_labels = []

    for mut_b, cnv_b, y_b in loader:
        mut_b = mut_b.to(device)
        cnv_b = cnv_b.to(device)
        _, extras = model(mut_b, cnv_b, return_attn=True)
        attn_list = extras.get("attn_weights", [])
        if not attn_list:
            continue
        layer_mean = torch.stack(attn_list, dim=0).mean(dim=0)  # [B, P, P]
        all_attn.append(layer_mean.detach().cpu().numpy())
        all_labels.append(y_b.detach().cpu().numpy())

    if not all_attn:
        return None, None
    return np.concatenate(all_attn, axis=0), np.concatenate(all_labels, axis=0)

def benjamini_hochberg_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for 1D array of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return pvals.copy()

    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals_ranked = ranked * n / (np.arange(1, n + 1))
    qvals_ranked = np.minimum.accumulate(qvals_ranked[::-1])[::-1]
    qvals_ranked = np.clip(qvals_ranked, 0.0, 1.0)

    qvals = np.empty_like(qvals_ranked)
    qvals[order] = qvals_ranked
    return qvals

def compute_and_save_edge_fdr(primary_samples, metastatic_samples, pathway_ids, pathway_names, outdir,
                              prefix="split", alpha=0.05):
    """
    Compute per-edge p-values (Welch t-test) and FDR-adjusted q-values.
    Tests directed, off-diagonal edges i->j.
    """
    if primary_samples is None or metastatic_samples is None:
        print(f"Warning: Missing sample-level matrices for {prefix}, skipping edge FDR.")
        return None

    os.makedirs(outdir, exist_ok=True)
    if primary_samples.ndim != 3 or metastatic_samples.ndim != 3:
        print(f"Warning: Unexpected sample shapes for {prefix}, skipping edge FDR.")
        return None

    n0, p0, p1 = primary_samples.shape
    n1, q0, q1 = metastatic_samples.shape
    if (p0 != p1) or (q0 != q1) or (p0 != q0):
        print(f"Warning: Matrix size mismatch for {prefix}, skipping edge FDR.")
        return None

    means_primary = primary_samples.mean(axis=0)
    means_met = metastatic_samples.mean(axis=0)
    delta = means_met - means_primary

    # Welch test per edge across samples: shape [P, P]
    # Uses scipy if available; otherwise uses normal-approximation for p-values.
    if HAS_SCIPY:
        _, pvals_mat = scipy_ttest_ind(
            metastatic_samples,
            primary_samples,
            axis=0,
            equal_var=False,
            nan_policy='omit'
        )
        pvals_mat = np.nan_to_num(pvals_mat, nan=1.0, posinf=1.0, neginf=1.0)
    else:
        n_met = metastatic_samples.shape[0]
        n_pri = primary_samples.shape[0]
        var_met = metastatic_samples.var(axis=0, ddof=1)
        var_pri = primary_samples.var(axis=0, ddof=1)
        denom = np.sqrt((var_met / max(n_met, 1)) + (var_pri / max(n_pri, 1)) + 1e-12)
        t_stat = (means_met - means_primary) / denom
        abs_t = np.abs(t_stat)
        normal = NormalDist()
        pvals_mat = np.vectorize(lambda x: 2.0 * (1.0 - normal.cdf(float(x))))(abs_t)
        pvals_mat = np.nan_to_num(pvals_mat, nan=1.0, posinf=1.0, neginf=1.0)
        print("Info: scipy not found; using normal-approximation p-values for Welch statistics.")

    P = len(pathway_ids)
    offdiag_mask = ~np.eye(P, dtype=bool)
    pvals_flat = pvals_mat[offdiag_mask]
    qvals_flat = benjamini_hochberg_fdr(pvals_flat)

    qvals_mat = np.ones((P, P), dtype=float)
    qvals_mat[offdiag_mask] = qvals_flat

    edge_rows = []
    row_idx, col_idx = np.where(offdiag_mask)
    for i, j in zip(row_idx, col_idx):
        edge_rows.append({
            "source_idx": int(i),
            "source_id": pathway_ids[i],
            "source_name": pathway_names[i],
            "target_idx": int(j),
            "target_id": pathway_ids[j],
            "target_name": pathway_names[j],
            "mean_primary": float(means_primary[i, j]),
            "mean_metastatic": float(means_met[i, j]),
            "delta_met_minus_primary": float(delta[i, j]),
            "p_value": float(pvals_mat[i, j]),
            "q_value": float(qvals_mat[i, j]),
            "significant_fdr": bool(qvals_mat[i, j] < alpha)
        })

    edge_df = pd.DataFrame(edge_rows).sort_values(
        by=["q_value", "p_value", "delta_met_minus_primary"],
        ascending=[True, True, False]
    )
    edge_csv = os.path.join(outdir, f"{prefix}_edge_stats_fdr.csv")
    edge_df.to_csv(edge_csv, index=False)

    qval_csv = os.path.join(outdir, f"{prefix}_edge_qvalues_matrix.csv")
    pval_csv = os.path.join(outdir, f"{prefix}_edge_pvalues_matrix.csv")
    save_named_matrix_csv(qvals_mat, pathway_ids, pathway_names, pathway_ids, qval_csv)
    save_named_matrix_csv(pvals_mat, pathway_ids, pathway_names, pathway_ids, pval_csv)

    sig_count = int((qvals_flat < alpha).sum())
    total_count = int(qvals_flat.size)
    print(f"Edge FDR ({prefix}): {sig_count}/{total_count} off-diagonal edges significant at q<{alpha}.")
    print(f"Per-edge stats saved to: {edge_csv}")
    print(f"Per-edge q-value matrix saved to: {qval_csv}")
    print(f"Per-edge p-value matrix saved to: {pval_csv}")

    return {
        "n_primary": int(n0),
        "n_metastatic": int(n1),
        "n_edges_tested": total_count,
        "n_significant": sig_count,
        "alpha": float(alpha),
        "edge_csv": edge_csv,
        "qval_csv": qval_csv,
        "pval_csv": pval_csv
    }

def create_topk_crosstalk_outputs(primary_mat, metastatic_mat, selected_indices,
                                  pathway_ids, pathway_names, outdir, prefix):
    """Save class-specific and differential crosstalk matrices for selected pathways."""
    if primary_mat is None or metastatic_mat is None:
        print(f"Warning: Missing class-specific matrix for {prefix}, skipping crosstalk outputs.")
        return

    os.makedirs(outdir, exist_ok=True)
    idx = np.array(selected_indices, dtype=int)

    primary_sub = primary_mat[np.ix_(idx, idx)]
    metastatic_sub = metastatic_mat[np.ix_(idx, idx)]
    delta_sub = metastatic_sub - primary_sub

    sel_ids = [pathway_ids[i] for i in idx]
    sel_names = [pathway_names[i] for i in idx]
    short_labels = [_short_clean_pathway_label(pathway_names[i], max_len=10) for i in idx]

    # Shared scale for primary/metastatic only
    pm_min = float(min(primary_sub.min(), metastatic_sub.min()))
    pm_max = float(max(primary_sub.max(), metastatic_sub.max()))
    pm_vmin, pm_vmax = pm_min, pm_max

    # Dedicated tight symmetric scale for delta so small changes (~0.01) are visible
    delta_abs_max = float(np.max(np.abs(delta_sub))) + 1e-12
    delta_vmin, delta_vmax = -delta_abs_max, delta_abs_max

    save_named_matrix_csv(
        primary_sub,
        row_ids=sel_ids,
        row_names=sel_names,
        col_ids=sel_ids,
        save_path=os.path.join(outdir, f"{prefix}_primary_top{len(idx)}.csv")
    )
    save_named_matrix_csv(
        metastatic_sub,
        row_ids=sel_ids,
        row_names=sel_names,
        col_ids=sel_ids,
        save_path=os.path.join(outdir, f"{prefix}_metastatic_top{len(idx)}.csv")
    )
    save_named_matrix_csv(
        delta_sub,
        row_ids=sel_ids,
        row_names=sel_names,
        col_ids=sel_ids,
        save_path=os.path.join(outdir, f"{prefix}_delta_met_minus_primary_top{len(idx)}.csv")
    )

    plot_crosstalk_heatmap(
        primary_sub,
        labels=short_labels,
        title=f"{prefix}: Primary Crosstalk (Top {len(idx)})",
        save_path=os.path.join(outdir, f"{prefix}_primary_top{len(idx)}.svg"),
        cmap="RdBu_r",
        center=None,
        vmin=pm_vmin,
        vmax=pm_vmax
    )
    plot_crosstalk_heatmap(
        metastatic_sub,
        labels=short_labels,
        title=f"{prefix}: Metastatic Crosstalk (Top {len(idx)})",
        save_path=os.path.join(outdir, f"{prefix}_metastatic_top{len(idx)}.svg"),
        cmap="RdBu_r",
        center=None,
        vmin=pm_vmin,
        vmax=pm_vmax
    )
    plot_crosstalk_heatmap(
        delta_sub,
        labels=short_labels,
        title=f"{prefix}: Delta Crosstalk (Metastatic - Primary, Top {len(idx)})",
        save_path=os.path.join(outdir, f"{prefix}_delta_met_minus_primary_top{len(idx)}.svg"),
        cmap="RdBu_r",
        center=0.0,
        vmin=delta_vmin,
        vmax=delta_vmax
    )
    print(f"Crosstalk outputs saved for {prefix} (Top {len(idx)} pathways).")

def create_full_crosstalk_outputs(primary_mat, metastatic_mat, pathway_ids, pathway_names, outdir, prefix):
    """Save full class-specific and differential crosstalk matrices and heatmaps."""
    if primary_mat is None or metastatic_mat is None:
        print(f"Warning: Missing class-specific matrix for {prefix}, skipping full crosstalk outputs.")
        return

    os.makedirs(outdir, exist_ok=True)
    idx = np.arange(len(pathway_ids), dtype=int)

    primary_full = primary_mat[np.ix_(idx, idx)]
    metastatic_full = metastatic_mat[np.ix_(idx, idx)]
    delta_full = metastatic_full - primary_full

    short_labels = [_short_clean_pathway_label(name, max_len=10) for name in pathway_names]

    pm_min = float(min(primary_full.min(), metastatic_full.min()))
    pm_max = float(max(primary_full.max(), metastatic_full.max()))

    delta_abs_max = float(np.max(np.abs(delta_full))) + 1e-12
    delta_vmin, delta_vmax = -delta_abs_max, delta_abs_max

    save_named_matrix_csv(
        primary_full,
        row_ids=pathway_ids,
        row_names=pathway_names,
        col_ids=pathway_ids,
        save_path=os.path.join(outdir, f"{prefix}_primary_full.csv")
    )
    save_named_matrix_csv(
        metastatic_full,
        row_ids=pathway_ids,
        row_names=pathway_names,
        col_ids=pathway_ids,
        save_path=os.path.join(outdir, f"{prefix}_metastatic_full.csv")
    )
    save_named_matrix_csv(
        delta_full,
        row_ids=pathway_ids,
        row_names=pathway_names,
        col_ids=pathway_ids,
        save_path=os.path.join(outdir, f"{prefix}_delta_met_minus_primary_full.csv")
    )

    plot_crosstalk_heatmap(
        primary_full,
        labels=short_labels,
        title=f"{prefix}: Primary Crosstalk (Full)",
        save_path=os.path.join(outdir, f"{prefix}_primary_full.svg"),
        cmap="RdBu_r",
        center=None,
        vmin=pm_min,
        vmax=pm_max
    )
    plot_crosstalk_heatmap(
        metastatic_full,
        labels=short_labels,
        title=f"{prefix}: Metastatic Crosstalk (Full)",
        save_path=os.path.join(outdir, f"{prefix}_metastatic_full.svg"),
        cmap="RdBu_r",
        center=None,
        vmin=pm_min,
        vmax=pm_max
    )
    plot_crosstalk_heatmap(
        delta_full,
        labels=short_labels,
        title=f"{prefix}: Delta Crosstalk (Metastatic - Primary, Full)",
        save_path=os.path.join(outdir, f"{prefix}_delta_met_minus_primary_full.svg"),
        cmap="RdBu_r",
        center=0.0,
        vmin=delta_vmin,
        vmax=delta_vmax
    )
    print(f"Crosstalk outputs saved for {prefix} (Full matrix).")

def extract_and_save_pathway_interactions(model, loader, pathway_ids, pathway_names,
                                          device, outdir, fold_num, normalize=False):
    attn_matrix = extract_pathway_attention_matrix(model, loader, device)
    if attn_matrix is None:
        print("Warning: No attention weights available to build pathway interaction matrix.")
        return None

    if normalize:
        attn_matrix, raw_min, raw_max = minmax_normalize_matrix(attn_matrix)
        print(f"Normalized pathway attention matrix to [0,1] (raw min={raw_min:.6g}, raw max={raw_max:.6g})")

    os.makedirs(outdir, exist_ok=True)
    attn_csv = os.path.join(outdir, f"pathway_attention_fold{fold_num}.csv")
    save_pathway_matrix_csv(attn_matrix, pathway_ids, pathway_names, attn_csv)
    print(f"Pathway attention matrix saved to: {attn_csv}")

    return attn_matrix

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss
        
# finding important pathways
def analyze_and_save_pathway_importance(model, test_loader, pathway_names, device, outdir, fold_num, top_k=10):
    """
    Analyze pathway importance and save top influential pathways
    
    Args:
        model: Trained model
        test_loader: Test data loader
        pathway_names: List of pathway names
        device: torch device
        outdir: Output directory
        fold_num: Current fold number
        top_k: Number of top pathways to display and save
    
    Returns:
        Dictionary with pathway rankings and scores
    """
    model.eval()
    all_pathway_weights = []
    
    print(f"\n{'='*70}")
    print(f"PATHWAY IMPORTANCE ANALYSIS - FOLD {fold_num}")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for mut_b, cnv_b, _ in test_loader:
            mut_b, cnv_b = mut_b.to(device), cnv_b.to(device)
            _, extras = model(mut_b, cnv_b, return_extras=True)
            all_pathway_weights.append(extras["pathway_weights"].cpu().numpy())
    
    # Calculate mean importance across all validation samples
    mean_pathway_weights = np.mean(np.vstack(all_pathway_weights), axis=0)
    
    # Rank pathways by importance (descending)
    pathway_ranking = np.argsort(-mean_pathway_weights)
    
    # Display top pathways
    print(f"\nTop {top_k} Most Influential Pathways:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Pathway ID':<15} {'Pathway Name':<40} {'Score':<12}")
    print("-" * 80)
    
    pathway_results = []
    for rank, idx in enumerate(pathway_ranking[:top_k], 1):
        pathway_name = pathway_names[idx] if idx < len(pathway_names) else f"Pathway_{idx}"
        score = mean_pathway_weights[idx]
        
        # Truncate long names for display
        display_name = pathway_name[:38] + "..." if len(pathway_name) > 40 else pathway_name
        
        print(f"{rank:<6} {idx:<15} {display_name:<40} {score:.8f}")
        
        pathway_results.append({
            'rank': rank,
            'pathway_idx': int(idx),
            'pathway_name': pathway_name,
            'importance_score': float(score)
        })
    
    # Save complete rankings to CSV
    all_rankings = []
    for rank, idx in enumerate(pathway_ranking, 1):
        pathway_name = pathway_names[idx] if idx < len(pathway_names) else f"Pathway_{idx}"
        all_rankings.append({
            'fold': fold_num,
            'rank': rank,
            'pathway_idx': int(idx),
            'pathway_name': pathway_name,
            'importance_score': float(mean_pathway_weights[idx])
        })
    
    # Save to CSV
    csv_file = os.path.join(outdir, f"pathway_importance_fold{fold_num}.csv")
    df_rankings = pd.DataFrame(all_rankings)
    df_rankings.to_csv(csv_file, index=False)
    print(f"\nComplete pathway rankings saved to: {csv_file}")
    
    # Save top pathways to JSON
    json_file = os.path.join(outdir, f"top_pathways_fold{fold_num}.json")
    with open(json_file, 'w') as f:
        json.dump({
            'fold': fold_num,
            'top_pathways': pathway_results,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"Top {top_k} pathways saved to: {json_file}")
    
    return {
        'fold': fold_num,
        'top_pathways': pathway_results,
        'all_rankings': all_rankings,
        'mean_scores': mean_pathway_weights
    }
# ===============================
# MAIN FUNCTION - Single Split (70/10/20)
# ===============================

def main_train_val_test():
    """Main function for a single stratified split: 70% train / 10% val / 20% test."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="outputs_split_70_10_20")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--min_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--use_focal_loss", action='store_true')
    ap.add_argument("--use_batch_norm", action='store_true', default=True)
    ap.add_argument("--pe_dim", type=int, default=16)
    ap.add_argument("--use_edge_aware_blocks", action='store_true', default=True)
    ap.add_argument("--use_full_graph", action='store_true', 
                    help="Use fully connected graph (all pathways attend to all) instead of sparse adjacency")
    ap.add_argument("--crosstalk_top_k", type=int, default=20,
                    help="Number of pathways for Primary vs Metastatic crosstalk comparison")
    ap.add_argument("--fdr_alpha", type=float, default=0.05,
                    help="FDR significance threshold for per-edge tests")
    ap.add_argument("--normalize_interactions", dest="normalize_interactions", action="store_true",
                    help="Normalize pathway interaction matrices to [0,1] for visualization (default: enabled)")
    ap.add_argument("--no_normalize_interactions", dest="normalize_interactions", action="store_false",
                    help="Disable min-max normalization for pathway interaction matrices")
    ap.set_defaults(normalize_interactions=True)
    
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print("SINGLE STRATIFIED TRAIN/VAL/TEST SPLIT")
    print(f"{'='*70}")
    print(f"Model: Improved Graph Transformer (Dwivedi & Bresson)")
    print("Split Strategy: 70% train, 10% validation, 20% test")
    print(f"{'='*70}\n")
    
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    # mut_df, cnv_df, labels, pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols = load_tables(args.data_dir)
    mut_df, cnv_df, labels, pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols = load_tables(
    args.data_dir, use_full_graph=args.use_full_graph
)
    print(f"Data: {len(mut_df)} patients | {len(gene_cols)} genes | {len(pathway_ids)} pathways")

    pathway_interaction_dir = os.path.join(args.outdir, "pathway_interactions")
    os.makedirs(pathway_interaction_dir, exist_ok=True)

    y_arr = binarize_labels(labels).values.astype(np.int64)
    
    unique, counts = np.unique(y_arr, return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} patients ({count/len(y_arr)*100:.1f}%)")

    split_path = os.path.join(args.outdir, f"split_70_10_20_rs{args.random_state}.json")
    split_data = load_train_val_test_split(split_path)
    if split_data is None:
        print("\nCreating new stratified 70/10/20 split...")
        split_data = create_train_val_test_split(
            y_arr,
            random_state=args.random_state,
            save_path=split_path
        )
    else:
        print("\nUsing existing saved split for reproducibility.")

    train_idx = np.array(split_data['train_idx'])
    val_idx = np.array(split_data['val_idx'])
    test_idx = np.array(split_data['test_idx'])

    set_seed(args.random_state)

    mut_tr, mut_va, mut_te, cnv_tr, cnv_va, cnv_te, y_tr, y_va, y_te, _, _, _ = prepare_data_from_splits(
        mut_df, cnv_df, labels, train_idx, val_idx, test_idx
    )

    train_ds = GenePatientDataset(mut_tr, cnv_tr, y_tr)
    val_ds = GenePatientDataset(mut_va, cnv_va, y_va)
    test_ds = GenePatientDataset(mut_te, cnv_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_M = max(len(lst) for lst in pathway_gene_lists)

    model = PathwayGraphTransformer(
        num_genes=len(gene_cols),
        num_pathways=len(pathway_ids),
        max_pathway_genes=max_M,
        d=args.d_model,
        layers=args.layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_film=True,
        use_edge_mask=not args.use_full_graph,
        use_edge_bias=True,
        pe_dim=args.pe_dim,
        use_batch_norm=args.use_batch_norm,
        use_edge_aware_blocks=args.use_edge_aware_blocks,
        full_graph_attention=args.use_full_graph
    ).to(device)
    model.set_structures(pathway_gene_lists, torch.from_numpy(A))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    classes = np.unique(y_tr.numpy())
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr.numpy())
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=weights, gamma=2.0) if args.use_focal_loss else nn.CrossEntropyLoss(weight=weights)

    best_val_metric = -float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    print("\nTraining single split model...")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        val_metrics = evaluate_metrics(model, val_loader, device)
        val_metric = val_metrics["auc"]

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d} | Loss: {tr_loss:.4f} | Val AUC: {val_metric:.4f} | "
                  f"Patience: {epochs_no_improve}/{args.patience}")

        if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_metrics = evaluate_metrics(model, val_loader, device)
    test_metrics = evaluate_metrics(model, test_loader, device)

    pathway_importance_dir = os.path.join(args.outdir, "pathway_importance")
    os.makedirs(pathway_importance_dir, exist_ok=True)
    pathway_analysis = analyze_and_save_pathway_importance(
        model=model,
        test_loader=test_loader,
        pathway_names=pathway_names,
        device=device,
        outdir=pathway_importance_dir,
        fold_num=1,
        top_k=10
    )

    fold_attn = extract_and_save_pathway_interactions(
        model=model,
        loader=test_loader,
        pathway_ids=pathway_ids,
        pathway_names=pathway_names,
        device=device,
        outdir=pathway_interaction_dir,
        fold_num=1,
        normalize=args.normalize_interactions
    )

    class_mats, class_counts = extract_pathway_attention_by_class(model, test_loader, device)
    split_primary = class_mats.get(0)
    split_metastatic = class_mats.get(1)
    split_primary_vis = split_primary
    split_metastatic_vis = split_metastatic
    if args.normalize_interactions and split_primary is not None and split_metastatic is not None:
        split_primary_vis, split_metastatic_vis, shared_min, shared_max = normalize_primary_metastatic_pair(
            split_primary, split_metastatic
        )
        print(f"Normalized class-specific matrices to [0,1] with shared scale "
              f"(raw min={shared_min:.6g}, raw max={shared_max:.6g})")

    split_attn_samples, split_labels = extract_pathway_attention_samples(model, test_loader, device)
    split_edge_stats = None
    if split_attn_samples is not None and split_labels is not None:
        primary_samples = split_attn_samples[split_labels == 0]
        metastatic_samples = split_attn_samples[split_labels == 1]
        if len(primary_samples) > 1 and len(metastatic_samples) > 1:
            split_edge_stats = compute_and_save_edge_fdr(
                primary_samples=primary_samples,
                metastatic_samples=metastatic_samples,
                pathway_ids=pathway_ids,
                pathway_names=pathway_names,
                outdir=pathway_interaction_dir,
                prefix="split",
                alpha=args.fdr_alpha
            )
        else:
            print("Warning: Not enough samples per class for per-edge FDR testing (need >=2 per class).")

    if split_primary_vis is not None and split_metastatic_vis is not None:
        create_full_crosstalk_outputs(
            primary_mat=split_primary_vis,
            metastatic_mat=split_metastatic_vis,
            pathway_ids=pathway_ids,
            pathway_names=pathway_names,
            outdir=pathway_interaction_dir,
            prefix="split"
        )
        top_k_split = min(args.crosstalk_top_k, len(pathway_ids))
        split_change_rank, split_change_scores = rank_pathways_by_interaction_change(
            split_primary_vis,
            split_metastatic_vis
        )
        split_top_idx = split_change_rank[:top_k_split]
        split_change_df = pd.DataFrame({
            'rank': np.arange(1, len(pathway_ids) + 1),
            'pathway_idx': split_change_rank.astype(int),
            'pathway_id': [pathway_ids[i] for i in split_change_rank],
            'pathway_name': [pathway_names[i] for i in split_change_rank],
            'interaction_change_score': [float(split_change_scores[i]) for i in split_change_rank]
        })
        split_change_csv = os.path.join(pathway_interaction_dir, "split_pathway_change_ranking.csv")
        split_change_df.to_csv(split_change_csv, index=False)
        print(f"Pathways ranked by Primary-Metastatic interaction change saved to: {split_change_csv}")
        create_topk_crosstalk_outputs(
            primary_mat=split_primary_vis,
            metastatic_mat=split_metastatic_vis,
            selected_indices=split_top_idx,
            pathway_ids=pathway_ids,
            pathway_names=pathway_names,
            outdir=pathway_interaction_dir,
            prefix="split"
        )
        print(f"Split class counts for crosstalk: Primary={class_counts[0]}, Metastatic={class_counts[1]}")
    else:
        print("Warning: Missing class-specific attention matrix; Primary vs Metastatic crosstalk not saved.")

    all_test_y_true = [test_metrics['y_true']]
    all_test_y_pred = [test_metrics['y_pred']]
    all_test_y_probs = [test_metrics['y_probs']]

    print(f"\n{'='*70}")
    print("SINGLE SPLIT TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation: AUC={final_val_metrics['auc']:.4f}, F1={final_val_metrics['f1_binary']:.4f}, "
          f"Acc={final_val_metrics['acc']:.4f}")
    print(f"Test:       AUC={test_metrics['auc']:.4f}, F1={test_metrics['f1_binary']:.4f}, "
          f"Acc={test_metrics['acc']:.4f}")

    print(f"\nGenerating visualizations...")
    viz_dir = os.path.join(args.outdir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_suffix = "fullgraph" if args.use_full_graph else "sparse"
    
    plot_aggregate_confusion_matrix(
        all_test_y_true, all_test_y_pred, 
        os.path.join(viz_dir, f"split_confusion_matrix_{graph_suffix}_{timestamp}.png"),
        1, title=f"Confusion Matrix - Test Set ({graph_suffix.upper()})"
    )
    
    plot_aggregate_roc_curve(
        all_test_y_true, all_test_y_probs,
        os.path.join(viz_dir, f"split_roc_curve_{graph_suffix}_{timestamp}.png"),
        1, title=f"ROC Curve - Test Set ({graph_suffix.upper()})"
    )
    
    plot_aggregate_pr_curve(
        all_test_y_true, all_test_y_probs,
        os.path.join(viz_dir, f"split_pr_curve_{graph_suffix}_{timestamp}.png"),
        1, title=f"Precision-Recall Curve - Test Set ({graph_suffix.upper()})"
    )

    # Aggregate pathway importance over available run(s) - here n_folds=1
    try:
        print(f"\nAggregating pathway importance...")
        aggregate_pathway_summary = aggregate_pathway_rankings_across_folds(
            outdir=args.outdir,
            n_folds=1
        )
    except Exception as e:
        print(f"Warning: Could not aggregate pathway importance: {str(e)}")
        aggregate_pathway_summary = None

    if fold_attn is not None:
        aggregate_attention = fold_attn
        aggregate_csv = os.path.join(pathway_interaction_dir, "pathway_attention_aggregate.csv")
        save_pathway_matrix_csv(aggregate_attention, pathway_ids, pathway_names, aggregate_csv)
        print(f"Aggregate pathway attention matrix saved to: {aggregate_csv}")
    else:
        print("Warning: No attention matrix collected; aggregate pathway interaction not saved.")

    if split_primary_vis is not None and split_metastatic_vis is not None:
        agg_primary = split_primary_vis
        agg_metastatic = split_metastatic_vis

        save_pathway_matrix_csv(
            agg_primary,
            pathway_ids,
            pathway_names,
            os.path.join(pathway_interaction_dir, "pathway_attention_aggregate_primary.csv")
        )
        save_pathway_matrix_csv(
            agg_metastatic,
            pathway_ids,
            pathway_names,
            os.path.join(pathway_interaction_dir, "pathway_attention_aggregate_metastatic.csv")
        )

        agg_change_rank, agg_change_scores = rank_pathways_by_interaction_change(
            agg_primary,
            agg_metastatic
        )
        top_k_agg = min(args.crosstalk_top_k, len(pathway_ids))
        top_indices = agg_change_rank[:top_k_agg]
        agg_change_df = pd.DataFrame({
            'rank': np.arange(1, len(pathway_ids) + 1),
            'pathway_idx': agg_change_rank.astype(int),
            'pathway_id': [pathway_ids[i] for i in agg_change_rank],
            'pathway_name': [pathway_names[i] for i in agg_change_rank],
            'interaction_change_score': [float(agg_change_scores[i]) for i in agg_change_rank]
        })
        agg_change_csv = os.path.join(pathway_interaction_dir, "aggregate_pathway_change_ranking.csv")
        agg_change_df.to_csv(agg_change_csv, index=False)
        print(f"Aggregate pathway change ranking saved to: {agg_change_csv}")

        create_full_crosstalk_outputs(
            primary_mat=agg_primary,
            metastatic_mat=agg_metastatic,
            pathway_ids=pathway_ids,
            pathway_names=pathway_names,
            outdir=pathway_interaction_dir,
            prefix="aggregate"
        )
        create_topk_crosstalk_outputs(
            primary_mat=agg_primary,
            metastatic_mat=agg_metastatic,
            selected_indices=top_indices,
            pathway_ids=pathway_ids,
            pathway_names=pathway_names,
            outdir=pathway_interaction_dir,
            prefix="aggregate"
        )
    else:
        print("Warning: No class-specific attention collected; aggregate Primary vs Metastatic crosstalk not saved.")

    results_file = os.path.join(args.outdir, f"split_results_{graph_suffix}_{timestamp}.json")
    results_data = {
        'timestamp': timestamp,
        'model_type': 'SingleSplit_Graph_Transformer',
        'graph_mode': 'full_graph' if args.use_full_graph else 'sparse_graph',
        'split_strategy': 'Stratified 70/10/20',
        'best_epoch': best_epoch,
        'validation_metrics': {k: final_val_metrics[k] for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'test_metrics': {k: test_metrics[k] for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'config': vars(args),
        'edge_fdr_stats': split_edge_stats
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results_data

if __name__ == "__main__":
    results = main_train_val_test()
    print(f"\nSingle-split experiment complete.")
    print(f"   Test AUC: {results['test_metrics']['auc']:.4f}")
    print(f"   Test F1:  {results['test_metrics']['f1_binary']:.4f}")
