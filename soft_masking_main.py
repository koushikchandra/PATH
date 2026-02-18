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
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import copy




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
# MODIFIED: 5-Fold CV Split Creation
# ===============================

def create_5fold_splits(y_arr, val_size=0.1, random_state=42, save_path="5fold_splits.json"):
    """
    Create 5-fold stratified cross-validation splits with validation set
    
    Args:
        y_arr: Labels array
        val_size: Proportion of training data for validation (0.1 = 10%)
        random_state: Random seed for reproducibility
        save_path: Path to save splits
    
    Returns:
        List of fold dictionaries with train/val/test indices
    """
    all_folds = []
    
    # Create 5-fold stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print(f"Creating 5-fold stratified CV splits...")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(y_arr)), y_arr), 1):
        print(f"\nFold {fold_idx}/5:")
        
        # Further split train_val into train and val (stratified)
        y_train_val = y_arr[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y_train_val,
            random_state=random_state
        )
        
        # Calculate actual proportions
        total_samples = len(y_arr)
        train_prop = len(train_idx) / total_samples
        val_prop = len(val_idx) / total_samples
        test_prop = len(test_idx) / total_samples
        
        # Calculate class distributions
        train_classes = np.bincount(y_arr[train_idx])
        val_classes = np.bincount(y_arr[val_idx])
        test_classes = np.bincount(y_arr[test_idx])
        
        fold_info = {
            'fold': fold_idx,
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
        
        all_folds.append(fold_info)
        
        print(f"  Train: {len(train_idx)} ({train_prop:.1%}) - Class dist: {train_classes}")
        print(f"  Val:   {len(val_idx)} ({val_prop:.1%}) - Class dist: {val_classes}")
        print(f"  Test:  {len(test_idx)} ({test_prop:.1%}) - Class dist: {test_classes}")
    
    # Save splits
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_folds, f, indent=2)
    
    print(f"\n5-fold CV splits saved to {save_path}")
    return all_folds

def load_5fold_splits(load_path="5fold_splits.json"):
    """Load previously saved 5-fold CV split indices"""
    if not os.path.exists(load_path):
        return None
    
    with open(load_path, 'r') as f:
        folds = json.load(f)
    
    print(f"5-fold CV splits loaded from {load_path}")
    return folds

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
                # mask = mask.masked_fill(nonedge, float('-inf'))  # HARD mask
                mask.fill_diagonal_(0.0)

        self.attn_mask = mask.detach()

    def forward(self, mut: torch.Tensor, cnv: torch.Tensor, return_extras: bool = False, 
                gene_ids: Optional[torch.Tensor] = None):
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

        for blk in self.layers:
            X, _, edge_feat = blk(X, attn_mask=self.attn_mask, edge_feat=edge_feat)

        pw_scores = self.pathway_attention(X)
        pw_weights = F.softmax(pw_scores, dim=1)
        global_repr = torch.sum(X * pw_weights, dim=1)
        logits = self.head(global_repr)

        if return_extras:
            return logits, {'gene_alpha': gene_alpha, 'pathway_weights': pw_weights.squeeze(-1)}
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
        val_loader: Validation data loader
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
# MAIN FUNCTION - 5-FOLD CV
# ===============================

def main_5fold_cv():
    """Main function for 5-fold stratified cross-validation"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="outputs_5fold_cv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--min_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--use_focal_loss", action='store_true')
    ap.add_argument("--use_batch_norm", action='store_true', default=True)
    ap.add_argument("--pe_dim", type=int, default=16)
    ap.add_argument("--use_edge_aware_blocks", action='store_true', default=True)
    ap.add_argument("--use_full_graph", action='store_true', 
                    help="Use fully connected graph (all pathways attend to all) instead of sparse adjacency")
    
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Model: Improved Graph Transformer (Dwivedi & Bresson)")
    print(f"CV Strategy: 5-Fold Stratified (1 fold test, 4 folds train, 10% of train as val)")
    print(f"Expected splits per fold: ~72% train, ~8% val, ~20% test")
    print(f"{'='*70}\n")
    
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    # mut_df, cnv_df, labels, pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols = load_tables(args.data_dir)
    mut_df, cnv_df, labels, pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols = load_tables(
    args.data_dir, use_full_graph=args.use_full_graph
)
    print(f"Data: {len(mut_df)} patients | {len(gene_cols)} genes | {len(pathway_ids)} pathways")

    y_arr = binarize_labels(labels).values.astype(np.int64)
    patient_ids = np.array(mut_df.index)
    
    unique, counts = np.unique(y_arr, return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} patients ({count/len(y_arr)*100:.1f}%)")

    # Create or load 5-fold splits
    folds_path = os.path.join(args.outdir, f"5fold_splits_rs{args.random_state}.json")
    folds = load_5fold_splits(folds_path)
    
    if folds is None:
        print(f"\nCreating new 5-fold stratified CV splits...")
        folds = create_5fold_splits(
            y_arr, 
            val_size=args.val_size,
            random_state=args.random_state,
            save_path=folds_path
        )
    else:
        print(f"\nUsing existing 5-fold CV splits for reproducibility")

    all_fold_val_metrics = []
    all_fold_test_metrics = []
    all_fold_details = []
    all_fold_predictions = []
    
    # Storage for aggregate visualizations
    all_test_y_true = []
    all_test_y_pred = []
    all_test_y_probs = []
    
    print(f"\nStarting 5-fold CV evaluation...")
    
    # MAIN FOLD LOOP
    for fold_data in folds:
        fold = fold_data['fold']
        train_idx = np.array(fold_data['train_idx'])
        val_idx = np.array(fold_data['val_idx'])
        test_idx = np.array(fold_data['test_idx'])
        
        print(f"\n{'='*50} FOLD {fold}/5 {'='*50}")
        
        # Set seed
        set_seed(args.random_state + fold)
        
        # Prepare data
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
        
        # Create model
        model = PathwayGraphTransformer(
            num_genes=len(gene_cols), 
            num_pathways=len(pathway_ids), 
            max_pathway_genes=max_M,
            d=args.d_model,
            layers=args.layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_film=True,
            use_edge_mask=not args.use_full_graph,  # No masking in full graph mode
            use_edge_bias=True,
            pe_dim=args.pe_dim,
            use_batch_norm=args.use_batch_norm,
            use_edge_aware_blocks=args.use_edge_aware_blocks,
            full_graph_attention=args.use_full_graph  # Enable full graph attention
        ).to(device)
        
        model.set_structures(pathway_gene_lists, torch.from_numpy(A))

        # Optimizer and loss
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        classes = np.unique(y_tr.numpy())
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr.numpy())
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        if args.use_focal_loss:
            criterion = FocalLoss(alpha=weights, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)

        best_val_metric = -float("inf")
        epochs_no_improve = 0
        best_epoch = 0
        best_model_state = None
        
        # Training loop
        print(f"\nTraining fold {fold}...")
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
            val_metrics = evaluate_metrics(model, val_loader, device)
            val_metric = val_metrics["auc"]
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                epochs_no_improve = 0
                best_epoch = epoch
                # best_model_state = model.state_dict().copy()
                best_model_state = copy.deepcopy(model.state_dict())  # ✓ CORRECT

            else:
                epochs_no_improve += 1
            
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:3d} | Loss: {tr_loss:.4f} | Val AUC: {val_metric:.4f} | "
                      f"Patience: {epochs_no_improve}/{args.patience}")
            
            if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate
        if best_model_state is not None:
            best_model_path = os.path.join(args.outdir, f"best_model_fold{fold}.pt")
            torch.save(best_model_state, best_model_path)
            model.load_state_dict(best_model_state)
        
        final_val_metrics = evaluate_metrics(model, val_loader, device)
        test_metrics = evaluate_metrics(model, test_loader, device)
        
        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        final_val_metrics = evaluate_metrics(model, val_loader, device)
        test_metrics = evaluate_metrics(model, test_loader, device)
        
        # ===== ADD THIS SECTION =====
        # Analyze pathway importance for this fold
        pathway_importance_dir = os.path.join(args.outdir, "pathway_importance")
        os.makedirs(pathway_importance_dir, exist_ok=True)
        
        pathway_analysis = analyze_and_save_pathway_importance(
            model=model,
            test_loader=test_loader,
            pathway_names=pathway_names,
            device=device,
            outdir=pathway_importance_dir,
            fold_num=fold,
            top_k=10  # Change this to get more/fewer top pathways
        )
        # ===== END OF ADDITION =====
        
        # Store test predictions
        all_test_y_true.append(test_metrics['y_true'])
        all_test_y_pred.append(test_metrics['y_pred'])
        all_test_y_probs.append(test_metrics['y_probs'])

        # Collect per-sample test predictions for CSV export
        test_sample_ids = patient_ids[test_idx]
        fold_predictions_df = pd.DataFrame({
            "Sample": test_sample_ids,
            "y_true_value": test_metrics['y_true'],
            "y_prediction_score": test_metrics['y_probs']
        })
        all_fold_predictions.append(fold_predictions_df)
        
        print(f"\nFold {fold} Complete!")
        print(f"   Val:  AUC={final_val_metrics['auc']:.4f}, F1={final_val_metrics['f1_binary']:.4f}")
        print(f"   Test: AUC={test_metrics['auc']:.4f}, F1={test_metrics['f1_binary']:.4f}, Acc={test_metrics['acc']:.4f}")
        
        # Clean metrics before storing
        test_metrics_clean = {k: v for k, v in test_metrics.items() 
                             if k not in ['y_true', 'y_pred', 'y_probs']}
        val_metrics_clean = {k: v for k, v in final_val_metrics.items() 
                            if k not in ['y_true', 'y_pred', 'y_probs']}
        
        all_fold_val_metrics.append(val_metrics_clean)
        all_fold_test_metrics.append(test_metrics_clean)
        all_fold_details.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'val_metrics': val_metrics_clean,
            'test_metrics': test_metrics_clean
        })

# FINAL RESULTS
    print(f"\n{'='*70}")
    print("5-FOLD CROSS-VALIDATION COMPLETE!")
    print(f"{'='*70}")
    
    print("\nVALIDATION PERFORMANCE (averaged across 5 folds):")
    for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']:
        vals = [m[k] for m in all_fold_val_metrics if k in m]
        print(f"  {k.upper():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print("\nTEST PERFORMANCE (averaged across 5 folds):")
    for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']:
        vals = [m[k] for m in all_fold_test_metrics if k in m]
        print(f"  {k.upper():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nPER-FOLD RESULTS:")
    for detail in all_fold_details:
        print(f"Fold {detail['fold']}: Val AUC={detail['val_metrics']['auc']:.4f}, "
              f"Test AUC={detail['test_metrics']['auc']:.4f}, "
              f"Test F1={detail['test_metrics']['f1_binary']:.4f}")

    # Generate aggregate visualizations
    print(f"\nGenerating aggregate visualizations...")
    viz_dir = os.path.join(args.outdir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_suffix = "fullgraph" if args.use_full_graph else "sparse"

    if all_fold_predictions:
        predictions_df = pd.concat(all_fold_predictions, ignore_index=True)
        predictions_csv = os.path.join(args.outdir, f"5fold_test_predictions_{graph_suffix}_{timestamp}.csv")
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"Per-sample test predictions saved to: {predictions_csv}")
    
    plot_aggregate_confusion_matrix(
        all_test_y_true, all_test_y_pred, 
        os.path.join(viz_dir, f"5fold_confusion_matrix_{graph_suffix}_{timestamp}.png"),
        5, title=f"5-Fold CV Confusion Matrix - Test Set ({graph_suffix.upper()})"
    )
    
    plot_aggregate_roc_curve(
        all_test_y_true, all_test_y_probs,
        os.path.join(viz_dir, f"5fold_roc_curve_{graph_suffix}_{timestamp}.png"),
        5, title=f"5-Fold CV ROC Curve - Test Set ({graph_suffix.upper()})"
    )
    
    plot_aggregate_pr_curve(
        all_test_y_true, all_test_y_probs,
        os.path.join(viz_dir, f"5fold_pr_curve_{graph_suffix}_{timestamp}.png"),
        5, title=f"5-Fold CV Precision-Recall Curve - Test Set ({graph_suffix.upper()})"
    )

    # ===== ADD THIS SECTION HERE =====
    # Aggregate pathway importance across all folds
    try:
        print(f"\nAggregating pathway importance across folds...")
        aggregate_pathway_summary = aggregate_pathway_rankings_across_folds(
            outdir=args.outdir,
            n_folds=5
        )
    except Exception as e:
        print(f"Warning: Could not aggregate pathway importance: {str(e)}")
        aggregate_pathway_summary = None
    # ===== END OF ADDITION =====

    # Save results
    results_file = os.path.join(args.outdir, f"5fold_results_{graph_suffix}_{timestamp}.json")
    results_data = {
        'timestamp': timestamp,
        'model_type': '5-Fold_CV_Graph_Transformer',
        'graph_mode': 'full_graph' if args.use_full_graph else 'sparse_graph',
        'cv_strategy': '5-Fold Stratified (20% test per fold, 10% val from remaining)',
        'validation_summary': {k: {'mean': np.mean([m[k] for m in all_fold_val_metrics]),
                                   'std': np.std([m[k] for m in all_fold_val_metrics])}
                              for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'test_summary': {k: {'mean': np.mean([m[k] for m in all_fold_test_metrics]),
                            'std': np.std([m[k] for m in all_fold_test_metrics])}
                        for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'fold_details': all_fold_details,
        'config': vars(args)
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results_data

if __name__ == "__main__":
    results = main_5fold_cv()
    print(f"\n5-Fold CV Experiment Complete!")
    print(f"   Mean Test AUC: {results['test_summary']['auc']['mean']:.4f} ± {results['test_summary']['auc']['std']:.4f}")
    print(f"   Mean Test F1:  {results['test_summary']['f1_binary']['mean']:.4f} ± {results['test_summary']['f1_binary']['std']:.4f}")
