#!/usr/bin/env python3
"""
Pathway Interaction Analysis with Edge-Aware Graph Transformer
Version 2.0 - Includes edge ablation experiments for significance testing

Changes from v1:
- Added edge ablation experiment to test significance of learned edge values
- Fixed duplicate evaluation code bug
- Cleaned up commented code
- Added visualization for ablation results
"""

import os
import re
import json
import math
import argparse
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import scipy.stats as stats


# ===============================
# Utility Functions
# ===============================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_train_only(train_vals: np.ndarray, other_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score normalize using training set statistics"""
    mu = train_vals.mean(axis=0, keepdims=True)
    sd = train_vals.std(axis=0, keepdims=True) + 1e-8
    return (train_vals - mu) / sd, (other_vals - mu) / sd


def binarize_labels(labels: pd.Series) -> pd.Series:
    """Convert labels to binary (0=Primary, 1=Metastatic)"""
    m = labels.astype(str).str.lower().str.strip()
    mapping = {"primary": 0, "metastatic": 1, "p": 0, "m": 1, "0": 0, "1": 1}
    y = m.map(mapping)
    if y.isna().any():
        y = pd.to_numeric(labels, errors="coerce")
    if y.isna().any():
        bad = labels[y.isna()]
        raise ValueError(f"Unrecognized labels (use Primary/Metastatic or 0/1). Offenders: {bad.unique()[:5]}")
    return y.astype(int)


# ===============================
# Data Loading
# ===============================

def load_tables(data_dir: str, use_full_graph: bool = False):
    """
    Load all required data tables

    Args:
        data_dir: Directory containing data files
        use_full_graph: If True, create fully connected adjacency matrix

    Returns:
        Tuple of (mut_df, cnv_df, labels, pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols)
    """
    mut_path = os.path.join(data_dir, "mutation_data.csv")
    cnv_path = os.path.join(data_dir, "cnv_data.csv")
    lab_path = os.path.join(data_dir, "patient_labels.csv")
    pw_path = os.path.join(data_dir, "filtered_pathways.csv")
    adj_path = os.path.join(data_dir, "adjacency_matrix.csv")

    # Load mutation and CNV data
    mut_df = pd.read_csv(mut_path)
    cnv_df = pd.read_csv(cnv_path)
    if mut_df.shape[1] < 2 or cnv_df.shape[1] < 2:
        raise ValueError("mutation_data.csv / cnv_data.csv must have patient id in col 1 and genes afterward.")
    mut_df = mut_df.set_index(mut_df.columns[0])
    cnv_df = cnv_df.set_index(cnv_df.columns[0])

    # Find common genes
    common_genes = [g for g in mut_df.columns if g in cnv_df.columns]
    if len(common_genes) == 0:
        raise ValueError("No overlapping gene columns between mutation_data.csv and cnv_data.csv.")
    mut_df = mut_df[common_genes]
    cnv_df = cnv_df[common_genes]

    # Load labels
    lab_df = pd.read_csv(lab_path)
    if lab_df.shape[1] < 2:
        raise ValueError("patient_labels.csv must have two columns: patient_id,label")
    lab_df = lab_df.set_index(lab_df.columns[0])
    lab_df = lab_df.iloc[:, :1]

    # Align patient IDs
    mut_df.index = mut_df.index.astype(str)
    cnv_df.index = cnv_df.index.astype(str)
    lab_df.index = lab_df.index.astype(str)
    common_ids = mut_df.index.intersection(cnv_df.index).intersection(lab_df.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping patient IDs across mutation/cnv/labels.")
    mut_df = mut_df.loc[common_ids].sort_index()
    cnv_df = cnv_df.loc[common_ids].sort_index()
    lab_df = lab_df.loc[common_ids].sort_index()

    # Load pathway definitions
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

    # Create adjacency matrix based on mode
    if use_full_graph:
        num_pathways = len(pathway_ids)
        A = np.ones((num_pathways, num_pathways), dtype=np.float32)
        np.fill_diagonal(A, 0.0)
        print(f"Using FULL GRAPH mode: {num_pathways}x{num_pathways} fully connected adjacency matrix")
    else:
        adj_df = pd.read_csv(adj_path)
        adj_df = adj_df.set_index(adj_df.columns[0])
        adj_df.index = adj_df.index.astype(str)
        adj_df.columns = adj_df.columns.astype(str)
        adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        adj_df = adj_df.reindex(index=pathway_ids, columns=pathway_ids).fillna(0)
        A = adj_df.values.astype(np.float32)
        A = 0.5 * (A + A.T)  # Symmetrize
        A = A / (A.max() + 1e-8)  # Normalize

        num_edges = (A > 0).sum() - len(pathway_ids)
        max_edges = len(pathway_ids) * (len(pathway_ids) - 1)
        sparsity = 100 * (1 - num_edges / max_edges)
        print(f"Using SPARSE GRAPH mode: {num_edges} edges, {sparsity:.1f}% sparsity")

    return mut_df, cnv_df, lab_df.iloc[:, 0], pathway_gene_lists, pathway_ids, pathway_names, A, gene_cols


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
# Cross-Validation Split Functions
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print(f"Creating 5-fold stratified CV splits...")

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(y_arr)), y_arr), 1):
        print(f"\nFold {fold_idx}/5:")

        y_train_val = y_arr[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
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

        fold_info = {
            'fold': fold_idx,
            'random_state': random_state,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'proportions': {'train': train_prop, 'val': val_prop, 'test': test_prop},
            'sample_counts': {
                'train': len(train_idx), 'val': len(val_idx),
                'test': len(test_idx), 'total': total_samples
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
# Model Components
# ===============================

class GeneEncoderB(nn.Module):
    """Gene encoder with FiLM-style conditioning from mutation/CNV values"""

    def __init__(self, num_genes, d=128, hidden=64, positive_gamma=True):
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, d)
        self.to_gammabeta = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d)
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
    """Attention-based pooling of gene features into pathway representations"""

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
    """Transformer block with edge-aware attention mechanism"""

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
    """
    Main Graph Transformer model for pathway-based classification

    Features:
    - Gene-level encoding with FiLM conditioning
    - Pathway-level attention pooling
    - Edge-aware graph transformer layers
    - Laplacian positional encoding
    """

    def __init__(self, num_genes, num_pathways, max_pathway_genes,
                 d=64, layers=2, num_heads=4, dropout=0.2,
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
            # Fallback to standard transformer blocks
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, dropout=dropout, batch_first=True)
                for _ in range(layers)
            ])

        self.pathway_attention = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d, 2)
        )

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
            if len(lst) == 0:
                continue
            idx[p, :len(lst)] = torch.tensor(lst, dtype=torch.long).clamp(0, num_genes - 1)
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
        """Initialize pathway structures and adjacency matrix"""
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

        if self.full_graph_attention:
            mask = torch.zeros_like(A)
        else:
            mask = torch.zeros_like(A)
            if self.use_edge_mask:
                nonedge = (A <= 0)
                mask = mask.masked_fill(nonedge, -10.0)
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
            if self.use_edge_aware_blocks:
                X, attn_w, edge_feat = blk(X, attn_mask=self.attn_mask, edge_feat=edge_feat)
            else:
                X = blk(X)
                attn_w = None
            if return_attn and attn_w is not None:
                attn_weights.append(attn_w.detach())
            if return_edge:
                edge_features.append(edge_feat.detach() if edge_feat is not None else None)

        pw_scores = self.pathway_attention(X)
        pw_weights = F.softmax(pw_scores, dim=1)
        global_repr = torch.sum(X * pw_weights, dim=1)
        logits = self.head(global_repr)

        if return_extras or return_attn or return_edge:
            extras = {'gene_alpha': gene_alpha, 'pathway_weights': pw_weights.squeeze(-1),
                      'global_repr': global_repr}
            if return_attn:
                extras['attn_weights'] = attn_weights
            if return_edge:
                extras['edge_features'] = edge_features
            return logits, extras
        return logits


# ===============================
# Dataset and Loss Functions
# ===============================

class GenePatientDataset(Dataset):
    """Dataset for patient mutation/CNV data"""

    def __init__(self, mut: torch.Tensor, cnv: torch.Tensor, y: torch.Tensor):
        self.mut, self.cnv, self.y = mut, cnv, y

    def __len__(self):
        return self.mut.size(0)

    def __getitem__(self, i):
        return self.mut[i], self.cnv[i], self.y[i]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

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


# ===============================
# Training and Evaluation
# ===============================

def train_one_epoch(model, loader, opt, criterion, device):
    """Train model for one epoch"""
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
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_y, all_p1 = [], []
    pbar = tqdm(loader, desc="Validating", leave=False, ncols=80)
    for mut_b, cnv_b, y_b in pbar:
        mut_b, cnv_b, y_b = mut_b.to(device), cnv_b.to(device), y_b.to(device)
        logits = model(mut_b, cnv_b)
        probs = logits.softmax(dim=-1)[:, 1]
        all_y.append(y_b.cpu().numpy())
        all_p1.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    p1 = np.concatenate(all_p1) if all_p1 else np.array([])

    prec, rec, thr = precision_recall_curve(y_true, p1)
    if thr.size:
        f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_thr = float(thr[int(np.nanargmax(f1s))])
    else:
        best_thr = 0.5
    pred = (p1 >= best_thr).astype(int)

    acc = accuracy_score(y_true, pred)
    auc_score = roc_auc_score(y_true, p1) if y_true.size > 0 and len(np.unique(y_true)) > 1 else float("nan")
    aupr = average_precision_score(y_true, p1) if y_true.size > 0 and len(np.unique(y_true)) > 1 else float("nan")

    f1_binary = f1_score(y_true, pred, average='binary', zero_division=0)
    precision_val = precision_score(y_true, pred, average='binary', zero_division=0)
    recall_val = recall_score(y_true, pred, average='binary', zero_division=0)

    return {
        "acc": acc,
        "auc": auc_score,
        "aupr": aupr,
        "f1_binary": f1_binary,
        "precision": precision_val,
        "recall": recall_val,
        "y_true": y_true,
        "y_pred": pred,
        "y_probs": p1
    }


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract global patient-level embeddings (pre-classifier head) from a data loader.

    Returns pooled test-set representations, one vector per patient, suitable
    for downstream UMAP / t-SNE visualisation and silhouette scoring.
    """
    model.eval()
    all_embs, all_labels = [], []
    for mut_b, cnv_b, y_b in loader:
        mut_b, cnv_b = mut_b.to(device), cnv_b.to(device)
        _, extras = model(mut_b, cnv_b, return_extras=True)
        all_embs.append(extras['global_repr'].cpu().numpy())
        all_labels.append(y_b.numpy())
    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


def plot_umap_tsne_embeddings(embeddings, labels, save_dir, prefix="test_embeddings",
                               n_neighbors=15, min_dist=0.1, perplexity=30,
                               random_state=42, dpi=300):
    """Generate UMAP and t-SNE plots for pooled test embeddings.

    Only test-set embeddings are visualised: these are honest, held-out
    representations that reflect genuine generalisation rather than
    training memorisation.  Comparing these across baselines is a fair,
    apples-to-apples evaluation.

    Also computes a silhouette score on the raw (pre-projection) embeddings
    as a quantitative complement to the visual inspection.

    Returns:
        float: silhouette score on the raw embeddings.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    os.makedirs(save_dir, exist_ok=True)

    X = StandardScaler().fit_transform(embeddings)

    palette = {0: "#1f77b4", 1: "#d62728"}   # Blue = Primary, Red = Metastatic
    label_names = {0: "Primary", 1: "Metastatic"}

    # Silhouette score on raw embeddings (before dimensionality reduction)
    sil_score = silhouette_score(embeddings, labels)
    print(f"  Silhouette score (raw test embeddings): {sil_score:.4f}")

    def _scatter(ax, coords, title, xlabel, ylabel):
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=50, alpha=0.7,
                c=palette.get(int(cls), "#555555"),
                label=label_names.get(int(cls), f"Class {int(cls)}"),
                edgecolors='white', linewidths=0.5
            )
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------
    umap_coords = None
    try:
        import umap.umap_ as umap_module
        n_nb = max(2, min(n_neighbors, X.shape[0] - 1))
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=n_nb, min_dist=min_dist,
            metric="euclidean", random_state=random_state
        )
        umap_coords = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        _scatter(ax, umap_coords,
                 f"UMAP – Pooled Test Embeddings (5-fold)\nSilhouette = {sil_score:.4f}",
                 "UMAP-1", "UMAP-2")
        plt.tight_layout()
        umap_path = os.path.join(save_dir, f"{prefix}_umap.png")
        plt.savefig(umap_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  UMAP saved: {umap_path}")
    except ImportError:
        print("  Warning: umap-learn not installed. Skipping UMAP. "
              "Install with: pip install umap-learn")

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    perp = min(perplexity, max(5, X.shape[0] // 3))
    tsne = TSNE(
        n_components=2, perplexity=perp, learning_rate=200,
        n_iter=1000, random_state=random_state, init="pca", verbose=0
    )
    tsne_coords = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter(ax, tsne_coords,
             f"t-SNE – Pooled Test Embeddings (5-fold)\nSilhouette = {sil_score:.4f}",
             "t-SNE-1", "t-SNE-2")
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, f"{prefix}_tsne.png")
    plt.savefig(tsne_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE saved: {tsne_path}")

    # ------------------------------------------------------------------
    # Combined side-by-side
    # ------------------------------------------------------------------
    if umap_coords is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        _scatter(ax1, umap_coords, "UMAP Projection", "UMAP-1", "UMAP-2")
        _scatter(ax2, tsne_coords, "t-SNE Projection", "t-SNE-1", "t-SNE-2")
        plt.suptitle(
            f"Pooled Test Embedding Visualizations (5-fold CV)  |  Silhouette = {sil_score:.4f}",
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        combined_path = os.path.join(save_dir, f"{prefix}_combined.png")
        plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Combined plot saved: {combined_path}")

    return sil_score


# ===============================
# Edge Ablation Experiment
# ===============================

@torch.no_grad()
def run_edge_ablation_experiment(model, test_loader, device, n_permutations=30):
    """
    Compare model performance with learned edges vs ablated edges

    This experiment tests whether the learned edge values (between 0 and 1)
    have statistical significance in improving AUC or F1 score.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        n_permutations: Number of random permutations for statistical testing

    Returns:
        Dictionary with ablation results and statistical tests
    """
    results = {}

    print(f"\n{'='*70}")
    print("EDGE ABLATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Testing significance of learned edge values...")
    print(f"Running {n_permutations} permutations for statistical testing\n")

    # Store original adjacency and attention mask
    original_A = model.A.clone()
    original_attn_mask = model.attn_mask.clone()

    # 1. Baseline: Learned edges (original)
    print("Condition 1: Learned edges (baseline)...")
    baseline_metrics = evaluate_metrics(model, test_loader, device)
    results['learned'] = {
        'auc': baseline_metrics['auc'],
        'f1': baseline_metrics['f1_binary'],
        'acc': baseline_metrics['acc'],
        'precision': baseline_metrics['precision'],
        'recall': baseline_metrics['recall']
    }
    print(f"  AUC={baseline_metrics['auc']:.4f}, F1={baseline_metrics['f1_binary']:.4f}")

    # 2. Random edges (multiple runs for statistics)
    print(f"\nCondition 2: Random edges ({n_permutations} runs)...")
    random_aucs, random_f1s = [], []
    for i in tqdm(range(n_permutations), desc="Random edges", leave=False):
        random_A = torch.rand_like(original_A)
        random_A = 0.5 * (random_A + random_A.T)  # Symmetrize
        random_A.fill_diagonal_(0.0)
        model.A = random_A

        metrics = evaluate_metrics(model, test_loader, device)
        random_aucs.append(metrics['auc'])
        random_f1s.append(metrics['f1_binary'])

    results['random'] = {
        'auc_mean': np.mean(random_aucs),
        'auc_std': np.std(random_aucs),
        'f1_mean': np.mean(random_f1s),
        'f1_std': np.std(random_f1s),
        'all_aucs': random_aucs,
        'all_f1s': random_f1s
    }
    print(f"  AUC={np.mean(random_aucs):.4f}±{np.std(random_aucs):.4f}, "
          f"F1={np.mean(random_f1s):.4f}±{np.std(random_f1s):.4f}")

    # 3. Fixed edges (all 0.5)
    print("\nCondition 3: Fixed edges (all 0.5)...")
    model.A = torch.full_like(original_A, 0.5)
    model.A.fill_diagonal_(0.0)
    fixed_05_metrics = evaluate_metrics(model, test_loader, device)
    results['fixed_0.5'] = {
        'auc': fixed_05_metrics['auc'],
        'f1': fixed_05_metrics['f1_binary']
    }
    print(f"  AUC={fixed_05_metrics['auc']:.4f}, F1={fixed_05_metrics['f1_binary']:.4f}")

    # 4. Fixed edges (all 1.0 - fully connected)
    print("\nCondition 4: Fixed edges (all 1.0 - fully connected)...")
    model.A = torch.ones_like(original_A)
    model.A.fill_diagonal_(0.0)
    fixed_10_metrics = evaluate_metrics(model, test_loader, device)
    results['fixed_1.0'] = {
        'auc': fixed_10_metrics['auc'],
        'f1': fixed_10_metrics['f1_binary']
    }
    print(f"  AUC={fixed_10_metrics['auc']:.4f}, F1={fixed_10_metrics['f1_binary']:.4f}")

    # 5. Fixed edges (all 0.0 - no edges)
    print("\nCondition 5: Fixed edges (all 0.0 - no connections)...")
    model.A = torch.zeros_like(original_A)
    fixed_00_metrics = evaluate_metrics(model, test_loader, device)
    results['fixed_0.0'] = {
        'auc': fixed_00_metrics['auc'],
        'f1': fixed_00_metrics['f1_binary']
    }
    print(f"  AUC={fixed_00_metrics['auc']:.4f}, F1={fixed_00_metrics['f1_binary']:.4f}")

    # 6. Shuffled edges (permute the learned values)
    print(f"\nCondition 6: Shuffled edges ({n_permutations} runs)...")
    shuffled_aucs, shuffled_f1s = [], []
    flat_A = original_A.flatten()
    for i in tqdm(range(n_permutations), desc="Shuffled edges", leave=False):
        perm_idx = torch.randperm(flat_A.size(0), device=device)
        shuffled_A = flat_A[perm_idx].reshape(original_A.shape)
        shuffled_A = 0.5 * (shuffled_A + shuffled_A.T)  # Symmetrize
        shuffled_A.fill_diagonal_(0.0)
        model.A = shuffled_A

        metrics = evaluate_metrics(model, test_loader, device)
        shuffled_aucs.append(metrics['auc'])
        shuffled_f1s.append(metrics['f1_binary'])

    results['shuffled'] = {
        'auc_mean': np.mean(shuffled_aucs),
        'auc_std': np.std(shuffled_aucs),
        'f1_mean': np.mean(shuffled_f1s),
        'f1_std': np.std(shuffled_f1s),
        'all_aucs': shuffled_aucs,
        'all_f1s': shuffled_f1s
    }
    print(f"  AUC={np.mean(shuffled_aucs):.4f}±{np.std(shuffled_aucs):.4f}, "
          f"F1={np.mean(shuffled_f1s):.4f}±{np.std(shuffled_f1s):.4f}")

    # 7. Binary thresholded edges (threshold at 0.5)
    print("\nCondition 7: Binary edges (threshold at 0.5)...")
    binary_A = (original_A > 0.5).float()
    model.A = binary_A
    binary_metrics = evaluate_metrics(model, test_loader, device)
    results['binary_0.5'] = {
        'auc': binary_metrics['auc'],
        'f1': binary_metrics['f1_binary']
    }
    print(f"  AUC={binary_metrics['auc']:.4f}, F1={binary_metrics['f1_binary']:.4f}")

    # Restore original
    model.A = original_A
    model.attn_mask = original_attn_mask

    # === STATISTICAL SIGNIFICANCE TESTS ===
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    # One-sample t-test: Is learned AUC significantly > mean of random?
    t_stat_auc_random, p_value_auc_random = stats.ttest_1samp(random_aucs, baseline_metrics['auc'])
    t_stat_f1_random, p_value_f1_random = stats.ttest_1samp(random_f1s, baseline_metrics['f1_binary'])

    # One-tailed: we want learned > random
    p_auc_random_onetail = p_value_auc_random / 2 if t_stat_auc_random < 0 else 1 - p_value_auc_random / 2
    p_f1_random_onetail = p_value_f1_random / 2 if t_stat_f1_random < 0 else 1 - p_value_f1_random / 2

    # Same for shuffled
    t_stat_auc_shuffled, p_value_auc_shuffled = stats.ttest_1samp(shuffled_aucs, baseline_metrics['auc'])
    t_stat_f1_shuffled, p_value_f1_shuffled = stats.ttest_1samp(shuffled_f1s, baseline_metrics['f1_binary'])

    p_auc_shuffled_onetail = p_value_auc_shuffled / 2 if t_stat_auc_shuffled < 0 else 1 - p_value_auc_shuffled / 2
    p_f1_shuffled_onetail = p_value_f1_shuffled / 2 if t_stat_f1_shuffled < 0 else 1 - p_value_f1_shuffled / 2

    results['significance'] = {
        'learned_vs_random': {
            'auc_improvement': baseline_metrics['auc'] - np.mean(random_aucs),
            'f1_improvement': baseline_metrics['f1_binary'] - np.mean(random_f1s),
            'auc_p_value': float(p_auc_random_onetail),
            'f1_p_value': float(p_f1_random_onetail),
            'auc_significant_0.05': p_auc_random_onetail < 0.05,
            'auc_significant_0.01': p_auc_random_onetail < 0.01,
            'f1_significant_0.05': p_f1_random_onetail < 0.05,
            'f1_significant_0.01': p_f1_random_onetail < 0.01
        },
        'learned_vs_shuffled': {
            'auc_improvement': baseline_metrics['auc'] - np.mean(shuffled_aucs),
            'f1_improvement': baseline_metrics['f1_binary'] - np.mean(shuffled_f1s),
            'auc_p_value': float(p_auc_shuffled_onetail),
            'f1_p_value': float(p_f1_shuffled_onetail),
            'auc_significant_0.05': p_auc_shuffled_onetail < 0.05,
            'auc_significant_0.01': p_auc_shuffled_onetail < 0.01,
            'f1_significant_0.05': p_f1_shuffled_onetail < 0.05,
            'f1_significant_0.01': p_f1_shuffled_onetail < 0.01
        }
    }

    # Print significance results
    sig = results['significance']

    print("\nLearned vs Random Edges:")
    print(f"  AUC improvement: {sig['learned_vs_random']['auc_improvement']:+.4f} "
          f"(p={sig['learned_vs_random']['auc_p_value']:.4f})")
    if sig['learned_vs_random']['auc_significant_0.01']:
        print(f"    → HIGHLY SIGNIFICANT (p < 0.01)")
    elif sig['learned_vs_random']['auc_significant_0.05']:
        print(f"    → SIGNIFICANT (p < 0.05)")
    else:
        print(f"    → NOT SIGNIFICANT (p >= 0.05)")

    print(f"  F1 improvement:  {sig['learned_vs_random']['f1_improvement']:+.4f} "
          f"(p={sig['learned_vs_random']['f1_p_value']:.4f})")
    if sig['learned_vs_random']['f1_significant_0.01']:
        print(f"    → HIGHLY SIGNIFICANT (p < 0.01)")
    elif sig['learned_vs_random']['f1_significant_0.05']:
        print(f"    → SIGNIFICANT (p < 0.05)")
    else:
        print(f"    → NOT SIGNIFICANT (p >= 0.05)")

    print("\nLearned vs Shuffled Edges:")
    print(f"  AUC improvement: {sig['learned_vs_shuffled']['auc_improvement']:+.4f} "
          f"(p={sig['learned_vs_shuffled']['auc_p_value']:.4f})")
    if sig['learned_vs_shuffled']['auc_significant_0.01']:
        print(f"    → HIGHLY SIGNIFICANT (p < 0.01)")
    elif sig['learned_vs_shuffled']['auc_significant_0.05']:
        print(f"    → SIGNIFICANT (p < 0.05)")
    else:
        print(f"    → NOT SIGNIFICANT (p >= 0.05)")

    print(f"  F1 improvement:  {sig['learned_vs_shuffled']['f1_improvement']:+.4f} "
          f"(p={sig['learned_vs_shuffled']['f1_p_value']:.4f})")
    if sig['learned_vs_shuffled']['f1_significant_0.01']:
        print(f"    → HIGHLY SIGNIFICANT (p < 0.01)")
    elif sig['learned_vs_shuffled']['f1_significant_0.05']:
        print(f"    → SIGNIFICANT (p < 0.05)")
    else:
        print(f"    → NOT SIGNIFICANT (p >= 0.05)")

    print(f"\n{'='*70}\n")

    return results


def visualize_edge_ablation(results, save_path):
    """Create visualization of ablation results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    conditions = ['Learned', 'Random', 'Shuffled', 'Fixed(0)', 'Fixed(0.5)', 'Fixed(1)', 'Binary']

    auc_means = [
        results['learned']['auc'],
        results['random']['auc_mean'],
        results['shuffled']['auc_mean'],
        results['fixed_0.0']['auc'],
        results['fixed_0.5']['auc'],
        results['fixed_1.0']['auc'],
        results['binary_0.5']['auc']
    ]
    auc_stds = [
        0,
        results['random']['auc_std'],
        results['shuffled']['auc_std'],
        0, 0, 0, 0
    ]

    f1_means = [
        results['learned']['f1'],
        results['random']['f1_mean'],
        results['shuffled']['f1_mean'],
        results['fixed_0.0']['f1'],
        results['fixed_0.5']['f1'],
        results['fixed_1.0']['f1'],
        results['binary_0.5']['f1']
    ]
    f1_stds = [
        0,
        results['random']['f1_std'],
        results['shuffled']['f1_std'],
        0, 0, 0, 0
    ]

    colors = ['green', 'red', 'orange', 'gray', 'blue', 'purple', 'brown']

    # AUC plot
    x = np.arange(len(conditions))
    bars1 = axes[0].bar(x, auc_means, yerr=auc_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('AUC by Edge Condition', fontsize=14, weight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=45, ha='right')
    axes[0].axhline(y=results['learned']['auc'], color='green', linestyle='--', alpha=0.5, label='Learned baseline')
    axes[0].legend()

    # Add significance markers
    sig = results.get('significance', {})
    if sig.get('learned_vs_random', {}).get('auc_significant_0.05'):
        axes[0].annotate('*', xy=(1, auc_means[1] + auc_stds[1] + 0.02), ha='center', fontsize=16, color='red')
    if sig.get('learned_vs_shuffled', {}).get('auc_significant_0.05'):
        axes[0].annotate('*', xy=(2, auc_means[2] + auc_stds[2] + 0.02), ha='center', fontsize=16, color='red')

    # F1 plot
    bars2 = axes[1].bar(x, f1_means, yerr=f1_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Edge Condition', fontsize=14, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=45, ha='right')
    axes[1].axhline(y=results['learned']['f1'], color='green', linestyle='--', alpha=0.5, label='Learned baseline')
    axes[1].legend()

    # Add significance markers
    if sig.get('learned_vs_random', {}).get('f1_significant_0.05'):
        axes[1].annotate('*', xy=(1, f1_means[1] + f1_stds[1] + 0.02), ha='center', fontsize=16, color='red')
    if sig.get('learned_vs_shuffled', {}).get('f1_significant_0.05'):
        axes[1].annotate('*', xy=(2, f1_means[2] + f1_stds[2] + 0.02), ha='center', fontsize=16, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation visualization saved to: {save_path}")


def visualize_edge_distribution(model, save_path):
    """Visualize the distribution of learned edge values"""
    A = model.A.cpu().numpy()

    # Get upper triangle (excluding diagonal)
    upper_tri = A[np.triu_indices_from(A, k=1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of edge values
    axes[0].hist(upper_tri, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Edge Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Learned Edge Values', fontsize=14, weight='bold')
    axes[0].axvline(x=np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.3f}')
    axes[0].axvline(x=np.median(upper_tri), color='orange', linestyle='--', label=f'Median: {np.median(upper_tri):.3f}')
    axes[0].legend()

    # Heatmap of edge values
    im = axes[1].imshow(A, cmap='viridis', aspect='auto')
    axes[1].set_xlabel('Pathway Index', fontsize=12)
    axes[1].set_ylabel('Pathway Index', fontsize=12)
    axes[1].set_title('Learned Edge Matrix (Adjacency)', fontsize=14, weight='bold')
    plt.colorbar(im, ax=axes[1], label='Edge Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Edge distribution visualization saved to: {save_path}")


# ===============================
# Visualization Functions
# ===============================

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
                    color='white' if cm_mean[i, j] > cm_mean.max() / 2 else 'black',
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
        precision_arr, recall_arr, _ = precision_recall_curve(yt, yp)
        pr_auc = auc(recall_arr, precision_arr)
        aucs.append(pr_auc)

        recall_rev = recall_arr[::-1]
        precision_rev = precision_arr[::-1]
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
# Pathway Analysis Functions
# ===============================

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
    """Save pathway interaction matrix to CSV"""
    df = pd.DataFrame(matrix, index=pathway_ids, columns=pathway_ids)
    df.insert(0, "Pathway_Name", pathway_names)
    df.index.name = "Pathway_ID"
    df.to_csv(save_path)


def extract_and_save_pathway_interactions(model, loader, pathway_ids, pathway_names,
                                          device, outdir, fold_num):
    """Extract and save pathway-pathway interaction matrix"""
    attn_matrix = extract_pathway_attention_matrix(model, loader, device)
    if attn_matrix is None:
        print("Warning: No attention weights available to build pathway interaction matrix.")
        return None

    os.makedirs(outdir, exist_ok=True)
    attn_csv = os.path.join(outdir, f"pathway_attention_fold{fold_num}.csv")
    save_pathway_matrix_csv(attn_matrix, pathway_ids, pathway_names, attn_csv)
    print(f"Pathway attention matrix saved to: {attn_csv}")

    return attn_matrix


def analyze_and_save_pathway_importance(model, test_loader, pathway_names, device, outdir, fold_num, top_k=10):
    """
    Analyze pathway importance and save top influential pathways.
    Rankings are based on differential attention (metastatic - primary) to
    identify pathways that specifically contribute to metastatic progression.
    """
    model.eval()
    weights_by_class = {0: [], 1: []}

    print(f"\n{'='*70}")
    print(f"PATHWAY IMPORTANCE ANALYSIS - FOLD {fold_num}")
    print(f"{'='*70}")

    with torch.no_grad():
        for mut_b, cnv_b, y_b in test_loader:
            mut_b, cnv_b = mut_b.to(device), cnv_b.to(device)
            _, extras = model(mut_b, cnv_b, return_extras=True)
            pw = extras["pathway_weights"].cpu().numpy()
            labels = y_b.numpy()
            for cls in (0, 1):
                mask = labels == cls
                if mask.any():
                    weights_by_class[cls].append(pw[mask])

    meta_weights = np.vstack(weights_by_class[1]) if weights_by_class[1] else None
    prim_weights = np.vstack(weights_by_class[0]) if weights_by_class[0] else None

    if meta_weights is None:
        print("Warning: No metastatic samples found in loader. Falling back to overall mean.")
        all_weights = np.vstack(weights_by_class[0])
        mean_meta = mean_prim = mean_pathway_weights = np.mean(all_weights, axis=0)
        diff_scores = mean_pathway_weights
    else:
        mean_meta = np.mean(meta_weights, axis=0)
        mean_prim = np.mean(prim_weights, axis=0) if prim_weights is not None else np.zeros_like(mean_meta)
        # Differential score: how much more active each pathway is in metastatic vs primary
        diff_scores = mean_meta - mean_prim
        mean_pathway_weights = mean_meta  # kept for backwards-compat in return value

    pathway_ranking = np.argsort(-diff_scores)

    print(f"\nTop {top_k} Pathways Contributing to Metastatic Progression")
    print(f"  (ranked by differential attention: metastatic - primary)")
    print("-" * 95)
    print(f"{'Rank':<6} {'Pathway Name':<45} {'Meta Score':<14} {'Prim Score':<14} {'Differential':<12}")
    print("-" * 95)

    pathway_results = []
    for rank, idx in enumerate(pathway_ranking[:top_k], 1):
        pathway_name = pathway_names[idx] if idx < len(pathway_names) else f"Pathway_{idx}"
        display_name = pathway_name[:43] + ".." if len(pathway_name) > 45 else pathway_name
        m_score = float(mean_meta[idx])
        p_score = float(mean_prim[idx])
        d_score = float(diff_scores[idx])
        print(f"{rank:<6} {display_name:<45} {m_score:<14.8f} {p_score:<14.8f} {d_score:<12.8f}")
        pathway_results.append({
            'rank': rank,
            'pathway_idx': int(idx),
            'pathway_name': pathway_name,
            'metastatic_score': m_score,
            'primary_score': p_score,
            'differential_score': d_score,
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
            'metastatic_score': float(mean_meta[idx]),
            'primary_score': float(mean_prim[idx]),
            'differential_score': float(diff_scores[idx]),
        })

    csv_file = os.path.join(outdir, f"pathway_importance_fold{fold_num}.csv")
    df_rankings = pd.DataFrame(all_rankings)
    df_rankings.to_csv(csv_file, index=False)
    print(f"\nComplete pathway rankings saved to: {csv_file}")

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


def aggregate_pathway_rankings_across_folds(outdir, n_folds=5):
    """Aggregate pathway importance rankings across all folds"""
    pathway_dir = os.path.join(outdir, "pathway_importance")

    if not os.path.exists(pathway_dir):
        print(f"Warning: Pathway importance directory not found: {pathway_dir}")
        return None

    print(f"\n{'='*70}")
    print("AGGREGATING PATHWAY IMPORTANCE ACROSS ALL FOLDS")
    print(f"{'='*70}")

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

    combined_df = pd.concat(all_fold_data, ignore_index=True)

    pathway_stats = combined_df.groupby('pathway_idx').agg({
        'pathway_name': 'first',
        'differential_score': ['mean', 'std', 'min', 'max'],
        'metastatic_score': 'mean',
        'primary_score': 'mean',
        'rank': ['mean', 'std', 'min', 'max']
    }).reset_index()

    pathway_stats.columns = [
        'pathway_idx', 'pathway_name',
        'mean_differential', 'std_differential', 'min_differential', 'max_differential',
        'mean_metastatic', 'mean_primary',
        'mean_rank', 'std_rank', 'min_rank', 'max_rank'
    ]

    pathway_stats = pathway_stats.sort_values('mean_differential', ascending=False).reset_index(drop=True)
    pathway_stats['overall_rank'] = range(1, len(pathway_stats) + 1)

    print(f"\nTop 20 Pathways Contributing to Metastatic Progression (Across {n_folds} Folds):")
    print(f"  (ranked by mean differential attention: metastatic - primary)")
    print("-" * 105)
    print(f"{'Rank':<6} {'Name':<40} {'Mean Differential':<20} {'Meta Score':<14} {'Prim Score':<12}")
    print("-" * 105)

    for _, row in pathway_stats.head(20).iterrows():
        display_name = row['pathway_name'][:38] + ".." if len(row['pathway_name']) > 40 else row['pathway_name']
        diff_str = f"{row['mean_differential']:.6f} ± {row['std_differential']:.6f}"
        print(f"{int(row['overall_rank']):<6} {display_name:<40} {diff_str:<20} "
              f"{row['mean_metastatic']:<14.6f} {row['mean_primary']:<12.6f}")

    aggregate_csv = os.path.join(pathway_dir, "aggregate_pathway_importance.csv")
    pathway_stats.to_csv(aggregate_csv, index=False)
    print(f"\nAggregate pathway importance saved to: {aggregate_csv}")

    top_pathways = []
    for _, row in pathway_stats.head(20).iterrows():
        top_pathways.append({
            'overall_rank': int(row['overall_rank']),
            'pathway_idx': int(row['pathway_idx']),
            'pathway_name': row['pathway_name'],
            'mean_differential': float(row['mean_differential']),
            'std_differential': float(row['std_differential']),
            'mean_metastatic': float(row['mean_metastatic']),
            'mean_primary': float(row['mean_primary']),
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


# ===============================
# MAIN FUNCTION - 5-FOLD CV
# ===============================

def main_5fold_cv():
    """Main function for 5-fold stratified cross-validation with edge ablation"""
    ap = argparse.ArgumentParser(description="Pathway Graph Transformer with Edge Ablation Experiment")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="outputs_5fold_cv_v2")
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
                    help="Use fully connected graph instead of sparse adjacency")
    ap.add_argument("--run_ablation", action='store_true', default=True,
                    help="Run edge ablation experiment after training")
    ap.add_argument("--ablation_permutations", type=int, default=30,
                    help="Number of permutations for ablation statistical tests")

    args = ap.parse_args()

    print(f"\n{'='*70}")
    print("5-FOLD STRATIFIED CROSS-VALIDATION WITH EDGE ABLATION")
    print(f"{'='*70}")
    print(f"Model: Edge-Aware Graph Transformer")
    print(f"CV Strategy: 5-Fold Stratified")
    print(f"Edge Ablation: {'Enabled' if args.run_ablation else 'Disabled'}")
    print(f"{'='*70}\n")

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
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
        print(f"  Label {label}: {count} patients ({count / len(y_arr) * 100:.1f}%)")

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
    all_fold_attention = []
    all_fold_ablation = []

    all_test_y_true = []
    all_test_y_pred = []
    all_test_y_probs = []

    # Accumulators for pooled test embeddings (one entry per fold)
    all_test_embeddings = []
    all_test_labels_emb = []

    print(f"\nStarting 5-fold CV evaluation...")

    # MAIN FOLD LOOP
    for fold_data in folds:
        fold = fold_data['fold']
        train_idx = np.array(fold_data['train_idx'])
        val_idx = np.array(fold_data['val_idx'])
        test_idx = np.array(fold_data['test_idx'])

        print(f"\n{'='*50} FOLD {fold}/5 {'='*50}")

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
            use_edge_mask=not args.use_full_graph,
            use_edge_bias=True,
            pe_dim=args.pe_dim,
            use_batch_norm=args.use_batch_norm,
            use_edge_aware_blocks=args.use_edge_aware_blocks,
            full_graph_attention=args.use_full_graph
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
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:3d} | Loss: {tr_loss:.4f} | Val AUC: {val_metric:.4f} | "
                      f"Patience: {epochs_no_improve}/{args.patience}")

            if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model and evaluate (FIXED: removed duplicate)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        final_val_metrics = evaluate_metrics(model, val_loader, device)
        test_metrics = evaluate_metrics(model, test_loader, device)

        # === EDGE ABLATION EXPERIMENT ===
        if args.run_ablation:
            ablation_results = run_edge_ablation_experiment(
                model=model,
                test_loader=test_loader,
                device=device,
                n_permutations=args.ablation_permutations
            )
            all_fold_ablation.append(ablation_results)

            # Save ablation results
            ablation_dir = os.path.join(args.outdir, "edge_ablation")
            os.makedirs(ablation_dir, exist_ok=True)

            # Prepare JSON-serializable version
            ablation_json = {}
            for k, v in ablation_results.items():
                if k == 'significance':
                    ablation_json[k] = v
                elif isinstance(v, dict):
                    ablation_json[k] = {kk: (vv if not isinstance(vv, (list, np.ndarray)) else
                                            list(vv) if isinstance(vv, np.ndarray) else vv)
                                       for kk, vv in v.items()}
                else:
                    ablation_json[k] = v

            with open(os.path.join(ablation_dir, f"ablation_fold{fold}.json"), 'w') as f:
                json.dump(ablation_json, f, indent=2)

            # Visualize
            visualize_edge_ablation(
                ablation_results,
                os.path.join(ablation_dir, f"ablation_fold{fold}.png")
            )

            # Visualize edge distribution
            visualize_edge_distribution(
                model,
                os.path.join(ablation_dir, f"edge_distribution_fold{fold}.png")
            )

        # Analyze pathway importance
        pathway_importance_dir = os.path.join(args.outdir, "pathway_importance")
        os.makedirs(pathway_importance_dir, exist_ok=True)

        analyze_and_save_pathway_importance(
            model=model,
            test_loader=test_loader,
            pathway_names=pathway_names,
            device=device,
            outdir=pathway_importance_dir,
            fold_num=fold,
            top_k=10
        )

        # Extract pathway-pathway interaction matrix
        fold_attn = extract_and_save_pathway_interactions(
            model=model,
            loader=test_loader,
            pathway_ids=pathway_ids,
            pathway_names=pathway_names,
            device=device,
            outdir=pathway_interaction_dir,
            fold_num=fold
        )
        if fold_attn is not None:
            all_fold_attention.append(fold_attn)

        # Store test predictions
        all_test_y_true.append(test_metrics['y_true'])
        all_test_y_pred.append(test_metrics['y_pred'])
        all_test_y_probs.append(test_metrics['y_probs'])

        # Extract test embeddings for later pooled visualization
        # Only test-set embeddings are used: they are honest held-out
        # representations that reflect genuine generalisation.
        fold_embs, fold_emb_labels = extract_embeddings(model, test_loader, device)
        all_test_embeddings.append(fold_embs)
        all_test_labels_emb.append(fold_emb_labels)

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

    # === AGGREGATE ABLATION RESULTS ===
    if args.run_ablation and all_fold_ablation:
        print(f"\n{'='*70}")
        print("AGGREGATE EDGE ABLATION RESULTS")
        print(f"{'='*70}")

        # Aggregate significance across folds
        auc_improvements_random = [r['significance']['learned_vs_random']['auc_improvement']
                                   for r in all_fold_ablation]
        f1_improvements_random = [r['significance']['learned_vs_random']['f1_improvement']
                                  for r in all_fold_ablation]
        auc_significant_random = [r['significance']['learned_vs_random']['auc_significant_0.05']
                                  for r in all_fold_ablation]
        f1_significant_random = [r['significance']['learned_vs_random']['f1_significant_0.05']
                                 for r in all_fold_ablation]

        print(f"\nLearned vs Random (across {len(all_fold_ablation)} folds):")
        print(f"  Mean AUC improvement: {np.mean(auc_improvements_random):+.4f} ± {np.std(auc_improvements_random):.4f}")
        print(f"  Mean F1 improvement:  {np.mean(f1_improvements_random):+.4f} ± {np.std(f1_improvements_random):.4f}")
        print(f"  Folds with significant AUC improvement: {sum(auc_significant_random)}/{len(auc_significant_random)}")
        print(f"  Folds with significant F1 improvement:  {sum(f1_significant_random)}/{len(f1_significant_random)}")

        # Save aggregate ablation summary
        ablation_summary = {
            'n_folds': len(all_fold_ablation),
            'learned_vs_random': {
                'auc_improvement_mean': float(np.mean(auc_improvements_random)),
                'auc_improvement_std': float(np.std(auc_improvements_random)),
                'f1_improvement_mean': float(np.mean(f1_improvements_random)),
                'f1_improvement_std': float(np.std(f1_improvements_random)),
                'folds_auc_significant': int(sum(auc_significant_random)),
                'folds_f1_significant': int(sum(f1_significant_random))
            }
        }

        with open(os.path.join(args.outdir, "edge_ablation", "aggregate_ablation_summary.json"), 'w') as f:
            json.dump(ablation_summary, f, indent=2)

    # Generate aggregate visualizations
    print(f"\nGenerating aggregate visualizations...")
    viz_dir = os.path.join(args.outdir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_suffix = "fullgraph" if args.use_full_graph else "sparse"

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

    # Pooled test embedding visualisation (UMAP + t-SNE)
    # Using only test-set embeddings: honest, held-out representations that
    # reflect genuine generalisation.  Each sample appears in the test set
    # exactly once across the 5 folds, so pooling gives full-dataset coverage
    # with zero data leakage — making cross-baseline comparison fair.
    embedding_silhouette = float('nan')
    try:
        print(f"\nGenerating pooled test embedding visualizations (UMAP + t-SNE)...")
        emb_mat = np.concatenate(all_test_embeddings, axis=0)
        emb_labels = np.concatenate(all_test_labels_emb, axis=0)
        emb_viz_dir = os.path.join(viz_dir, "embeddings")
        embedding_silhouette = plot_umap_tsne_embeddings(
            emb_mat, emb_labels,
            save_dir=emb_viz_dir,
            prefix=f"test_embeddings_{graph_suffix}_{timestamp}",
            random_state=args.random_state,
            dpi=300
        )
        print(f"  Pooled test silhouette score: {embedding_silhouette:.4f}")
    except Exception as e:
        print(f"Warning: Could not generate embedding visualizations: {str(e)}")

    # Aggregate pathway importance
    try:
        print(f"\nAggregating pathway importance across folds...")
        aggregate_pathway_rankings_across_folds(outdir=args.outdir, n_folds=5)
    except Exception as e:
        print(f"Warning: Could not aggregate pathway importance: {str(e)}")

    # Aggregate pathway-pathway attention
    if all_fold_attention:
        aggregate_attention = np.mean(np.stack(all_fold_attention, axis=0), axis=0)
        aggregate_csv = os.path.join(pathway_interaction_dir, "pathway_attention_aggregate.csv")
        save_pathway_matrix_csv(aggregate_attention, pathway_ids, pathway_names, aggregate_csv)
        print(f"Aggregate pathway attention matrix saved to: {aggregate_csv}")

    # Save results
    results_file = os.path.join(args.outdir, f"5fold_results_{graph_suffix}_{timestamp}.json")
    results_data = {
        'timestamp': timestamp,
        'model_type': '5-Fold_CV_Edge_Aware_Graph_Transformer_v2',
        'graph_mode': 'full_graph' if args.use_full_graph else 'sparse_graph',
        'cv_strategy': '5-Fold Stratified (20% test per fold, 10% val from remaining)',
        'edge_ablation_enabled': args.run_ablation,
        'validation_summary': {k: {'mean': float(np.mean([m[k] for m in all_fold_val_metrics])),
                                   'std': float(np.std([m[k] for m in all_fold_val_metrics]))}
                              for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'test_summary': {k: {'mean': float(np.mean([m[k] for m in all_fold_test_metrics])),
                            'std': float(np.std([m[k] for m in all_fold_test_metrics]))}
                        for k in ['auc', 'f1_binary', 'acc', 'precision', 'recall']},
        'fold_details': all_fold_details,
        'embedding_silhouette_score': embedding_silhouette,
        'config': vars(args)
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results_data


if __name__ == "__main__":
    results = main_5fold_cv()
    print(f"\n{'='*70}")
    print("5-Fold CV Experiment Complete!")
    print(f"{'='*70}")
    print(f"Mean Test AUC: {results['test_summary']['auc']['mean']:.4f} ± {results['test_summary']['auc']['std']:.4f}")
    print(f"Mean Test F1:  {results['test_summary']['f1_binary']['mean']:.4f} ± {results['test_summary']['f1_binary']['std']:.4f}")
