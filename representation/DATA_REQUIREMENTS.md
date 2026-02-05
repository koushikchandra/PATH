# Data Requirements for Generating Embeddings

To generate the `5fold_test_embeddings_*.csv` file, you need to run `representation.py` with the required input data.

## Required Data Files

Create a `data/` directory with these 5 CSV files:

```
data/
├── mutation_data.csv        # Gene mutation data
├── cnv_data.csv            # Copy number variation data
├── patient_labels.csv      # Patient class labels
├── filtered_pathways.csv   # Pathway definitions
└── adjacency_matrix.csv    # Pathway-pathway relationships
```

## File Format Specifications

### 1. `mutation_data.csv`
Binary mutation data (0 = no mutation, 1 = mutation)

| Patient_ID | GENE1 | GENE2 | GENE3 | ... |
|------------|-------|-------|-------|-----|
| P001       | 0     | 1     | 0     | ... |
| P002       | 1     | 0     | 1     | ... |
| P003       | 0     | 0     | 0     | ... |

**Format:**
- First column: Patient IDs
- Remaining columns: Gene names (must match `cnv_data.csv`)
- Values: 0 (wild-type) or 1 (mutated)

### 2. `cnv_data.csv`
Copy number variation data (continuous values)

| Patient_ID | GENE1 | GENE2 | GENE3 | ... |
|------------|-------|-------|-------|-----|
| P001       | -0.5  | 1.2   | 0.1   | ... |
| P002       | 0.8   | -1.1  | 0.3   | ... |
| P003       | 0.2   | 0.0   | -0.7  | ... |

**Format:**
- First column: Patient IDs (must match `mutation_data.csv`)
- Remaining columns: Gene names (must match `mutation_data.csv`)
- Values: Log2 copy number ratios or similar continuous CNV measurements

### 3. `patient_labels.csv`
Classification labels for patients

| Patient_ID | Label      |
|------------|------------|
| P001       | Primary    |
| P002       | Metastatic |
| P003       | Primary    |

**Format:**
- Column 1: Patient IDs (must match mutation/CNV data)
- Column 2: Class labels
  - Accepted values: `Primary`/`Metastatic`, `P`/`M`, or `0`/`1`
  - 0/Primary = Primary tumor
  - 1/Metastatic = Metastatic tumor

### 4. `filtered_pathways.csv`
Biological pathway definitions

| Pathway_ID | Pathway_Name              | Genes                    |
|------------|---------------------------|--------------------------|
| PW001      | EGFR signaling pathway    | EGFR,KRAS,BRAF,PIK3CA   |
| PW002      | p53 pathway               | TP53,MDM2,CDKN2A        |
| PW003      | DNA repair                | BRCA1,BRCA2,ATM         |

**Format:**
- `Pathway_ID`: Unique identifier for each pathway
- `Pathway_Name`: Human-readable pathway name
- `Genes`: Comma-separated or space-separated list of gene symbols
  - Genes must match column names in `mutation_data.csv` and `cnv_data.csv`

### 5. `adjacency_matrix.csv`
Pathway-pathway interaction strengths

|        | PW001 | PW002 | PW003 | ... |
|--------|-------|-------|-------|-----|
| PW001  | 0.0   | 0.3   | 0.1   | ... |
| PW002  | 0.3   | 0.0   | 0.5   | ... |
| PW003  | 0.1   | 0.5   | 0.0   | ... |

**Format:**
- Row/column indices: Pathway IDs (must match `filtered_pathways.csv`)
- Values: Interaction strength (0.0-1.0)
  - 0.0 = No interaction
  - 1.0 = Strong interaction
- Diagonal should be 0.0 (no self-loops)
- Matrix should be symmetric (or will be symmetrized automatically)

**Alternative:** Use `--use_full_graph` flag to skip this file and use a fully connected graph

## Quick Data Validation Checklist

✓ All patient IDs match across `mutation_data.csv`, `cnv_data.csv`, and `patient_labels.csv`
✓ Gene names match between `mutation_data.csv` and `cnv_data.csv`
✓ Pathway genes in `filtered_pathways.csv` exist in mutation/CNV data
✓ Pathway IDs in `adjacency_matrix.csv` match `filtered_pathways.csv`
✓ No missing values (NaN) in critical columns
✓ Labels are properly formatted (Primary/Metastatic or 0/1)

## Example: Create Sample Data

If you want to test the pipeline with synthetic data, you can create minimal sample files:

```bash
# Create data directory
mkdir -p data

# You'll need to create these files with your actual genomic data
# Or contact your bioinformatics team for preprocessed files
```

## Common Data Sources

Real genomic data typically comes from:
- **TCGA** (The Cancer Genome Atlas)
- **GEO** (Gene Expression Omnibus)
- **cBioPortal** for Cancer Genomics
- **ICGC** (International Cancer Genome Consortium)
- Your own sequencing experiments

## After Preparing Data

Once you have all 5 files in the `data/` directory, run:

```bash
python representation/representation.py \
    --data_dir data \
    --outdir outputs_5fold_cv \
    --epochs 200
```

This will:
1. Train the model using 5-fold cross-validation
2. Generate embeddings for test samples in each fold
3. Save embeddings to `outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv`
4. Create visualizations and performance metrics

## Estimated Runtime

- Small dataset (< 100 patients): 10-30 minutes
- Medium dataset (100-500 patients): 30-120 minutes
- Large dataset (> 500 patients): 2-4 hours

*Runtime depends on: number of patients, genes, pathways, epochs, and GPU availability*

## Using Full Graph Mode (Simpler Alternative)

If you don't have pathway interaction data (`adjacency_matrix.csv`), use:

```bash
python representation/representation.py \
    --data_dir data \
    --outdir outputs_5fold_cv \
    --use_full_graph \
    --epochs 200
```

This creates a fully connected pathway graph and doesn't require `adjacency_matrix.csv`.
