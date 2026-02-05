# Why Are There Five Clusters? Understanding the Embedding Artifact

## 🔍 The Problem

When visualizing `5fold_test_embeddings_*.csv`, you see **5 distinct clusters** instead of 2 (Primary vs Metastatic). Each cluster contains a mix of both classes.

### Visual Evidence:
- ✅ Exactly 5 spatial clusters
- ✅ Each cluster has mixed Primary (blue) and Metastatic (red) points
- ✅ Similar class distributions within each cluster

---

## 🧠 Root Cause: Cross-Validation Embedding Misalignment

The 5 clusters correspond to the **5 folds** in your cross-validation, NOT biological subtypes.

### What Happens During 5-Fold CV:

```
Fold 1: Train Model A → Test on 20% of data → Extract embeddings
Fold 2: Train Model B → Test on 20% of data → Extract embeddings
Fold 3: Train Model C → Test on 20% of data → Extract embeddings
Fold 4: Train Model D → Test on 20% of data → Extract embeddings
Fold 5: Train Model E → Test on 20% of data → Extract embeddings

Combined: Concatenate all test embeddings → 5fold_test_embeddings.csv
```

**Problem:** Models A, B, C, D, E each learned slightly different embedding spaces because they were trained on different data subsets.

### Why This Creates 5 Clusters:

1. **Each model learns its own coordinate system** for the embedding space
2. **Rotation/scaling differences** between models create spatial separation
3. **No shared reference frame** to align the embeddings
4. When concatenated, embeddings from each fold form a separate cluster

### Analogy:
Imagine 5 people describing locations using their own personal coordinate systems:
- Person A: "School is at (2, 3)"
- Person B: "School is at (-1, 5)" (rotated their axes 90°)
- Person C: "School is at (4, -2)" (scaled differently)

When you plot all descriptions together, the same "school" appears in 5 different locations!

---

## ✅ Solutions

### **Solution 1: Visualize Single Fold** (Quickest)

Extract and visualize embeddings from just one fold:

```bash
python representation/extract_single_fold.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \
    --fold 1 \
    --output fold1_only_embeddings.csv

python representation/visualize_embeddings.py \
    --input fold1_only_embeddings.csv \
    --output_dir ./single_fold_plots
```

**Pros:**
- ✅ Quick and simple
- ✅ Shows true embedding structure from one model

**Cons:**
- ❌ Only 20% of your data visualized
- ❌ Results may vary by fold

---

### **Solution 2: Align Embeddings** (Recommended)

Use Procrustes analysis to align all folds to a common space:

```bash
python representation/align_fold_embeddings.py \
    --input outputs_5fold_cv/embeddings/5fold_test_embeddings_*.csv \
    --output aligned_embeddings.csv

python representation/visualize_embeddings.py \
    --input aligned_embeddings.csv \
    --output_dir ./aligned_plots
```

**How it works:**
- Uses orthogonal Procrustes to find optimal rotation/scaling
- Aligns Folds 2-5 to Fold 1's coordinate system
- Preserves relative distances within each fold

**Pros:**
- ✅ Uses all 100% of data
- ✅ Statistically sound alignment method
- ✅ Preserves within-fold structure

**Cons:**
- ❌ Slightly more complex
- ❌ Assumes linear relationship between fold spaces

---

### **Solution 3: Train Single Final Model** (Best for Production)

Modify the training script to save embeddings from a single model trained on all data:

```python
# After 5-fold CV completes, train one final model on ALL data
# Then extract embeddings from that single model

# This requires modifying representation.py to add:
# - Final model training on full dataset
# - Embedding extraction from the final model
```

**Pros:**
- ✅ Single coherent embedding space
- ✅ Uses full model capacity
- ✅ Best for deployment/production

**Cons:**
- ❌ No test set (all data used for training)
- ❌ Can't evaluate generalization performance

---

## 📊 Expected Results After Fixing

After applying any solution, you should see:

### Instead of 5 clusters:
- **Biological structure emerges** (if present)
- **Primary vs Metastatic separation** (if learnable)
- **Potential subtypes** (if real biological heterogeneity exists)
- **Continuous gradients** (if disease progression is gradual)

### Possible outcomes:

**Scenario A: Good Separation**
```
Primary patients cluster together (blue)
Metastatic patients cluster together (red)
Some overlap in boundary regions (difficult cases)
```

**Scenario B: Biological Subtypes**
```
2-3 clusters with mixed labels
Each cluster may represent a molecular subtype
Primary/Metastatic is a feature, not the main grouping
```

**Scenario C: Continuous Spectrum**
```
No discrete clusters
Gradient from Primary-like to Metastatic-like
Represents disease progression continuum
```

---

## 🔬 Advanced: Investigating What You See

### Check if 5 clusters are truly fold-related:

```python
import pandas as pd

df = pd.read_csv("5fold_test_embeddings.csv")

# Add fold labels (assuming equal splits)
n_per_fold = len(df) // 5
df['fold'] = [i // n_per_fold + 1 for i in range(len(df))]

# Check if clusters align with folds
# Color by fold instead of class to verify
```

### Quantify alignment quality:

```python
from sklearn.metrics import silhouette_score

# Before alignment
silhouette_folds = silhouette_score(embeddings, fold_labels)

# After alignment
silhouette_class = silhouette_score(aligned_embeddings, class_labels)

# Higher silhouette_class = better separation by disease class
```

---

## 🎓 Why This Matters

### For Research:
- Cross-validation is for **evaluation**, not embedding visualization
- Need single reference frame for meaningful spatial interpretation
- Embedding quality depends on consistent coordinate system

### For Interpretation:
- 5 clusters ≠ 5 biological subtypes (likely an artifact)
- Always check if clustering aligns with known batch effects
- Use aligned embeddings for biological interpretation

### For Publication:
- Never show raw concatenated CV embeddings
- Use single-model embeddings or aligned embeddings
- Report alignment method in methods section

---

## 🛠️ Quick Decision Tree

```
Do you need to visualize ALL test samples?
├─ YES → Use Solution 2 (Align embeddings)
└─ NO
   └─ Is 20% of data sufficient?
      ├─ YES → Use Solution 1 (Single fold)
      └─ NO → Use Solution 3 (Train final model)
```

---

## 📚 Technical Background

### Procrustes Analysis
- Finds optimal rotation, reflection, and scaling
- Minimizes sum of squared differences
- Commonly used in shape analysis and embedding alignment

### Why CV Creates Different Spaces
- Different training data → different learned features
- Random initialization → different local minima
- Gradient descent path varies → different final parameters

### Mathematical Formulation
```
Given embeddings X (fold 1) and Y (fold 2):
Find rotation R that minimizes ||XR - Y||²

Solution: R = UV^T where UΣV^T = SVD(X^T Y)
```

---

## 🔗 Related Reading

- "Procrustes Problems" - Gower & Dijksterhuis
- "Embedding Alignment" - Hamilton et al., ACL 2016
- "Cross-lingual Word Embeddings" - Ruder et al., 2019
- "Representation Learning" - Bengio et al., 2013

---

## 💡 Key Takeaways

1. ✅ **5 clusters = 5 folds**, not 5 biological groups
2. ✅ **Cross-validation embeddings need alignment** for visualization
3. ✅ **Use single-fold or aligned embeddings** for interpretation
4. ✅ **Always validate** that clusters match biological expectations
5. ✅ **Document your approach** in papers/reports

---

## 🆘 Still Have Questions?

**Q: Should I always see 5 clusters with CV?**
A: Yes, if embeddings from different models are concatenated without alignment.

**Q: Is my model bad if I see 5 clusters?**
A: No! Your model is fine. This is expected behavior from concatenating embeddings.

**Q: Which solution should I use?**
A: For quick exploration: Solution 1. For comprehensive analysis: Solution 2.

**Q: Can I use these embeddings for downstream tasks?**
A: Not directly. Align them first, or extract from a single model.

**Q: Does this affect my CV performance metrics?**
A: No! Your AUC/F1 scores are still valid. This only affects embedding visualization.
