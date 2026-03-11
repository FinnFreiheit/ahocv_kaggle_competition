# Leaderboard Experiments — Jaguar Re-Identification

**Course**: Applied Hands-on Computer Vision  

**Competition**: [Jaguar Re-Identification](https://www.kaggle.com/competitions/jaguar-re-id/)  

[W&B Project](https://wandb.ai/finnfreiheit/Jaguar-Re-identification-Challenge?nw=nwuserfinnfrei)

---

## Shared Baseline Configuration

All experiments start from the same baseline configuration unless explicitly noted. This ensures comparability across experiments.

| Parameter | Value |
|-----------|-------|
| Backbone | MegaDescriptor-L-384 (frozen, ~305M params) |
| Projection Head | 1536 → 512 → 256 (BatchNorm + ReLU + Dropout(0.3)) |
| Loss | ArcFace (margin=0.5, scale=64.0) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5) |
| Batch Size | 32 |
| Epochs | 50 (patience=10) |
| Train/Val Split | Stratified 80/20 (seed=42) |
| Metric | Identity-balanced mAP |

---


## Pre-Augmentation

**Notebook**: [Assignment/experiment-preaugmentation.ipynb](Assignment/experiment-preaugmentation.ipynb)  
[Kaggle Submission](https://www.kaggle.com/code/finnfrei/experiment-preaugmentation?scriptVersionId=297879351)  


### Hypothesis

Augmenting training images **before** MegaDescriptor embedding extraction produces a larger, more diverse training set (4× expansion), leading to improved generalization and higher mAP.

### Methodology

1. For each training image, generate 3 augmented copies using a curated augmentation pipeline
2. Extract MegaDescriptor embeddings from all original + augmented images (4× more training embeddings)
3. Train ArcFace on the enlarged embedding set
4. Same validation and submission pipeline as baseline (no augmentation on val/test)

### Configuration Changes from Baseline

| Parameter | Value |
|-----------|-------|
| num_augmented_copies | 3 |
| RandomResizedCrop | scale=(0.7, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 |
| GaussianBlur | kernel=5, sigma=(0.1, 2.0) |
| Foreground-Only Random Erasing | p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), min_fg_frac=0.6 |

### Key Design Decision

Random erasing is applied only to the **foreground** (animal body) to avoid destroying the consistent white background. This preserves dataset structure while simulating partial occlusion of jaguar patterns.

### Result

- **Training set**: 1,516 → 6,064 samples (+300%)
- **Best Validation mAP**: 0.7878 (epoch 39)
- **Improvement**: **+6.3%** over baseline

---

## Backbone Architecture Comparison

**Notebook**: [Assignment/experiment-backbone-comparison.ipynb](Assignment/experiment-backbone-comparison.ipynb)  
[Kaggle Submission ID](https://www.kaggle.com/code/finnfrei/experiment-backbone-comparison?scriptVersionId=302240640)

### Hypothesis

Different backbone architectures capture different visual features. MegaDescriptor's wildlife-specific pre-training may outperform general-purpose models, but larger or modern self-supervised architectures might compensate.

### Methodology

1. **Sweep phase** (20 epochs each): Train all 4 backbones with identical projection head + ArcFace
2. **Full training** (50 epochs): Best backbone trained to convergence
3. All backbones are frozen — only projection head is trained

### Backbones Compared

| Backbone | Parameters | Embedding Dim | Input Size | Pre-Training |
|----------|-----------|---------------|-----------|-------------|
| MegaDescriptor-L-384 | ~305M | 1536 | 384 | Wildlife Re-ID |
| DINOv2-B-14 | ~86M | 768 | 518 | Self-supervised (LVD-142M) |
| ConvNeXt V2 Base | ~89M | 1024 | 224 | FCMAE → IN-22k → IN-1k |
| EfficientNet-B4 | ~19M | 1792 | 380 | Supervised (IN-1k) |


---

## Loss Function Comparison

**Notebook**: [Assignment/experiment-loss-functions.ipynb](Assignment/experiment-loss-functions.ipynb)  
[Kaggle Submission](https://www.kaggle.com/code/finnfrei/experiment-loss-functions?scriptVersionId=302237979)

### Hypothesis

Different loss functions impose different geometric constraints on the embedding space. Comparing angular margin losses (ArcFace, CosFace) with distance-based losses (Triplet Loss) reveals the optimal metric learning objective for jaguar re-identification.

### Methodology

1. **Sweep phase** (20 epochs each): Train each loss function with identical model and data
2. **Full training** (50 epochs): Best loss function trained to convergence
3. Triplet Loss uses larger batch size (64) for effective hard mining

### Loss Functions Compared

| Loss | Margin | Scale | Batch Size | Type |
|------|--------|-------|-----------|------|
| ArcFace (baseline) | 0.5 | 64 | 32 | Additive angular: $\cos(\theta + m)$ |
| CosFace (m=0.35) | 0.35 | 64 | 32 | Additive cosine: $\cos(\theta) - m$ |
| CosFace (m=0.40) | 0.40 | 64 | 32 | Larger cosine margin |
| Triplet (m=0.3) | 0.3 | — | 64 | Online hard mining |
| Triplet (m=0.5) | 0.5 | — | 64 | Larger triplet margin |

---

## Hyperparameter Optimization

**Notebook**: [Assignment/experiment-hyperparameter-optimization.ipynb](Assignment/experiment-hyperparameter-optimization.ipynb)  
[Kaggle Submission](https://www.kaggle.com/code/finnfrei/experiment-hyperparameter-optimization?scriptVersionId=301471696)

### Hypothesis

Systematic hyperparameter tuning across four independent categories (LR schedule, optimizer, embedding dimensions, ArcFace margin/scale) will identify a configuration that improves upon the baseline defaults.

### Methodology

1. Four independent sweeps, each varying one hyperparameter category while holding all others at baseline
2. Each configuration trained for 20 epochs (patience=7)
3. Winners from each sweep combined into best configuration
4. Best combined configuration trained for full 50 epochs

---

## k-Reciprocal Re-Ranking

**Notebook**: [Assignment/experiment-reranking.ipynb](Assignment/experiment-reranking.ipynb)  
[Kaggle Submission ID](https://www.kaggle.com/code/finnfrei/experiment-reranking?scriptVersionId=302658131)

### Hypothesis

k-Reciprocal re-ranking (Zhong et al., CVPR 2017) applied as a post-processing step on ArcFace embeddings will improve mAP by exploiting the global neighborhood structure of the embedding space.

### Methodology

1. Train identical model to baseline (MegaDescriptor → ArcFace)
2. Extract embeddings for all training + test images
3. Apply k-reciprocal re-ranking algorithm:
   - Find k1 nearest neighbors
   - Filter to mutual (reciprocal) nearest neighbors
   - Expand via local query expansion (≥ 2/3 overlap threshold)
   - Compute Jaccard distance from neighborhood encoding
   - Fuse: $d^*(p,g) = (1-\lambda) \cdot d_J(p,g) + \lambda \cdot d(p,g)$
4. Generate submission from re-ranked distances

### Configuration Changes from Baseline

| Parameter | Value |
|-----------|-------|
| rerank_k1 | 20 |
| rerank_k2 | 6 |
| rerank_lambda | 0.3 |

### Hyperparameter Sensitivity

| Parameter | Sweep Values |
|-----------|-------------|
| k1 | 5, 10, 15, 20, 25, 30 |
| k2 | 2, 4, 6, 8, 10 |
| lambda | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 |


---

## Backbone Fine-Tuning
**Notebook**: [Assignment/finetune-backbone.ipynb](Assignment/finetune-backbone.ipynb)  
[Kaggle Submission](https://www.kaggle.com/code/finnfrei/finetune-backbone?scriptVersionId=301350487)

### Hypothesis

Unfreezing the last 2 transformer blocks (Stage 3) of MegaDescriptor allows the model to learn jaguar-specific features in the deepest layers while preserving general visual features learned in earlier layers.

### Methodology

1. Unfreeze last 2 transformer blocks (Stage 3, ~61M params) + final LayerNorm of MegaDescriptor
2. Apply differential learning rates: backbone at 1e-5 (10× lower than head at 1e-4)
3. Use image-based DataLoader (images pass through backbone each epoch — no cached embeddings)
4. Reduce batch size to 16 due to increased GPU memory requirements

### Configuration Changes from Baseline

| Parameter | Baseline | Fine-Tuning |
|-----------|----------|-------------|
| Backbone | Frozen | Last 2 blocks unfrozen (~61M trainable) |
| Backbone LR | — | 1e-5 |
| Head LR | 1e-4 | 1e-4 |
| Batch Size | 32 | 16 |
| Data Loading | Cached embeddings | Image-based (through backbone) |

### MegaDescriptor Architecture

| Stage | Blocks | Params | Status |
|-------|--------|--------|--------|
| Stage 0 | 2 | ~0.9M | Frozen |
| Stage 1 | 2 | ~3.9M | Frozen |
| Stage 2 | 18 | ~129M | Frozen |
| Stage 3 | 2 | ~61M | **Unfrozen** |

