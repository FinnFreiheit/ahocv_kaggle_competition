# Assignment Report — Jaguar Re-Identification

**Course**: Applied Hands-on Computer Vision  
**Student**: Finn Freiheit 
**Competition**: [Jaguar Re-Identification](https://www.kaggle.com/competitions/jaguar-re-id/)  
**GitHub Repository**: [INSERT GITHUB REPO LINK]  
[**W&B Project**](https://wandb.ai/finnfreiheit/Jaguar-Re-identification-Challenge?nw=nwuserfinnfrei)

---

## 1. Exploratory Data Analysis

The jaguar re-identification dataset contains approximately 2,401 camera trap images spanning 58 unique jaguar identities. A key challenge is the severe class imbalance: the most frequent identity has ~191 images while the least frequent has only ~4 — a 47:1 ratio. Despite this, the identity-balanced mAP metric ensures that all identities contribute equally to evaluation, making performance on rare identities critical.

**Embedding space analysis** using Multidimensional Scaling (MDS) on geodesic distances revealed that frozen MegaDescriptor-L-384 embeddings already exhibit partial identity clustering, confirming the value of wildlife-specific pre-training. After ArcFace fine-tuning, clusters become significantly tighter and more separated, validating the angular margin approach for this task.

I conducted a **backbone architecture comparison**, evaluating MegaDescriptor-L-384, DINOv2-B-14, ConvNeXt V2 Base, and EfficientNet-B4 under identical conditions (frozen backbone, same projection head and ArcFace loss). MDS visualizations of raw embeddings revealed qualitative differences in how each backbone structures the feature space before fine-tuning.

A **loss function study** compared ArcFace, CosFace (two margin settings), and Triplet Loss (two margin settings) to understand how different geometric constraints shape embedding distributions. Angular margin losses (ArcFace, CosFace) provide gradient signal from all classes, while Triplet Loss relies on online hard mining within each batch.

**Hyperparameter sensitivity analysis** across four dimensions — learning rate schedules, optimizers, embedding dimensions, and ArcFace margin/scale — quantified which parameters the model is most sensitive to. This structured sweep approach identified the optimal combination independent of architectural choices.

I also analyzed the **augmentation strategy**: pre-augmentation (augmenting images before embedding extraction) effectively expands the training set by 4× while using a foreground-only random erasing technique that preserves dataset structure. The **re-ranking neighborhood analysis** examined how k-reciprocal re-ranking modifies the distance relationships between embeddings, revealing the structure and reliability of nearest-neighbor relationships in the learned space.

---

## 2. Model Training and Evaluation

All experiments share a common pipeline: frozen MegaDescriptor-L-384 extracts 1536-dim embeddings, an EmbeddingProjection (1536 → 512 → 256) with BatchNorm, ReLU, and Dropout(0.3) projects them, and an ArcFace head (margin=0.5, scale=64) trains the classification objective. Training uses AdamW (lr=1e-4), ReduceLROnPlateau scheduling, and early stopping (patience=10) on a stratified 80/20 split (seed=42).

The **baseline** achieves 0.741 mAP on the public Kaggle leaderboard. **Pre-augmentation** with 3 augmented copies per image (RandomResizedCrop, ColorJitter, GaussianBlur, foreground-only erasing) improved mAP to **0.7878 (+6.3%)**, demonstrating that increasing training diversity before embedding extraction is highly effective.

The **backbone comparison** evaluated whether general-purpose architectures (DINOv2, ConvNeXt V2, EfficientNet-B4) can compete with MegaDescriptor's wildlife-specific representations under identical training conditions. The **loss function comparison** tested whether alternative margin losses (CosFace) or distance-based losses (Triplet) yield better identity separation than the baseline ArcFace.

**Hyperparameter optimization** conducted four independent sweeps — LR schedules (ReduceLROnPlateau, CosineAnnealing, StepLR, OneCycleLR), optimizers (AdamW, Adam, SGD, Muon), embedding dimensions (5 configurations from 128 to 512), and ArcFace parameters (8 margin/scale combinations). Sweep winners were combined into a final best configuration.

**k-Reciprocal re-ranking** (Zhong et al., CVPR 2017) was applied as a post-processing step, exploiting the global neighborhood structure of the embedding space without additional training. The algorithm computes Jaccard distances from reciprocal nearest neighbor sets and fuses them with original cosine distances. Sensitivity analysis across k1, k2, and lambda parameters quantified the robustness of this approach.

**Backbone fine-tuning** unfroze the last 2 transformer blocks (~61M parameters) of MegaDescriptor with differential learning rates (backbone: 1e-5, head: 1e-4), testing whether task-specific adaptation of the deepest layers improves jaguar-specific feature extraction while preserving general visual knowledge.

