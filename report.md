# Jaguar Re-Identification — Written Report

**Student**: Finn Freiheit  
**Course**: Applied Hands-on Computer Vision  
**W&B Project**: [jaguar-re-identification-challenge](https://wandb.ai/finnfreiheit/Jaguar-Re-identification-Challenge?nw=nwuserfinnfrei)  
**Competition**: [Kaggle — Jaguar Re-ID](https://www.kaggle.com/competitions/jaguar-re-id/)

---

## Exploratory Data Analysis

The training dataset contains 1,516 images across multiple jaguar identities captured by camera traps. During initial exploration, key observations emerged: the class distribution is imbalanced, with some individuals having significantly more samples than others, motivating the use of stratified splits and identity-balanced mAP as the evaluation metric. All images share a consistent white background, which informed the design of foreground-only random erasing in the pre-augmentation experiment — standard random erasing would destroy the uniform background and introduce artifacts, so erasing was restricted to the animal body region (minimum 60% foreground fraction). Embedding space analysis via MegaDescriptor showed that baseline embeddings already cluster by identity, but several hard cases exist where visually similar rosette patterns cause inter-identity confusion; this motivated experiments with re-ranking to exploit global neighborhood structure.

---

## Model Training and Evaluation

Six experiments were conducted, all building on a frozen MegaDescriptor-L-384 backbone (~305M params) with an ArcFace projection head, trained on cached embeddings.

| # | Experiment | Key Change | Notebook |
|---|-----------|-----------|----------|
| 1 | Pre-Augmentation | 4× training set via augmented embeddings | [experiment-preaugmentation.ipynb](Assignment/experiment-preaugmentation.ipynb) |
| 2 | Backbone Comparison | DINOv2, ConvNeXt V2, EfficientNet-B4 vs MegaDescriptor | [experiment-backbone-comparison.ipynb](Assignment/experiment-backbone-comparison.ipynb) |
| 3 | Loss Functions | ArcFace vs CosFace vs Triplet Loss | [experiment-loss-functions.ipynb](Assignment/experiment-loss-functions.ipynb) |
| 4 | Hyperparameter Optimization | LR schedule, optimizer, embedding dim, ArcFace margin/scale sweeps | [experiment-hyperparameter-optimization.ipynb](Assignment/experiment-hyperparameter-optimization.ipynb) |
| 5 | k-Reciprocal Re-Ranking | Post-processing with Jaccard distance fusion | [experiment-reranking.ipynb](Assignment/experiment-reranking.ipynb) |
| 6 | Backbone Fine-Tuning | Unfreeze last 2 transformer blocks with differential LR | [finetune-backbone.ipynb](Assignment/finetune-backbone.ipynb) |

**Pre-Augmentation** expanded the training set from 1,516 to 6,064 samples by generating 3 augmented copies per image before embedding extraction. Foreground-only random erasing preserved dataset structure while simulating partial occlusion, yielding +6.3% mAP improvement.

**Backbone Comparison** evaluated MegaDescriptor-L-384, DINOv2-B-14, ConvNeXt V2 Base, and EfficientNet-B4 under identical training conditions. MegaDescriptor's wildlife-specific pre-training proved advantageous over general-purpose models for this domain.

**Loss Function Comparison** tested angular margin losses (ArcFace, CosFace at margins 0.35 and 0.40) against distance-based Triplet Loss (margins 0.3, 0.5 with online hard mining at batch size 64).

**Hyperparameter Optimization** ran four independent sweeps (LR schedule, optimizer, embedding dimensions, ArcFace margin/scale), each varying one category while holding others at baseline, then combined winners for full training.

**k-Reciprocal Re-Ranking** applied post-processing on learned embeddings using mutual nearest neighbor filtering, local query expansion, and Jaccard–cosine distance fusion ($d^* = (1-\lambda) \cdot d_J + \lambda \cdot d$), with sensitivity analysis over k1, k2, and λ.

**Backbone Fine-Tuning** unfroze MegaDescriptor's last 2 transformer blocks (~61M params) with a 10× lower learning rate (1e-5 vs 1e-4 for the head), switching from cached embeddings to image-based data loading.

All experiments are tracked in Weights & Biases with hyperparameters, training/validation metrics, and model parameter counts logged. Detailed methodology, configurations, and results are documented in [LEADERBOARD_EXPERIMENTS.md](LEADERBOARD_EXPERIMENTS.md).
