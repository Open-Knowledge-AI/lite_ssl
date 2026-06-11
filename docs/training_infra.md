# Training Infrastructure & Reproducibility

This document describes the hardware setup, software environment, and decisions made to ensure reproducible training across all models in this collection.

---

## Hardware

All models were trained on a single node with 8 GPUs. The global batch size of 1024 was achieved with 128 samples per GPU — no gradient accumulation was used.

| Component | Specification            |
|---|--------------------------|
| GPUs | 8× (single node)         |
| Per-GPU batch size | 128                      |
| Global batch size | 1024                     |
| Inter-GPU communication | NCCL                     |
| Precision | FP16-Mixed (PyTorch AMP) |

---

## Software Environment

| Package     | Role |
|-------------|---|
| PyTorch     | Core framework |
| xFormers    | Memory-efficient attention (Flash Attention-style) |
| torchvision | ImageNet-1K data loading and augmentation |
| sbatch      | SLURM job scheduling |

---

## Precision Strategy

Training used **BF16 mixed precision**:
- via PyTorch Lightning

---

## Batch Size and Learning Rate Scaling

The learning rate is specified as a `base_lr` in each config and is intended for the reference batch size of 256. If you reproduce training with a different batch size, apply linear scaling:

```
effective_lr = base_lr × (global_batch_size / 256)
```

No square-root scaling is used.

---

## Reproducibility

]- Drop path masks are generated per-forward-pass and are not seeded beyond the global seed.
- Data augmentation uses `torchvision.transforms` with a fixed random state per worker.
- The same dataset split (ImageNet-1K `train`) was used for all pre-training runs; no held-out subset was used during pre-training.

---

## What Is and Is Not Controlled

**Controlled (identical across all runs):**

- Dataset, split, and augmentation pipeline
- Hardware topology (8 GPUs, single node)
- Batch size (global and per-GPU)
- Precision (BF16-mixed)
- Architecture family (ViT-v2 without registers, without layer scale and without dropath)
- Checkpoint cadence (every 10 epochs)

**Not controlled (varies by method):**

- Loss function and projection head parameters
- Warmup length, and weight decay schedule (exists only for DINO-like methods and not LeJEPA; see individual configs)
- EMA momentum schedule (same parameters, but only exists for DINO-like methods and not LeJEPA)

---

## Known Limitations

- Models were pre-trained exclusively on ImageNet-1K. They have not been exposed to web-scale data and may underperform DINOv2/LVD-142M-pretrained checkpoints on out-of-distribution tasks.
- No register tokens are used. For tasks sensitive to attention map quality on high-resolution inputs, consider fine-tuning with registers added. For "small"-scale dataset like ImageNet-1k this is fine, but when scaling to webdatasets this might be needed.
- BF16-mixed pre-training may introduce minor numerical differences compared to FP32, particularly in loss scaling edge cases.
