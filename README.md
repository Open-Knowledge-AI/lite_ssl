license: apache-2.0
tags:
  - vision
  - image-feature-extraction
  - self-supervised-learning
  - vit
  - imagenet
  - dino
  - ibot
  - LeJEPA

datasets:
  - imagenet-1k

library_name: pytorch

pipeline_tag: image-feature-extraction

---

# Lite SSL

Rapid prototyping and training of vision foundation models, specifically designed for researchers with benchmarking and logging in mind.

## ImageNet-1K Vision Transformer Collection

A unified collection of Vision Transformer (ViT) models pre-trained exclusively on **ImageNet-1K** using three self-supervised methodologies: **LeJEPA**, **DINO
**, and **iBOT
**. All models share a single codebase, training strategy, and hardware configuration, making cross-method and cross-duration comparisons directly meaningful.

---

## Model Variants

Models are named by methodology, architecture, and training duration:

| Model ID               | Method     | Arch | Epochs |
|------------------------|------------|---|---|
| `lejepa-vit-s16-ep100` | LeJEPA      | ViT-S/16 | 100 |
| `lejepa-vit-s16-ep300` | LeJEPA      | ViT-S/16 | 300 |
| `lejepa-vit-b16-ep100` | LeJEPA      | ViT-B/16 | 100 |
| `lejepa-vit-b16-ep300` | LeJEPA      | ViT-B/16 | 300 |
| `dino-vit-s16-ep100`   | DINO<br/> | ViT-S/16 | 100 |
| `dino-vit-s16-ep300`   | DINO<br/> | ViT-S/16 | 300 |
| `dino-vit-b16-ep100`   | DINO<br/> | ViT-B/16 | 100 |
| `dino-vit-b16-ep300`   | DINO<br/> | ViT-B/16 | 300 |
| `ibot-vit-s16-ep100`   | iBOT<br/> | ViT-S/16 | 100 |
| `ibot-vit-s16-ep300`   | iBOT<br/> | ViT-S/16 | 300 |
| `ibot-vit-b16-ep100`   | iBOT<br/> | ViT-B/16 | 100 |
| `ibot-vit-b16-ep300`   | iBOT<br/> | ViT-B/16 | 300 |

Intermediate checkpoints saved every 10 epochs are available under `checkpoints/ep{NNN}/` within each model repository. See [`docs/checkpoints.md`](docs/checkpoints.md) for details.

> **No registers.** All models follow the ViT-v2 (DINOv2-style) architecture **without** register tokens.

---

## Architecture

All models follow the **ViT-v2** design as used in DINOv2:
- LayerScale in every transformer block
- Stochastic depth (drop path)
- xFormers memory-efficient attention
- Patch size 16×16, input resolution 224×224

| Architecture | Params | Layers | Heads | Hidden dim |
|---|---|---|---|---|
| ViT-S/16 | 22M | 12 | 6 | 384 |
| ViT-B/16 | 86M | 12 | 12 | 768 |

---

## Training Setup

All models were trained with an identical hardware and software configuration via PyTorch Lightning:

| Setting | Value |
|---|---|
| Dataset | ImageNet-1K (1.28M images, 1000 classes) |
| Precision | BF16 mixed (`bf16-mixed`) |
| GPUs | 8× (4 devices × 2 nodes, DDP) |
| Global batch size | 1024 |
| Per-GPU batch size | 128 |
| Optimizer | AdamW, layerwise LR decay 0.9 |
| Base LR | 5e-4 (scaled: `base_lr × batch_size / 256 = 2e-3`) |
| LR schedule | Linear warmup (10 epochs) → cosine decay to 1e-6 |
| Gradient clipping | norm, max 3.0 |
| Training durations | 100 epochs, 300 epochs |
| Checkpoint cadence | Every 10 epochs |
| Registers | None |

Exact configs are in the `configs/` directory. See [`docs/training_infra.md`](docs/training_infra.md) for reproducibility notes.

---

## Methods

### LeJEPA

A latent-space JEPA (Joint Embedding Predictive Architecture) adapted for ViT backbones. The model learns to predict the representations of masked image regions from context in latent space — no pixel reconstruction. Uses a context encoder and a target encoder; no EMA teacher. Weight decay is held constant (`ConstSched(5e-2)`) as prescribed by the method.

- `trainer: "ms_lj"`, `loss.id: "lj"`
- No teacher momentum or temperature schedules

### DINO + KoLeo

DINO trains a student network to match the output distribution of a momentum teacher. The KoLeo regulariser promotes uniform spreading of representations across the embedding hypersphere, reducing collapse.

- `trainer: "dino"`, `loss.id: "dino"`
- Teacher momentum: `CosSched(0.994, 1.0)`
- Teacher temperature: linear warmup `0.04 → 0.07` (10 epochs for ep100, 30 for ep300)
- Weight decay: cosine anneal `0.04 → 0.4`

### iBOT + KoLeo

iBOT combines a masked image modelling objective in token space with a DINO-style global CLS objective, using the momentum teacher itself as the online tokenizer. KoLeo regularisation is applied to the global CLS embeddings.

- `trainer: "dinobot"`, `loss.id: "db"`
- `do_mask: true`, `project_patch: true`
- Teacher checkpoints saved periodically via `pl.periodic` (prefix `teacher_model`)
- Teacher momentum and temperature schedules identical to DINO

---

## Repository Structure

```
.
├── README.md
├── configs/
│   ├── lejepa/
│   │   ├── vit_s16_ep100.json
│   │   ├── vit_s16_ep300.json
│   │   ├── vit_b16_ep100.json
│   │   └── vit_b16_ep300.json
│   ├── dino_koleo/
│   │   ├── vit_s16_ep100.json
│   │   ├── vit_s16_ep300.json
│   │   ├── vit_b16_ep100.json
│   │   └── vit_b16_ep300.json
│   └── ibot_koleo/
│       ├── vit_s16_ep100.json
│       ├── vit_s16_ep300.json
│       ├── vit_b16_ep100.json
│       └── vit_b16_ep300.json
└── docs/
    ├── checkpoints.md
    └── training_infra.md
```

---

## Usage

### Loading a model

```python
import torch
from transformers import AutoModel, AutoImageProcessor

model_id = "your-org/dino-vit-b16-ep300"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()
```

### Extracting features

```python
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# CLS token
cls_features = outputs.last_hidden_state[:, 0, :]       # (1, hidden_dim)
# Patch tokens
patch_features = outputs.last_hidden_state[:, 1:, :]    # (1, num_patches, hidden_dim)
```

### Loading an intermediate checkpoint

```python
model = AutoModel.from_pretrained(
    "your-org/dino-vit-b16-ep300",
    subfolder="checkpoints/ep100",
    torch_dtype=torch.bfloat16,
)
```

See [`docs/checkpoints.md`](docs/checkpoints.md) for the full checkpoint inventory and loading patterns.

---

## Citation

If you use these models, please cite the corresponding method papers:

```bibtex
@article{Balestriero2025Nov,
	author = {Balestriero, Randall and LeCun, Yann},
	title = {{LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}},
	year = {2025},
	month = nov,
	doi = {10.48550/arXiv.2511.08544}
}

@article{Caron2021Apr,
	author = {Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\ifmmode\acute{e}\else\'{e}\fi}gou, Herv{\ifmmode\acute{e}\else\'{e}\fi} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
	title = {{Emerging Properties in Self-Supervised Vision Transformers}},
	year = {2021},
	month = apr,
	doi = {10.48550/arXiv.2104.14294}
}

@article{Zhou2021Nov,
	author = {Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
	title = {{iBOT: Image BERT Pre-Training with Online Tokenizer}},
	year = {2021},
	month = nov,
	doi = {10.48550/arXiv.2111.07832}
}
```

---

## License

All model weights and configs are released under the [Apache 2.0 License](LICENSE).
