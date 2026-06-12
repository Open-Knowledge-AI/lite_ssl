# Lite SSL

Rapid prototyping and training of vision foundation models, specifically designed for researchers with benchmarking and logging in mind.

## ImageNet-1K Vision Transformer Collection

A unified collection of Vision Transformer (ViT) models pre-trained exclusively on **ImageNet-1K** using three self-supervised methodologies: **LeJEPA**, **DINO**, and **iBOT**. All models share a single codebase, training strategy, and hardware configuration, making cross-method and cross-duration comparisons directly meaningful.

---

## Model Variants

Models are named by methodology, architecture, and pretraining data:

| Model ID                       | Method    | Arch     | Epochs    |
|--------------------------------|-----------|----------|-----------|
| `lejepa-vits16-pretrain-in1k`  | LeJEPA    | ViT-S/16 | 100 / 300 |
| `lejepa-vitb16-pretrain-in1k`  | LeJEPA    | ViT-B/16 | 100 / 300 |
| `dino-vits16-pretrain-in1k`    | DINO<br/> | ViT-S/16 | 100 / 300 |
| `dino-vitb16-pretrain-in1k`    | DINO<br/> | ViT-B/16 | 100 / 300 |
| `ibot-vits16-pretrain-in1k`    | iBOT<br/> | ViT-S/16 | 100 / 300 |
| `ibot-vitb16-pretrain-in1k`    | iBOT<br/> | ViT-B/16 | 100 / 300 |

We first test model scale effects, small to large, to detemrine if new methods scale with size.
Similarly, every model has two checkpoint variants, 100 and 300 epochs to determine how well methods scale with training duration.
We plan to expand this analysis with also data scale, but currently only supply model checkpoints for in1k pretrained models.

> **No registers.** All models follow the ViT-v2 (DINOv2-style) architecture **without** register tokens.

---

## Architecture

All models follow the **ViT-v2** design as used in DINOv2:
- LayerScale in every transformer block
- Stochastic depth (drop path)
- xFormers memory-efficient attention
- Patch size 16×16, input resolution 224×224

| Architecture | Params | Layers | Heads | Hidden dim |
|--------------|--------|--------|-------|------------|
| ViT-S/16     | 22M    | 12     | 6     | 384        |
| ViT-B/16     | 86M    | 12     | 12    | 768        |

---

## Training Setup

All models were trained with an identical hardware and software configuration via PyTorch Lightning:

| Setting           | Value                                                                                              |
|-------------------|----------------------------------------------------------------------------------------------------|
| Dataset           | ImageNet-1K (1.45M images)                                                                         |
| Architecture      | ViT-S/16 (ViT-v2, no registers)                                                                    |
| Epochs            | 100,300                                                                                            |
| Views             | 8 total — 2 global (224×224) + 6 local (96×96)                                                     |
| Precision         | BF16 mixed (`bf16-mixed`)                                                                          |
| GPUs              | 8× (4 devices × 2 nodes, DDP via PyTorch Lightning)                                                |
| Global batch size | 1024 (128 per GPU)                                                                                 |
| Optimizer         | AdamW, layerwise LR decay 0.9, patch embed LR mult 0.2 (from DiNOv2)                               |
| Learning rate     | Base 5e-4 → effective 2e-3 (scaled by batch / 256); linear warmup 10 epochs → cosine decay to 1e-6 |
| Weight decay      | Cosine anneal 0.04 → 0.4 (for DiNO and iBOT), Constant 5e-2 for LeJEPA                             |
| Gradient clipping | Norm, max 3.0                                                                                      |

Exact configs are in the `configs/` directory. See [`docs/training_infra.md`](docs/training_infra.md) for reproducibility notes.

---

## Methods

### DINO

DINO trains a student network to match the output distribution of a momentum teacher. The KoLeo regulariser promotes uniform spreading of representations across the embedding hypersphere, reducing collapse.

- `trainer: "dino"`, `loss.id: "dino"`
- Teacher momentum: `CosSched(0.994, 1.0)`
- Teacher temperature: linear warmup `0.04 → 0.07` (10 epochs for ep100, 30 for ep300)
- Weight decay: cosine anneal `0.04 → 0.4`

### iBOT

iBOT combines a masked image modelling objective in token space with a DINO-style global CLS objective, using the momentum teacher itself as the online tokenizer. KoLeo regularisation is applied to the global CLS embeddings.

- `trainer: "dinobot"`, `loss.id: "db"`
- `do_mask: true`, `project_patch: true`
- Teacher checkpoints saved periodically via `pl.periodic` (prefix `teacher_model`)
- Teacher momentum and temperature schedules identical to DINO

### LeJEPA

A latent-space JEPA (Joint Embedding Predictive Architecture) adapted for ViT backbones. The model learns to predict the representations of masked image regions from context in latent space — no pixel reconstruction. Uses a context encoder and a target encoder; no EMA teacher. Weight decay is held constant (`ConstSched(5e-2)`) as prescribed by the method.

- `trainer: "ms_lj"`, `loss.id: "lj"`
- No teacher momentum or temperature schedules

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
│   ├── dino/
│   │   ├── vit_s16_ep100.json
│   │   ├── vit_s16_ep300.json
│   │   ├── vit_b16_ep100.json
│   │   └── vit_b16_ep300.json
│   └── ibot/
│       ├── vit_s16_ep100.json
│       ├── vit_s16_ep300.json
│       ├── vit_b16_ep100.json
│       └── vit_b16_ep300.json
└── docs/
    ├── checkpoints.md
    └── training_infra.md
```

---

## Training

| Setting           | Value                                                                                              |
|-------------------|----------------------------------------------------------------------------------------------------|
| Dataset           | ImageNet-1K (1.45M images)                                                                         |
| Architecture      | ViT-S/16 (ViT-v2, no registers)                                                                    |
| Epochs            | 100,300                                                                                            |
| Views             | 8 total — 2 global (224×224) + 6 local (96×96)                                                     |
| Precision         | BF16 mixed (`bf16-mixed`)                                                                          |
| GPUs              | 8× (4 devices × 2 nodes, DDP via PyTorch Lightning)                                                |
| Global batch size | 1024 (128 per GPU)                                                                                 |
| Optimizer         | AdamW, layerwise LR decay 0.9, patch embed LR mult 0.2 (from DiNOv2)                               |
| Learning rate     | Base 5e-4 → effective 2e-3 (scaled by batch / 256); linear warmup 10 epochs → cosine decay to 1e-6 |
| Weight decay      | Cosine anneal 0.04 → 0.4 (for DiNO and iBOT), Constant 5e-2 for LeJEPA                             |
| Gradient clipping | Norm, max 3.0                                                                                      |

The exact config used for this run is available at `configs/{dino,ibot,lejepa}/vit_{s,b}16_ep{100,300}.json` in the [code repository](https://github.com/Open-Knowledge-AI/lite_ssl).

---

## Evaluation

Metrics are only available for the teacher weights.

| Model                 | IN-1K online probe (acc@1) | IN-1K linear probe (acc@1) | IN-1K k-NN (acc@1) | NYU Depth (δ1) | Pascal VOC (mAP) |
|-----------------------|----------------------------|----------------------------|--------------------|----------------|------------------|
| DINO ViT-S/16 ep100   | 69.32                      | -                          | -                  | -              | -                |
| DINO ViT-S/16 ep300   | 73.88                      | -                          | -                  | -              | -                |
| DINO ViT-B/16 ep100   | 73.49                      | -                          | -                  | -              | -                |
| DINO ViT-B/16 ep300   | Soon                       | -                          | -                  | -              | -                |
| iBOT ViT-S/16 ep100   | 69.70                      | -                          | -                  | -              | -                |
| iBOT ViT-S/16 ep300   | 74.32                      | -                          | -                  | -              | -                |
| iBOT ViT-B/16 ep100   | 76.50                      | -                          | -                  | -              | -                |
| iBOT ViT-B/16 ep300   | 78.74                      | -                          | -                  | -              | -                |
| LeJEPA ViT-S/16 ep100 | 61.85                      | -                          | -                  | -              | -                |
| LeJEPA ViT-S/16 ep300 | 65.99                      | -                          | -                  | -              | -                |
| LeJEPA ViT-B/16 ep100 | 69.28                      | -                          | -                  | -              | -                |
| LeJEPA ViT-B/16 ep300 | 72.04                      | -                          | -                  | -              | -                |

Grouped by method, DINO first since you have results for it. Let me know if you'd prefer a different ordering (e.g. by architecture, or alphabetical by method).
> Online probe results are logged during pre-training. Linear probe, k-NN, and downstream evaluations are coming soon.

---

## Checkpoints

```python
from transformers import AutoModel

# default checkpoint (teacher checkpoint after 300 epochs)
objective = "dino"  # lejepa, ibot
model_size = "s"  # b
pretrain_dataset = "in1k"  # currently only in1k is planned unless compute can be expanded.

hf_model_string = f"OK-AI/{objective}-vit{model_size}16-pretrain-{pretrain_dataset}"
model = AutoModel.from_pretrained(hf_model_string)

# alternate training checkpoints
epoch_variant = 100  # 300
state_dict_of = "student"  # `teacher` is default for dino and ibot, lejepa only has student.
model = AutoModel.from_pretrained(
    hf_model_string,
    revision=f"ep{epoch_variant}/{state_dict_of}",
)
```

---

## Usage

```python
import requests

import torch

from PIL import Image

from transformers import AutoModel, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("OK-AI/dino-vits16-pretrain-in1k")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
pil_input = Image.open(requests.get(url, stream=True).raw)

preprocessed = processor(pil_input, return_tensors="pt")

model_input = preprocessed.data["pixel_values"]
print(f"Preprocessed Input Shape: {model_input.shape}")

model = AutoModel.from_pretrained(
    "OK-AI/dino-vits16-pretrain-in1k",
    # revision="ep100/teacher",
    trust_remote_code=True,
)
outputs = model(model_input)
print(f"Output Keys: {outputs.keys()}")
print(f"{outputs["latent"].shape=}, {outputs["patch_latent"].shape=}")

# {
#     "latent": cls_tokens[:, 0],
#     "patch_latent": patch_tokens,
#     "raw_latent": x[:, 0],
#     "last_self_attention": attn,
#     "logits": self.head(cls_tokens[:, 0]),  # only exists for comptability, head is always identity in this case.
# }

# CLS token — use for classification, retrieval, k-NN
cls = outputs["latent"]        # (1, 384)

# Patch tokens — use for dense tasks (depth, segmentation)
patches = outputs["patch_latent"]   # (1, 196, 384)
```

---

## HuggingFace Spaces

### ViT Patch PCA Visualisation

Explore and compare patch-token representations learned by our self-supervised Vision Transformers:

🔗 **Space:** https://huggingface.co/spaces/OK-AI/ViT-Patch-PCA-Visualisation

This interactive demo projects ViT patch embeddings into a low-dimensional PCA space, allowing qualitative inspection of how different self-supervised objectives organise visual information. The Space supports models trained with:

- **DINO**
- **iBOT**
- **LeJEPA**

### Features

- Upload your own images for analysis.
- Visualise patch-token embeddings using PCA.
- Compare representation structure across methods, model sizes, and training durations.
- Inspect spatial organisation of learned features without task-specific supervision.

The Space is intended as a lightweight representation-analysis tool for understanding how self-supervised ViTs encode image content beyond standard downstream benchmarks.

---

## Citation

```bibtex
@article{Sablayrolles2018Jun,
	author = {Sablayrolles, Alexandre and Douze, Matthijs and Schmid, Cordelia and J{\ifmmode\acute{e}\else\'{e}\fi}gou, Herv{\ifmmode\acute{e}\else\'{e}\fi}},
	title = {{Spreading vectors for similarity search}},
	year = {2018},
	month = jun,
	doi = {10.48550/arXiv.1806.03198}
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

@article{Dong2022Dec,
	author = {Dong, Xiaoyi and Bao, Jianmin and Zhang, Ting and Chen, Dongdong and Gu, Shuyang and Zhang, Weiming and Yuan, Lu and Chen, Dong and Wen, Fang and Yu, Nenghai},
	title = {{CLIP Itself is a Strong Fine-tuner: Achieving 85.7{\%} and 88.0{\%} Top-1 Accuracy with ViT-B and ViT-L on ImageNet}},
	year = {2022},
	month = dec,
	doi = {10.48550/arXiv.2212.06138}
}

@article{Oquab2023Apr,
	author = {Oquab, Maxime and Darcet, Timoth{\ifmmode\acute{e}\else\'{e}\fi}e and Moutakanni, Th{\ifmmode\acute{e}\else\'{e}\fi}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Assran, Mahmoud and Ballas, Nicolas and Galuba, Wojciech and Howes, Russell and Huang, Po-Yao and Li, Shang-Wen and Misra, Ishan and Rabbat, Michael and Sharma, Vasu and Synnaeve, Gabriel and Xu, Hu and Jegou, Herv{\ifmmode\acute{e}\else\'{e}\fi} and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
	title = {{DINOv2: Learning Robust Visual Features without Supervision}},
	year = {2023},
	month = apr,
	doi = {10.48550/arXiv.2304.07193}
}

@article{Balestriero2025Nov,
	author = {Balestriero, Randall and LeCun, Yann},
	title = {{LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}},
	year = {2025},
	month = nov,
	doi = {10.48550/arXiv.2511.08544}
}
```


---

## License

All model weights and configs are released under the [Apache 2.0 License](LICENSE).
