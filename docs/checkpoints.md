# Checkpoint Guide

All model repositories include intermediate checkpoints saved every 10 epochs throughout training, in addition to the final checkpoint.

---

## Naming Convention

Checkpoints are stored in subfolders named `checkpoints/ep{NNN}/` where `NNN` is the zero-padded epoch number:

```
checkpoints/
├── ep010/
│   ├── pytorch_model.bin
│   └── config.json
├── ep020/
│   ├── pytorch_model.bin
│   └── config.json
...
├── ep090/          # last intermediate for 100-epoch models
│   └── ...
├── ep100/          # = final for 100-epoch models (also at root)
│   └── ...
...
├── ep290/          # last intermediate for 300-epoch models
│   └── ...
└── ep300/          # = final for 300-epoch models (also at root)
    └── ...
```

The root-level `pytorch_model.bin` and `config.json` always correspond to the **final** trained checkpoint (epoch 100 or 300 depending on the model).

---

## Available Checkpoints Per Model

| Training duration | Intermediate checkpoints | Final checkpoint |
|---|---|---|
| 100 epochs | ep010, ep020, …, ep090 (9 checkpoints) | ep100 |
| 300 epochs | ep010, ep020, …, ep290 (29 checkpoints) | ep300 |

---

## Loading Checkpoints

### Final checkpoint (default)

```python
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained("your-org/dino-koleo-vit-b16-ep300")
processor = AutoImageProcessor.from_pretrained("your-org/dino-koleo-vit-b16-ep300")
```

### Specific intermediate checkpoint

Use the `subfolder` argument to point at a `checkpoints/ep{NNN}/` subdirectory:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/dino-koleo-vit-b16-ep300",
    subfolder="checkpoints/ep150",
)
```

### Iterating over all checkpoints (e.g. for a training curve analysis)

```python
from transformers import AutoModel
import torch

model_id = "your-org/dino-koleo-vit-b16-ep300"
checkpoints = [f"checkpoints/ep{e:03d}" for e in range(10, 300, 10)]

for ckpt in checkpoints:
    model = AutoModel.from_pretrained(model_id, subfolder=ckpt)
    # ... evaluate
```

---

## Notes

- All checkpoints (including intermediates) are saved in FP16 to match training precision.
- The `config.json` at each checkpoint subfolder is identical to the root `config.json` — it is duplicated for convenience so each subfolder is self-contained.
- EMA / teacher weights are part of the same ckpt.
