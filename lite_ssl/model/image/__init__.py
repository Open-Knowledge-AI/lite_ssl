from .resnet.cifar import ResNet18, ResNeXt29
from .resnet.imagenet import resnet18, resnet50
from .vitv2 import (
    vitv2_tiny,
    vitv2_small,
    vitv2_base,
    vitv2_large,
)
from .vitv3 import (
    vitv3_tiny,
    vitv3_small,
    vitv3_base,
    vitv3_large,
)

from lite_ssl.util import MODEL_TYPE, STORE

STORE.register(MODEL_TYPE, "rn18", ResNet18)
STORE.register(MODEL_TYPE, "rnxt29", ResNeXt29)
STORE.register(MODEL_TYPE, "rn18im", resnet18)
STORE.register(MODEL_TYPE, "rn50im", resnet50)
STORE.register(MODEL_TYPE, "vitv2_t", vitv2_tiny)
STORE.register(MODEL_TYPE, "vitv2_s", vitv2_small)
STORE.register(MODEL_TYPE, "vitv2_b", vitv2_base)
STORE.register(MODEL_TYPE, "vitv2_l", vitv2_large)
STORE.register(MODEL_TYPE, "vitv3_t", vitv3_tiny)
STORE.register(MODEL_TYPE, "vitv3_s", vitv3_small)
STORE.register(MODEL_TYPE, "vitv3_b", vitv3_base)
STORE.register(MODEL_TYPE, "vitv3_l", vitv3_large)

__all__ = [
    "ResNet18",
    "ResNeXt29",
    "resnet18",
    "resnet50",
    "vitv2_tiny",
    "vitv2_small",
    "vitv2_base",
    "vitv2_large",
    "vitv3_tiny",
    "vitv3_small",
    "vitv3_base",
    "vitv3_large",
]
