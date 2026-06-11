from .dino import DINOAugmentation

from lite_ssl.util import AUG_TYPE, STORE

STORE.register(AUG_TYPE, "mst", DINOAugmentation)

__all__ = ["DINOAugmentation"]
