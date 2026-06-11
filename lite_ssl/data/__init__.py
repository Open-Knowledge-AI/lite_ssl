from lite_ssl.util import STORE, DATA_TYPE

from .util import make_loaders
from .im_simple import ImageNet, ImageNet100

STORE.register(DATA_TYPE, "in", ImageNet)
STORE.register(DATA_TYPE, "in100", ImageNet100)

__all__ = ["make_loaders"]
