from lite_ssl.util import STORE, DATA_TYPE

from .nico import NICO
from .mnist import M, FM, KM
from .util import make_loaders
from .coco import CocoUnlabelled
from .imagenet import ImageNet21K
from .im_simple import ImageNet, ImageNet100
from .cifar import C10, C100, C10G, C10C, C100C, C10CBar, C100CBar, STL10, IM10

STORE.register(DATA_TYPE, "nico", NICO)

STORE.register(DATA_TYPE, "m", M)
STORE.register(DATA_TYPE, "fm", FM)
STORE.register(DATA_TYPE, "km", KM)

STORE.register(DATA_TYPE, "c10", C10)
STORE.register(DATA_TYPE, "c10g", C10G)
STORE.register(DATA_TYPE, "c10c", C10C)
STORE.register(DATA_TYPE, "c10cb", C10CBar)

STORE.register(DATA_TYPE, "c100", C100)
STORE.register(DATA_TYPE, "c100c", C100C)
STORE.register(DATA_TYPE, "c100cb", C100CBar)

STORE.register(DATA_TYPE, "stl", STL10)

STORE.register(DATA_TYPE, "im10", IM10)
STORE.register(DATA_TYPE, "in", ImageNet)
STORE.register(DATA_TYPE, "in100", ImageNet100)
STORE.register(DATA_TYPE, "in21k", ImageNet21K)

STORE.register(DATA_TYPE, "coco-un", CocoUnlabelled)

__all__ = ["make_loaders"]
