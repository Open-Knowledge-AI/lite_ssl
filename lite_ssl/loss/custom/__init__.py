from .sigreg import SIGReg
from .koleo import KoLeoLoss
from .ibot import IBotPatchLoss
from .contrastive import NTXentLoss, GlobalCosineRegression
from .dino import DINOLoss, SymplecticDINOLoss, SlottedDINOLoss, EvidenceDINOLoss
from .consis import CosineRegressionLoss, EuclideanRegressionLoss, GramMatrixConsistencyLoss

__all__ = [
    "SIGReg",
    "KoLeoLoss",
    "CosineRegressionLoss",
    "EuclideanRegressionLoss",
    "GramMatrixConsistencyLoss",
    "NTXentLoss",
    "GlobalCosineRegression",
    "DINOLoss",
    "SymplecticDINOLoss",
    "SlottedDINOLoss",
    "EvidenceDINOLoss",
    "IBotPatchLoss",
]
