"""Package for pytorch NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "FNN",
    "MIONetCartesianProd",
    "MIONetCartesianProd_V2",
    "MIONetCartesianProd_3Branches",
    "NN",
    "PFNN",
    "PODDeepONet",
    "PODMIONet",
]

from .deeponet import DeepONet, DeepONetCartesianProd, PODDeepONet
from .mionet import MIONetCartesianProd, PODMIONet, MIONetCartesianProd_V2, MIONetCartesianProd_3Branches
from .fnn import FNN, PFNN
from .nn import NN
