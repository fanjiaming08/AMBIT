"""AMBIT: curvature-informed adaptive moment estimation in a Fermi-Dirac Bregman mirror step
for density-based topology optimization."""

from .continuation import Continuation, nondiscreteness
from .model import Evaluation, TopOptModel
from .optimizers import AMBIT, MMA, OC, AdaptiveMoment, SiMPL
from .problems import Problem, benchmark, cantilever, l_bracket, mbb, michell, random_instances

__version__ = "1.0.0"

__all__ = [
    "AMBIT", "AdaptiveMoment", "Continuation", "Evaluation", "MMA", "OC", "Problem", "SiMPL",
    "TopOptModel", "benchmark", "cantilever", "l_bracket", "mbb", "michell", "nondiscreteness",
    "random_instances",
]
