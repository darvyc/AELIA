"""AELIA: Adaptive Ellipsoidal Latent Inference Architecture."""

from .config import ModelConfig, PredictiveConfig, MemoryConfig, AttentionConfig
from .model import AELIALM, AELIAOutput
from .mixture import GaussianMixturePredictor, MixtureParams
from .recurrent import ContractiveDeltaMemory, RecurrentState
from .targets import HellingerCountSketch, FrozenWhitening
from .controls import SameSupervisionDeterministicControl
from .future_features import FutureObservationAssembler, MultiscaleFutureProjector

__all__ = [
    "AELIALM",
    "AELIAOutput",
    "ModelConfig",
    "PredictiveConfig",
    "MemoryConfig",
    "AttentionConfig",
    "GaussianMixturePredictor",
    "MixtureParams",
    "ContractiveDeltaMemory",
    "RecurrentState",
    "HellingerCountSketch",
    "FrozenWhitening",
    "SameSupervisionDeterministicControl",
    "FutureObservationAssembler",
    "MultiscaleFutureProjector",
]

__version__ = "0.1.0"
