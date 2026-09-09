"""AELIA: Adaptive Ellipsoidal Latent Inference Architecture."""

from .attention import AttentionState
from .config import AttentionConfig, MemoryConfig, ModelConfig, PredictiveConfig
from .controls import SameSupervisionDeterministicControl
from .future_features import FutureObservationAssembler, MultiscaleFutureProjector
from .mixture import GaussianMixturePredictor, MixtureParams
from .model import AELIALM, AELIAOutput
from .recurrent import ContractiveDeltaMemory, RecurrentState
from .targets import FrozenWhitening, HellingerCountSketch

__all__ = [
    "AELIALM",
    "AELIAOutput",
    "ModelConfig",
    "PredictiveConfig",
    "MemoryConfig",
    "AttentionConfig",
    "AttentionState",
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
