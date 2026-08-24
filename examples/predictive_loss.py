import torch

from aelia.config import PredictiveConfig
from aelia.losses import characteristic_matching_loss, predictive_nll
from aelia.mixture import GaussianMixturePredictor


def main() -> None:
    cfg = PredictiveConfig(d_model=128, target_dim=32, modes=4, characteristic_features=16)
    predictor = GaussianMixturePredictor(cfg)
    hidden = torch.randn(8, 64, 128)
    target = torch.randn(8, 64, 32)
    params = predictor(hidden)
    nll = predictive_nll(predictor, params, target)

    # Example branch supervision with M teacher continuations per prefix.
    branches = torch.randn(8, 64, 6, 32)
    cf = predictor.characteristic(params)
    cf_loss = characteristic_matching_loss(cf, branches, predictor.omega)
    print({"predictive_nll": nll.detach().item(), "characteristic_loss": cf_loss.detach().item()})


if __name__ == "__main__":
    main()
