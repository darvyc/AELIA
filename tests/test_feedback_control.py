import torch

from aelia.config import PredictiveConfig
from aelia.controls import SameSupervisionDeterministicControl
from aelia.feedback import DistributionSummary, PredictiveFeedback
from aelia.mixture import GaussianMixturePredictor


def test_feedback_shape_and_zero_gradient_scale_to_density():
    torch.manual_seed(0)
    predictor = GaussianMixturePredictor(
        PredictiveConfig(d_model=16, target_dim=6, modes=2, context_dim=8, feature_dim=8, mode_dim=4, characteristic_features=4)
    )
    h = torch.randn(2, 3, 16, requires_grad=True)
    params = predictor(h)
    summary_module = DistributionSummary(predictor, projected_moments=3)
    summary = summary_module(params)
    feedback = PredictiveFeedback(16, summary_module.output_dim, predictive_layers=2, gradient_scale_value=0.0)
    y = feedback(h, summary)
    loss = y.square().mean()
    loss.backward()
    # Feedback can train h directly, but predictor parameters receive no LM feedback at eta=0.
    predictor_grad = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in predictor.parameters())
    assert predictor_grad == 0.0
    assert y.shape == h.shape


def test_same_supervision_control():
    control = SameSupervisionDeterministicControl(16, 10)
    h = torch.randn(2, 4, 16)
    teacher = torch.randn(2, 4, 10)
    y, summary = control(h)
    assert y.shape == h.shape
    assert summary.shape == teacher.shape
    assert torch.isfinite(control.regression_loss(h, teacher))
