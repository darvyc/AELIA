import torch

from aelia.future_features import FutureObservationAssembler, MultiscaleFutureProjector


def test_future_feature_pipeline_shapes():
    assembler = FutureObservationAssembler()
    semantic = torch.randn(2, 7, 4)
    sketch = torch.randn(2, 7, 6)
    entropy = torch.randn(2, 7)
    margin = torch.randn(2, 7)
    surprisal = torch.randn(2, 7)
    obs = assembler(semantic, sketch, entropy, margin, surprisal)
    assert obs.shape == (2, 7, 13)

    projector = MultiscaleFutureProjector(13, scales=((0,), (1, 2), (3, 4, 5, 6)), output_dims=(8, 10, 12))
    z = projector(obs)
    assert z.shape == (2, 30)
    assert torch.isfinite(z).all()
