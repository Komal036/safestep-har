import numpy as np
import pytest
import torch

from src.augment import Augmenter, random_rotation, xyz_groups
from src.data.uci_har import CHANNELS
from src.models import REGISTRY, build_model, count_params


@pytest.mark.parametrize("name", list(REGISTRY))
def test_forward_shape(name):
    model = build_model(name, n_channels=9, n_classes=6, seq_len=128)
    out = model(torch.randn(4, 128, 9))
    assert out.shape == (4, 6)
    assert count_params(model) > 0


@pytest.mark.parametrize("name", list(REGISTRY))
def test_backward(name):
    model = build_model(name, 3, 2, 64)
    loss = model(torch.randn(2, 64, 3)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_transformer_rejects_bad_patch():
    with pytest.raises(ValueError):
        build_model("transformer", 9, 6, seq_len=100, patch=8)


def test_rotation_is_orthonormal():
    R = random_rotation(np.random.default_rng(0), 30)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_xyz_groups_uci():
    assert len(xyz_groups(CHANNELS)) == 3


def test_augmenter_preserves_shape_and_norm():
    X = np.random.default_rng(1).normal(size=(8, 128, 9)).astype(np.float32)
    aug = Augmenter(CHANNELS, jitter_std=0, scale_std=0, rotate_deg=30, p=1.0)
    Xa = aug(X)
    assert Xa.shape == X.shape
    # pure rotation keeps per-sample vector magnitude for each sensor triplet
    g = xyz_groups(CHANNELS)[0]
    assert np.allclose(np.linalg.norm(X[..., g], axis=-1),
                       np.linalg.norm(Xa[..., g], axis=-1), atol=1e-4)
