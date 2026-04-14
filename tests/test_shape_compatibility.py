"""End-to-end shape regression tests.

These exercise the full simulator -> on-disk -> dataloader -> network
pipeline for several (num_features, num_features_dataset) combinations,
including the scalar-dimension cases (n == 1 or d == 1) where lsbi squeezes
its output and previously broke the pipeline.
"""

import logging
import os

import numpy as np
import pytest
import swyft
import torch
from torch.utils.data import DataLoader

from examples.mvg.simulator import Simulator
from polyswyft.dataloader import PolySwyftSequential
from polyswyft.settings import PolySwyftSettings
from tests.conftest import Network

SHAPE_CASES = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (3, 2), (3, 5)]


def _build_settings(tmp_path, n, d):
    s = PolySwyftSettings(root=str(tmp_path))
    s.num_features = n
    s.num_features_dataset = d
    s.logger = logging.getLogger("shape_test")
    s.activate_wandb = False
    return s


def _build_simulator(settings):
    n = settings.num_features
    d = settings.num_features_dataset
    torch.manual_seed(0)
    return Simulator(
        polyswyftSettings=settings,
        m=torch.zeros(d, dtype=torch.float64),
        M=torch.randn(d, n, dtype=torch.float64),
        C=torch.eye(d, dtype=torch.float64),
        mu=torch.zeros(n, dtype=torch.float64),
        Sigma=torch.eye(n, dtype=torch.float64),
    )


@pytest.mark.parametrize("n,d", SHAPE_CASES)
class TestShapeCompatibility:
    def test_simulator_scalar_methods_return_1d(self, tmp_path, n, d):
        """prior/likelihood/posterior must return shape (n,) / (d,) / (n,)
        even when n == 1 or d == 1 (lsbi otherwise squeezes these)."""
        settings = _build_settings(tmp_path, n, d)
        sim = _build_simulator(settings)

        z = sim.prior()
        assert z.shape == (n,), f"prior() shape {z.shape} != ({n},)"

        x = sim.likelihood(z)
        assert x.shape == (d,), f"likelihood() shape {x.shape} != ({d},)"

        post = sim.posterior(x)
        assert post.shape == (n,), f"posterior() shape {post.shape} != ({n},)"

        lr = sim.logratio(x, z)
        assert np.ndim(lr) == 0 and np.isfinite(lr)

    def test_sim_sample_produces_correct_2d_arrays(self, tmp_path, n, d):
        """swyft.Simulator.sample(N) must stack scalar outputs into (N, n) / (N, d)."""
        settings = _build_settings(tmp_path, n, d)
        settings.model = _build_simulator(settings).model
        sim = _build_simulator(settings)
        batch = sim.sample(8)
        z_arr = np.asarray(batch[settings.targetKey])
        x_arr = np.asarray(batch[settings.obsKey])
        assert z_arr.shape == (8, n)
        assert x_arr.shape == (8, d)

    def test_dataloader_yields_correctly_shaped_batches(self, tmp_path, n, d):
        """Sample -> save -> PolySwyftSequential -> DataLoader batches must
        arrive with shape (batch, n) and (batch, d)."""
        settings = _build_settings(tmp_path, n, d)
        settings.model = _build_simulator(settings).model
        sim = _build_simulator(settings)
        batch = sim.sample(10)
        z_arr = np.asarray(batch[settings.targetKey])
        x_arr = np.asarray(batch[settings.obsKey])

        os.makedirs(f"{settings.root}/round_0", exist_ok=True)
        np.save(f"{settings.root}/round_0/{settings.targetKey}.npy", z_arr)
        np.save(f"{settings.root}/round_0/{settings.obsKey}.npy", x_arr)

        ds = PolySwyftSequential(polyswyftSettings=settings, rd=0)
        sample0 = ds[0]
        assert sample0[settings.targetKey].shape == (n,)
        assert sample0[settings.obsKey].shape == (d,)

        loader = DataLoader(ds, batch_size=4)
        loader_batch = next(iter(loader))
        assert loader_batch[settings.targetKey].shape == (4, n)
        assert loader_batch[settings.obsKey].shape == (4, d)

    def test_network_forward_on_dataloader_batch(self, tmp_path, n, d):
        """Network (float32, as produced by the dataloader) must accept the
        dataloader's batches and return logratios of shape (batch, 1)."""
        settings = _build_settings(tmp_path, n, d)
        settings.model = _build_simulator(settings).model
        sim = _build_simulator(settings)
        batch = sim.sample(10)

        os.makedirs(f"{settings.root}/round_0", exist_ok=True)
        np.save(
            f"{settings.root}/round_0/{settings.targetKey}.npy",
            np.asarray(batch[settings.targetKey]),
        )
        np.save(
            f"{settings.root}/round_0/{settings.obsKey}.npy",
            np.asarray(batch[settings.obsKey]),
        )

        ds = PolySwyftSequential(polyswyftSettings=settings, rd=0)
        loader = DataLoader(ds, batch_size=4)
        loader_batch = next(iter(loader))

        obs = swyft.Sample(x=torch.randn(1, d, dtype=torch.float32))
        net = Network(polyswyftSettings=settings, obs=obs).eval()

        with torch.no_grad():
            out = net(loader_batch, loader_batch)
        assert out.logratios.shape == (4, 1)

    def test_network_logratio_1d_and_batch(self, tmp_path, n, d):
        """logRatio must handle both a single 1-D theta and a batch."""
        settings = _build_settings(tmp_path, n, d)
        settings.model = _build_simulator(settings).model

        obs = swyft.Sample(x=torch.randn(1, d, dtype=torch.float64))
        net = Network(polyswyftSettings=settings, obs=obs).double().eval()

        with torch.no_grad():
            ll1, derived1 = net.logRatio(np.zeros(n))
        assert isinstance(ll1, float)
        assert derived1 == []
        assert np.isfinite(ll1)

        with torch.no_grad():
            llB, derivedB = net.logRatio(np.random.randn(5, n))
        assert torch.is_tensor(llB)
        assert llB.shape == (5,)
        assert torch.all(torch.isfinite(llB))
        assert derivedB == []

    def test_network_prior_cube_transform(self, tmp_path, n, d):
        """Network.prior(cube) must return an array of shape (n,) for the
        unit-cube mapping, even when n == 1."""
        settings = _build_settings(tmp_path, n, d)
        settings.model = _build_simulator(settings).model

        obs = swyft.Sample(x=torch.randn(1, d, dtype=torch.float64))
        net = Network(polyswyftSettings=settings, obs=obs).double().eval()

        theta = net.prior(np.full(n, 0.5))
        assert theta.shape == (n,)
        # For N(0, I) prior the center of the unit cube maps to 0.
        np.testing.assert_allclose(theta, np.zeros(n), atol=1e-10)
