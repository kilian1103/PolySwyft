"""Tests for PolySwyft.PolySwyft_Simulator_MultiGauss.Simulator."""

from unittest.mock import MagicMock

import numpy as np
import torch
from lsbi.model import LinearModel

from examples.mvg.simulator import Simulator


def test_simulator_init_stores_model(mvg_simulator):
    assert isinstance(mvg_simulator.model, LinearModel)


def test_simulator_init_records_dimensions(mvg_simulator, polyswyft_settings):
    assert mvg_simulator.n == polyswyft_settings.num_features
    assert mvg_simulator.d == polyswyft_settings.num_features_dataset


def test_simulator_prior_shape(mvg_simulator, polyswyft_settings):
    sample = mvg_simulator.prior()
    assert sample.shape == (polyswyft_settings.num_features,)
    assert np.all(np.isfinite(sample))


def test_simulator_likelihood_shape(mvg_simulator, polyswyft_settings):
    z = mvg_simulator.prior()
    x = mvg_simulator.likelihood(z)
    assert x.shape == (polyswyft_settings.num_features_dataset,)
    assert np.all(np.isfinite(x))


def test_simulator_posterior_shape(mvg_simulator, polyswyft_settings):
    x = mvg_simulator.model.evidence().rvs()
    theta = mvg_simulator.posterior(x)
    assert theta.shape == (polyswyft_settings.num_features,)
    assert np.all(np.isfinite(theta))


def test_simulator_logratio_is_scalar(mvg_simulator):
    z = mvg_simulator.prior()
    x = mvg_simulator.likelihood(z)
    lr = mvg_simulator.logratio(x, z)
    assert np.ndim(lr) == 0  # scalar
    assert np.isfinite(lr)


def test_simulator_logratio_identity(mvg_simulator):
    """logratio == log p(x|z) - log p(x)."""
    z = mvg_simulator.prior()
    x = mvg_simulator.likelihood(z)
    expected = mvg_simulator.model.likelihood(z).logpdf(x) - mvg_simulator.model.evidence().logpdf(x)
    got = mvg_simulator.logratio(x, z)
    assert np.isclose(float(got), float(expected), rtol=1e-10)


def test_simulator_prior_mean_close_to_mu(polyswyft_settings):
    """Statistically, the mean of many prior draws should be near mu."""
    n = polyswyft_settings.num_features
    d = polyswyft_settings.num_features_dataset
    m = torch.zeros(d, dtype=torch.float64)
    M = torch.eye(d, n, dtype=torch.float64)
    C = torch.eye(d, dtype=torch.float64)
    mu = torch.tensor([1.5, -2.0], dtype=torch.float64)
    Sigma = torch.eye(n, dtype=torch.float64) * 0.01  # small variance -> tight concentration
    sim = Simulator(polyswyftSettings=polyswyft_settings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)

    np.random.seed(0)
    samples = np.array([sim.prior() for _ in range(2000)])
    mean_est = samples.mean(axis=0)
    assert np.allclose(mean_est, mu.numpy(), atol=0.05)


def test_simulator_build_registers_correct_nodes(mvg_simulator, polyswyft_settings):
    """build() should call graph.node with the keys from settings in expected order."""
    graph = MagicMock()
    mvg_simulator.build(graph)

    call_args = [call.args for call in graph.node.call_args_list]
    assert len(call_args) == 4

    # Order: z (prior), x (likelihood), post (posterior), l (logratio)
    assert call_args[0][0] == polyswyft_settings.targetKey
    assert call_args[1][0] == polyswyft_settings.obsKey
    assert call_args[2][0] == polyswyft_settings.posteriorsKey
    assert call_args[3][0] == polyswyft_settings.contourKey


def test_simulator_build_passes_callables(mvg_simulator):
    """The functions registered on the graph are the simulator's methods."""
    graph = MagicMock()
    mvg_simulator.build(graph)
    calls = graph.node.call_args_list
    # Bound methods create fresh objects per access, so compare underlying funcs.
    assert calls[0].args[1].__func__ is Simulator.prior
    assert calls[1].args[1].__func__ is Simulator.likelihood
    assert calls[2].args[1].__func__ is Simulator.posterior
    assert calls[3].args[1].__func__ is Simulator.logratio
