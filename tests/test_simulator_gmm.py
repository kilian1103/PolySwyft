"""Tests for examples.gmm.simulator.Simulator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip the whole module if examples.gmm.simulator can't be imported
# (requires lsbi >= 0.12.0 for MixtureModel).
simulator_module = pytest.importorskip(
    "examples.gmm.simulator",
    reason="GMM simulator requires lsbi >= 0.12.0 (MixtureModel)",
)
Simulator = simulator_module.Simulator

from lsbi.model import MixtureModel  # noqa: E402


@pytest.fixture
def gmm_model(polyswyft_settings):
    n = polyswyft_settings.num_features
    d = polyswyft_settings.num_features_dataset
    a = polyswyft_settings.num_mixture_components
    rng = np.random.default_rng(7)
    M = np.broadcast_to(rng.standard_normal((d, n)), (a, d, n)).copy()
    C = np.broadcast_to(np.eye(d), (a, d, d)).copy()
    Sigma = np.broadcast_to(np.eye(n), (a, n, n)).copy()
    mu = rng.standard_normal((a, n))
    m = np.zeros((a, d))
    logw = np.log(np.ones(a) / a)
    return MixtureModel(M=M, C=C, Sigma=Sigma, mu=mu, m=m, logw=logw, n=n, d=d)


@pytest.fixture
def gmm_simulator(polyswyft_settings, gmm_model):
    sim = Simulator(polyswyftSettings=polyswyft_settings, model=gmm_model)
    polyswyft_settings.model = sim.model
    return sim


def test_gmm_simulator_init(gmm_simulator, gmm_model):
    assert gmm_simulator.model is gmm_model


def test_gmm_simulator_records_dimensions(gmm_simulator, polyswyft_settings):
    assert gmm_simulator.n == polyswyft_settings.num_features
    assert gmm_simulator.d == polyswyft_settings.num_features_dataset
    assert gmm_simulator.a == polyswyft_settings.num_mixture_components


def test_gmm_simulator_prior_shape(gmm_simulator, polyswyft_settings):
    z = gmm_simulator.prior()
    assert z.shape == (polyswyft_settings.num_features,)
    assert np.all(np.isfinite(z))


def test_gmm_simulator_likelihood_shape(gmm_simulator, polyswyft_settings):
    z = gmm_simulator.prior()
    x = gmm_simulator.likelihood(z)
    assert x.shape == (polyswyft_settings.num_features_dataset,)
    assert np.all(np.isfinite(x))


def test_gmm_simulator_posterior_shape(gmm_simulator, polyswyft_settings):
    x = gmm_simulator.model.evidence().rvs()
    theta = gmm_simulator.posterior(x)
    assert theta.shape == (polyswyft_settings.num_features,)
    assert np.all(np.isfinite(theta))


def test_gmm_simulator_logratio_scalar(gmm_simulator):
    z = gmm_simulator.prior()
    x = gmm_simulator.likelihood(z)
    lr = gmm_simulator.logratio(x, z)
    assert np.ndim(lr) == 0
    assert np.isfinite(lr)


def test_gmm_simulator_build_keys(gmm_simulator, polyswyft_settings):
    graph = MagicMock()
    gmm_simulator.build(graph)
    calls = graph.node.call_args_list
    assert len(calls) == 4
    assert calls[0].args[0] == polyswyft_settings.targetKey
    assert calls[1].args[0] == polyswyft_settings.obsKey
    assert calls[2].args[0] == polyswyft_settings.posteriorsKey
    assert calls[3].args[0] == polyswyft_settings.contourKey
