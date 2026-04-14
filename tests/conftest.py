"""Shared pytest fixtures for PolySwyft tests."""

import logging

import numpy as np
import pytest
import swyft
import torch
from anesthetic import NestedSamples
from lsbi.model import LinearModel

from polyswyft.network import PolySwyftNetwork
from polyswyft.settings import PolySwyftSettings


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
@pytest.fixture
def polyswyft_settings(tmp_path):
    """Minimal PolySwyftSettings with small dimensions for fast tests."""
    settings = PolySwyftSettings(root=str(tmp_path / "test_root"))
    settings.num_features = 2
    settings.num_features_dataset = 4
    settings.num_mixture_components = 2
    settings.n_training_samples = 50
    settings.n_weighted_samples = 20
    settings.n_DKL_estimates = 10
    settings.logger = logging.getLogger("polyswyft_test")
    settings.dm_kwargs = {
        "fractions": [0.8, 0.1, 0.1],
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 0,
    }
    # Disable wandb by default in tests
    settings.activate_wandb = False
    return settings


# -----------------------------------------------------------------------------
# LSBI models
# -----------------------------------------------------------------------------
@pytest.fixture
def lsbi_linear_model(polyswyft_settings):
    """A small LinearModel with deterministic parameters."""
    n = polyswyft_settings.num_features
    d = polyswyft_settings.num_features_dataset
    rng = np.random.default_rng(42)
    m = np.zeros(d)
    M = rng.standard_normal((d, n))
    C = np.eye(d)
    mu = np.zeros(n)
    Sigma = np.eye(n)
    return LinearModel(M=M, C=C, Sigma=Sigma, mu=mu, m=m, n=n, d=d)


# -----------------------------------------------------------------------------
# Simulators
# -----------------------------------------------------------------------------
@pytest.fixture
def mvg_simulator(polyswyft_settings):
    """A PolySwyft_Simulator_MultiGauss.Simulator; also sets settings.model."""
    from examples.mvg.simulator import Simulator

    n = polyswyft_settings.num_features
    d = polyswyft_settings.num_features_dataset
    rng = np.random.default_rng(42)
    m = np.zeros(d)
    M = rng.standard_normal((d, n))
    C = np.eye(d)
    mu = np.zeros(n)
    Sigma = np.eye(n)

    sim = Simulator(polyswyftSettings=polyswyft_settings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)
    polyswyft_settings.model = sim.model
    return sim


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
@pytest.fixture
def fake_obs(polyswyft_settings):
    """A swyft.Sample observation with shape (1, d)."""
    d = polyswyft_settings.num_features_dataset
    torch.manual_seed(7)
    return swyft.Sample(x=torch.randn(1, d, dtype=torch.float64))


# -----------------------------------------------------------------------------
# NestedSamples
# -----------------------------------------------------------------------------
def _make_nested_samples(n_samples, n_features, logL_values=None, seed=0):
    """Utility to construct a valid anesthetic.NestedSamples object."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, n_features))
    if logL_values is None:
        logL = np.sort(rng.standard_normal(n_samples))
    else:
        logL = np.asarray(logL_values, dtype=float)
        assert logL.shape == (n_samples,)
    # logL_birth must be strictly less than logL for each sample; for valid NS,
    # we use [-inf, logL[0], logL[1], ...] which guarantees logL_birth[i] < logL[i].
    logL_birth = np.concatenate([[-np.inf], logL[:-1]])
    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth)


@pytest.fixture
def fake_nested_samples(polyswyft_settings):
    """NestedSamples with synthetic monotonic logL, matching settings.num_features."""
    return _make_nested_samples(n_samples=80, n_features=polyswyft_settings.num_features, seed=42)


@pytest.fixture
def nested_samples_factory(polyswyft_settings):
    """A factory so individual tests can vary sample size / logL shape."""
    n_features = polyswyft_settings.num_features

    def _factory(n_samples=50, logL_values=None, seed=0):
        return _make_nested_samples(n_samples, n_features, logL_values, seed)

    return _factory


# -----------------------------------------------------------------------------
# Concrete Network for tests
# -----------------------------------------------------------------------------
class Network(PolySwyftNetwork):
    """Concrete PolySwyftNetwork for the MVG test case."""

    def __init__(self, polyswyftSettings, obs):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.obs = obs
        self.optimizer_init = swyft.OptimizerInit(
            torch.optim.Adam,
            dict(lr=polyswyftSettings.learning_rate_init),
            torch.optim.lr_scheduler.ExponentialLR,
            dict(gamma=polyswyftSettings.learning_rate_decay),
        )
        self.network = swyft.LogRatioEstimator_Ndim(
            num_features=polyswyftSettings.num_features_dataset,
            marginals=(tuple(range(polyswyftSettings.num_features)),),
            varnames=polyswyftSettings.targetKey,
            dropout=polyswyftSettings.dropout,
            hidden_features=128,
            Lmax=0,
        )

    def forward(self, A, B):
        return self.network(
            A[self.polyswyftSettings.obsKey],
            B[self.polyswyftSettings.targetKey],
        )

    def prior(self, cube):
        return self.polyswyftSettings.model.prior().bijector(x=cube)

    def logRatio(self, theta):
        theta = torch.tensor(theta)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        prediction = self.network(self.obs[self.polyswyftSettings.obsKey], theta)
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        return prediction.logratios[:, 0], []

    def get_new_network(self):
        return Network(polyswyftSettings=self.polyswyftSettings, obs=self.obs)


@pytest.fixture
def network(polyswyft_settings, mvg_simulator, fake_obs):
    """A Network instance with the MVG simulator's model attached to settings."""
    # mvg_simulator already sets polyswyft_settings.model
    net = Network(polyswyftSettings=polyswyft_settings, obs=fake_obs)
    net.double()
    net.eval()
    return net


# -----------------------------------------------------------------------------
# Round data on disk
# -----------------------------------------------------------------------------
@pytest.fixture
def round_data_on_disk(polyswyft_settings, tmp_path):
    """Write .npy files for two rounds under polyswyft_settings.root."""
    root = tmp_path / "round_data_root"
    root.mkdir()
    polyswyft_settings.root = str(root)

    n = polyswyft_settings.num_features
    d = polyswyft_settings.num_features_dataset
    n_samples_per_round = 50
    rng = np.random.default_rng(1234)

    for rd in range(2):
        round_dir = root / f"round_{rd}"
        round_dir.mkdir()
        np.save(
            round_dir / f"{polyswyft_settings.targetKey}.npy",
            rng.standard_normal((n_samples_per_round, n)).astype(np.float64),
        )
        np.save(
            round_dir / f"{polyswyft_settings.obsKey}.npy",
            rng.standard_normal((n_samples_per_round, d)).astype(np.float64),
        )

    return polyswyft_settings
