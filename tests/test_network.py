"""Tests for PolySwyftNetwork implementations."""

import numpy as np
import torch

from tests.conftest import Network


class TestNetworkConstruction:
    def test_init_attributes(self, network, fake_obs):
        assert network.obs is fake_obs
        assert hasattr(network, "optimizer_init")
        assert hasattr(network, "network")  # the inner LogRatioEstimator_Ndim

    def test_inner_network_is_logratio_estimator(self, network):
        import swyft

        assert isinstance(network.network, swyft.LogRatioEstimator_Ndim)

    def test_get_new_network_returns_fresh_instance(self, network):
        new = network.get_new_network()
        assert new is not network
        assert isinstance(new, Network)

    def test_get_new_network_preserves_settings(self, network):
        new = network.get_new_network()
        assert new.polyswyftSettings is network.polyswyftSettings
        assert new.obs is network.obs


class TestNetworkForward:
    def test_forward_returns_logratios(self, network, polyswyft_settings):
        batch_size = 3
        A = {
            polyswyft_settings.obsKey: torch.randn(
                batch_size, polyswyft_settings.num_features_dataset, dtype=torch.float64
            )
        }
        B = {
            polyswyft_settings.targetKey: torch.randn(batch_size, polyswyft_settings.num_features, dtype=torch.float64)
        }
        with torch.no_grad():
            out = network(A, B)
        assert hasattr(out, "logratios")
        assert out.logratios.shape[0] == batch_size

    def test_forward_batch_varies(self, network, polyswyft_settings):
        """Different batch sizes should work."""
        for batch_size in [1, 4, 16]:
            A = {
                polyswyft_settings.obsKey: torch.randn(
                    batch_size, polyswyft_settings.num_features_dataset, dtype=torch.float64
                )
            }
            B = {
                polyswyft_settings.targetKey: torch.randn(
                    batch_size, polyswyft_settings.num_features, dtype=torch.float64
                )
            }
            with torch.no_grad():
                out = network(A, B)
            assert out.logratios.shape[0] == batch_size


class TestNetworkLogRatio:
    def test_logratio_1d_returns_float_and_list(self, network, polyswyft_settings):
        theta = np.zeros(polyswyft_settings.num_features)
        with torch.no_grad():
            result = network.logRatio(theta)
        assert isinstance(result, tuple)
        assert len(result) == 2
        logL, derived = result
        assert isinstance(logL, float)
        assert derived == []
        assert np.isfinite(logL)

    def test_logratio_batch_returns_tensor(self, network, polyswyft_settings):
        theta_batch = np.random.randn(5, polyswyft_settings.num_features)
        with torch.no_grad():
            result = network.logRatio(theta_batch)
        logL, derived = result
        assert torch.is_tensor(logL)
        assert logL.shape == (5,)
        assert torch.all(torch.isfinite(logL))
        assert derived == []

    def test_logratio_finite_everywhere(self, network, polyswyft_settings):
        """Randomly sampled theta should always yield finite logRatio values."""
        rng = np.random.default_rng(0)
        for _ in range(5):
            theta = rng.standard_normal(polyswyft_settings.num_features)
            with torch.no_grad():
                logL, _ = network.logRatio(theta)
            assert np.isfinite(logL)


class TestNetworkPrior:
    def test_prior_output_shape(self, network, polyswyft_settings):
        cube = np.full(polyswyft_settings.num_features, 0.5)
        theta = network.prior(cube)
        assert theta.shape == (polyswyft_settings.num_features,)

    def test_prior_output_finite(self, network, polyswyft_settings):
        cube = np.array([0.1, 0.9])
        theta = network.prior(cube)
        assert np.all(np.isfinite(theta))

    def test_prior_center_maps_to_mu(self, network, polyswyft_settings):
        """For a standard normal prior N(0, I), cube [0.5, 0.5] maps to mean 0."""
        cube = np.full(polyswyft_settings.num_features, 0.5)
        theta = network.prior(cube)
        np.testing.assert_allclose(theta, np.zeros(polyswyft_settings.num_features), atol=1e-10)


class TestNetworkPrecision:
    def test_double_precision(self, network):
        """After .double() call in fixture, params should be float64."""
        for p in network.parameters():
            assert p.dtype == torch.float64
            break  # one is sufficient

    def test_eval_mode(self, network):
        assert network.training is False

    def test_train_mode_toggle(self, network):
        network.train()
        assert network.training is True
        network.eval()
        assert network.training is False


class TestNetworkStateDict:
    def test_state_dict_roundtrip(self, network, tmp_path):
        """Save, mutate, load -- parameters should match the saved snapshot."""
        path = tmp_path / "net.pt"
        torch.save(network.state_dict(), path)

        # Mutate parameters
        with torch.no_grad():
            for p in network.parameters():
                p.zero_()

        loaded = torch.load(path)
        network.load_state_dict(loaded)

        # Check at least one parameter is no longer zero
        nonzero_found = False
        for p in network.parameters():
            if torch.any(p != 0):
                nonzero_found = True
                break
        assert nonzero_found


class TestNetworkDumper:
    def test_dumper_runs_without_error(self, network, capsys):
        """dumper() just prints the last dead point; smoke test."""
        live = np.zeros((5, 3))
        dead = np.zeros((10, 3))
        logweights = np.zeros(10)
        network.dumper(live, dead, logweights, logZ=0.0, logZerr=0.0)
        captured = capsys.readouterr()
        assert "Last dead point" in captured.out
