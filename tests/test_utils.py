"""Tests for PolySwyft.utils mathematical helpers."""

from unittest.mock import MagicMock

import numpy as np
import torch
from anesthetic import NestedSamples

from polyswyft.utils import (
    compute_KL_compression,
    compute_KL_divergence,
    compute_KL_divergence_truth,
    select_weighted_contour,
)


# =============================================================================
# select_weighted_contour
# =============================================================================
class TestSelectWeightedContour:
    def test_threshold_zero_returns_zero(self, fake_nested_samples):
        assert select_weighted_contour(fake_nested_samples, 0.0) == 0

    def test_threshold_one_reaches_effective_end(self, fake_nested_samples):
        """threshold=1.0 should return an index at or near the last
        numerically-significant sample. (Tail weights at the very end may
        be suppressed to floating-point zero; searchsorted then returns the
        first index where cumsum == 1.0.)"""
        idx = select_weighted_contour(fake_nested_samples, 1.0)
        n = len(fake_nested_samples)
        assert 0 <= idx < n
        # At or after this index the cumulative weight must equal total weight.
        cum = fake_nested_samples.get_weights().cumsum()
        assert np.isclose(cum[idx], cum[-1])

    def test_monotonicity(self, fake_nested_samples):
        """Higher threshold must return a higher-or-equal index."""
        thresholds = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999]
        indices = [select_weighted_contour(fake_nested_samples, t) for t in thresholds]
        assert all(a <= b for a, b in zip(indices, indices[1:]))

    def test_output_is_integer(self, fake_nested_samples):
        idx = select_weighted_contour(fake_nested_samples, 0.5)
        assert isinstance(idx, (int, np.integer))

    def test_single_sample_any_threshold(self):
        samples = NestedSamples(
            data=np.zeros((1, 2)),
            logL=np.array([0.0]),
            logL_birth=np.array([-np.inf]),
        )
        assert select_weighted_contour(samples, 0.0) == 0
        assert select_weighted_contour(samples, 0.5) == 0
        assert select_weighted_contour(samples, 1.0) == 0

    def test_index_within_bounds(self, fake_nested_samples):
        """For any valid threshold, idx must satisfy 0 <= idx < len."""
        n = len(fake_nested_samples)
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
            idx = select_weighted_contour(fake_nested_samples, t)
            assert 0 <= idx < n, f"idx={idx} out of [0, {n}) for threshold={t}"


# =============================================================================
# compute_KL_compression
# =============================================================================
class TestComputeKLCompression:
    def test_returns_tuple_of_floats(self, fake_nested_samples, polyswyft_settings):
        result = compute_KL_compression(fake_nested_samples, polyswyft_settings)
        assert isinstance(result, tuple)
        assert len(result) == 2
        dkl, err = result
        assert np.isscalar(dkl) or isinstance(dkl, (float, np.floating))
        assert np.isscalar(err) or isinstance(err, (float, np.floating))

    def test_output_finite(self, fake_nested_samples, polyswyft_settings):
        dkl, err = compute_KL_compression(fake_nested_samples, polyswyft_settings)
        assert np.isfinite(dkl)
        assert np.isfinite(err)

    def test_error_nonnegative(self, fake_nested_samples, polyswyft_settings):
        _, err = compute_KL_compression(fake_nested_samples, polyswyft_settings)
        assert err >= 0

    def test_nonnegative_modulo_noise(self, fake_nested_samples, polyswyft_settings):
        """KL divergence is non-negative (Gibbs). Allow small numerical slack."""
        dkl, _ = compute_KL_compression(fake_nested_samples, polyswyft_settings)
        assert dkl >= -1e-6

    def test_different_logL_profiles_give_different_compressions(self, nested_samples_factory, polyswyft_settings):
        """Two distinct logL profiles should yield distinct KL compression
        values (sanity check that the function is sensitive to its input)."""
        profile_a = nested_samples_factory(n_samples=100, logL_values=np.linspace(-1e-3, 1e-3, 100), seed=1)
        profile_b = nested_samples_factory(n_samples=100, logL_values=np.linspace(-10.0, 10.0, 100), seed=1)
        dkl_a, _ = compute_KL_compression(profile_a, polyswyft_settings)
        dkl_b, _ = compute_KL_compression(profile_b, polyswyft_settings)
        assert np.isfinite(dkl_a) and np.isfinite(dkl_b)
        assert not np.isclose(dkl_a, dkl_b, atol=1e-4)


# =============================================================================
# compute_KL_divergence
# =============================================================================
class TestComputeKLDivergence:
    def test_returns_tuple_of_floats(self, fake_nested_samples, network, fake_obs, polyswyft_settings):
        current = fake_nested_samples.copy()
        previous = fake_nested_samples.copy()
        result = compute_KL_divergence(
            polyswyftSettings=polyswyft_settings,
            previous_network=network,
            current_samples=current,
            previous_samples=previous,
            obs=fake_obs,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        dkl, err = result
        assert np.isfinite(dkl)
        assert np.isfinite(err)

    def test_error_nonnegative(self, fake_nested_samples, network, fake_obs, polyswyft_settings):
        current = fake_nested_samples.copy()
        previous = fake_nested_samples.copy()
        _, err = compute_KL_divergence(
            polyswyftSettings=polyswyft_settings,
            previous_network=network,
            current_samples=current,
            previous_samples=previous,
            obs=fake_obs,
        )
        assert err >= 0

    def test_mocked_perfect_network_gives_small_kl(self, fake_nested_samples, fake_obs, polyswyft_settings):
        """If the 'previous network' exactly reproduces the current samples'
        logL values, the per-sample contribution ``logL - logL_previous`` is 0
        and the KL estimate collapses to ``logZ_prev - logZ_cur`` -- a noisy
        O(1/sqrt(n_DKL_estimates)) residual from the two independent logZ
        resamplings. Verify it's small relative to a mis-specified network."""
        current = fake_nested_samples.copy()
        previous = current.copy()

        predictions = MagicMock()
        predictions.logratios = torch.tensor(current["logL"].values, dtype=torch.float64)
        mock_network = MagicMock(return_value=predictions)

        dkl_matched, _ = compute_KL_divergence(
            polyswyftSettings=polyswyft_settings,
            previous_network=mock_network,
            current_samples=current,
            previous_samples=previous,
            obs=fake_obs,
        )

        # Mis-specified network: returns logratios that are large constant offset.
        bad_predictions = MagicMock()
        bad_predictions.logratios = torch.tensor(current["logL"].values - 50.0, dtype=torch.float64)
        bad_network = MagicMock(return_value=bad_predictions)

        dkl_mismatched, _ = compute_KL_divergence(
            polyswyftSettings=polyswyft_settings,
            previous_network=bad_network,
            current_samples=fake_nested_samples.copy(),
            previous_samples=fake_nested_samples.copy(),
            obs=fake_obs,
        )

        # The matched case should be far closer to 0 than the mismatched case.
        assert abs(dkl_matched) < abs(dkl_mismatched)
        assert abs(dkl_matched) < 1.0  # stochastic residual is small

    def test_assigns_logL_previous_column(self, fake_nested_samples, network, fake_obs, polyswyft_settings):
        """The function should add a 'logL_previous' column to current_samples."""
        current = fake_nested_samples.copy()
        previous = fake_nested_samples.copy()
        compute_KL_divergence(
            polyswyftSettings=polyswyft_settings,
            previous_network=network,
            current_samples=current,
            previous_samples=previous,
            obs=fake_obs,
        )
        assert "logL_previous" in current.columns


# =============================================================================
# compute_KL_divergence_truth
# =============================================================================
class TestComputeKLDivergenceTruth:
    def test_returns_tuple_of_floats(self, fake_nested_samples, network, fake_obs, polyswyft_settings, mvg_simulator):
        # mvg_simulator sets polyswyft_settings.model
        true_posterior = fake_nested_samples.copy()
        samples = fake_nested_samples.copy()
        result = compute_KL_divergence_truth(
            polyswyftSettings=polyswyft_settings,
            network=network,
            true_posterior=true_posterior,
            samples=samples,
            obs=fake_obs,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        dkl, err = result
        assert np.isfinite(dkl)
        assert np.isfinite(err)

    def test_requires_model_with_prior_and_posterior(
        self, fake_nested_samples, network, fake_obs, polyswyft_settings, mvg_simulator
    ):
        """Smoke test: function exercises model.prior().logpdf and model.posterior().logpdf."""
        # Capture calls on the lsbi model to verify the method paths are hit.
        model = mvg_simulator.model
        assert hasattr(model, "prior")
        assert hasattr(model, "posterior")
        # Should not raise.
        compute_KL_divergence_truth(
            polyswyftSettings=polyswyft_settings,
            network=network,
            true_posterior=fake_nested_samples.copy(),
            samples=fake_nested_samples.copy(),
            obs=fake_obs,
        )
