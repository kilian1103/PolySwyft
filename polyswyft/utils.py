"""Utility functions: KL diagnostics, simulator dispatch, plotting reload.

Implements the convergence metrics referenced by the NSNRE cycle:

* :func:`compute_KL_divergence` — paper eq. (10), ``KL(P_i || P_{i-1})``.
* :func:`compute_KL_compression` — paper eq. (12), ``KL(P_i || pi)``.
* :func:`compute_KL_divergence_truth` — analytical
  ``KL(P_true || P_i)`` for toy problems with closed-form posteriors.

Also provides :func:`resimulate_deadpoints` (the MPI-aware simulator
dispatcher) and :func:`reload_data_for_plotting` (post-hoc artefact
loader for the example plotting scripts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anesthetic
import numpy as np
import swyft
import torch
from anesthetic import NestedSamples
from scipy.special import logsumexp
from swyft import Simulator
from swyft import collate_output as reformat_samples

from polyswyft.network import PolySwyftNetwork
from polyswyft.settings import PolySwyftSettings

if TYPE_CHECKING:
    from pypolychord import PolyChordSettings


def select_weighted_contour(data: NestedSamples, threshold: float) -> int:
    """Index of the dead point whose cumulative weight crosses ``threshold``.

    Used by the optional dynamic-nested-sampling pass to locate the log-r
    iso-contour that encloses a given posterior fraction (e.g. 99.9%).

    Parameters
    ----------
    data : anesthetic.NestedSamples
        PolyChord output; weights are taken from ``data.get_weights()``.
    threshold : float
        Cumulative-weight fraction in ``[0, 1]``.

    Returns
    -------
    int
        Index into ``data`` of the first sample whose cumulative weight
        exceeds ``threshold``.
    """
    cumulative_weights = data.get_weights().cumsum()
    cumulative_weights_norm = cumulative_weights / cumulative_weights[-1]
    index = np.searchsorted(cumulative_weights_norm, threshold)
    return index


def compute_KL_divergence(
    polyswyftSettings: PolySwyftSettings,
    previous_network: swyft.SwyftModule,
    current_samples: anesthetic.Samples,
    previous_samples: anesthetic.Samples,
    obs: swyft.Sample,
) -> tuple[float, float]:
    """Estimate ``KL(P_i || P_{i-1})`` between two PolyChord rounds.

    Implements paper eq. (10): each summand is

    .. math::
        \\log r_i^*(\\theta_n, D_{obs}) - \\log Z_{NS,i}
        - \\log r_{i-1}^*(\\theta_n, D_{obs}) + \\log Z_{NS,i-1}

    weighted by the round-i posterior weights, with ``Z_NS`` estimated by
    each round's PolyChord run (so the constant offset between raw
    ``log r*`` and the prior-corrected ratio cancels).

    Parameters
    ----------
    polyswyftSettings : PolySwyftSettings
        Provides ``targetKey``, ``num_features`` and ``n_DKL_estimates``
        bootstrap count.
    previous_network : swyft.SwyftModule
        NRE from round ``i-1`` (eval mode).
    current_samples : anesthetic.Samples
        Round-i PolyChord output; ``current_samples["logL"]`` carries the
        round-i log-ratio at each dead point.
    previous_samples : anesthetic.Samples
        Round ``i-1`` PolyChord output; only its ``logZ`` is used.
    obs : swyft.Sample
        Observed data passed to ``previous_network`` to evaluate
        ``log r_{i-1}^*`` at the round-i samples.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the bootstrapped KL estimate.
    """

    samples = {
        polyswyftSettings.targetKey: torch.as_tensor(
            current_samples.iloc[:, : polyswyftSettings.num_features].to_numpy()
        )
    }
    with torch.no_grad():
        predictions = previous_network(obs, samples)
    current_samples["logL_previous"] = predictions.logratios.detach().cpu().numpy().squeeze()

    logw = current_samples.logw(polyswyftSettings.n_DKL_estimates)
    logpqs = (
        current_samples["logL"].values[:, None]
        - current_samples.logZ(logw).values
        - current_samples["logL_previous"].values[:, None]
        + previous_samples.logZ(polyswyftSettings.n_DKL_estimates).values
    )
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()

    return DKL, DKL_err


def compute_KL_divergence_truth(
    polyswyftSettings: PolySwyftSettings,
    network: swyft.SwyftModule,
    true_posterior: anesthetic.Samples,
    samples: anesthetic.Samples,
    obs: swyft.Sample,
) -> tuple[float, float]:
    """Estimate ``KL(P_true || P_i)`` for a toy problem with analytical posterior.

    Used by the MVG and GMM examples to score PolySwyft against the closed-form
    answer. Requires ``polyswyftSettings.model`` to expose ``prior()`` and
    ``posterior(D)`` with ``.logpdf(theta)``.

    Parameters
    ----------
    polyswyftSettings : PolySwyftSettings
        Must have ``model`` set to an analytical ``lsbi`` model.
    network : swyft.SwyftModule
        Trained NRE for round ``i``.
    true_posterior : anesthetic.Samples
        Direct draws from the analytical posterior with unit weights.
    samples : anesthetic.Samples
        Round-i PolyChord output; only its ``logZ`` is used as ``log Z_NS``.
    obs : swyft.Sample
        Observed data fed to the NRE.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the KL estimate across the
        ``true_posterior`` sample bootstrap.
    """
    swyft_samples = {
        polyswyftSettings.targetKey: torch.as_tensor(
            true_posterior.iloc[:, : polyswyftSettings.num_features].to_numpy()
        )
    }
    with torch.no_grad():
        predictions = network(obs, swyft_samples)
    true_posterior["logR"] = predictions.logratios.detach().cpu().numpy().squeeze()
    true_posterior_samples = true_posterior.iloc[:, : polyswyftSettings.num_features].squeeze()
    true_prior = polyswyftSettings.model.prior().logpdf(true_posterior_samples)
    true_posterior_logL = polyswyftSettings.model.posterior(obs[polyswyftSettings.obsKey].numpy().squeeze()).logpdf(
        true_posterior_samples
    )
    true_posterior["logL"] = true_posterior_logL

    logpqs = (
        true_posterior["logL"].values[:, None]
        - true_posterior["logR"].values[:, None]
        - true_prior[:, None]
        + samples.logZ(polyswyftSettings.n_DKL_estimates).values
    )
    DKL_estimates = logpqs.mean(axis=0)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return (DKL, DKL_err)


def compute_KL_compression(samples: anesthetic.NestedSamples, polyswyftSettings: PolySwyftSettings):
    """Estimate ``KL(P_i || pi)`` — compression of prior to posterior at round ``i``.

    Implements paper eq. (12). Used both as a convergence diagnostic
    (PolySwyft should terminate only when this exceeds the user-chosen
    ``C_comp``) and as the reported compression in the per-round KL plots.

    Parameters
    ----------
    samples : anesthetic.NestedSamples
        Round-i PolyChord output; ``samples["logL"]`` carries the round-i
        log-ratio at each dead point.
    polyswyftSettings : PolySwyftSettings
        Provides ``n_DKL_estimates`` bootstrap count.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the bootstrapped KL estimate.
    """
    logw = samples.logw(polyswyftSettings.n_DKL_estimates)
    logpqs = samples["logL"].values[:, None] - samples.logZ(logw).values
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return DKL, DKL_err


def reload_data_for_plotting(
    polyswyftSettings: PolySwyftSettings,
    network: PolySwyftNetwork,
    polyset: PolyChordSettings,
    until_round: int,
    only_last_round=False,
) -> tuple[
    dict[int, str], dict[int, swyft.SwyftModule], dict[int, anesthetic.NestedSamples], dict[int, tuple[float, float]]
]:
    """Reload per-round artefacts from disk for post-hoc plotting.

    Walks ``{root}/round_0..round_{until_round}/`` and restores the trained
    network, PolyChord chains, and ``KL(P_i || P_{i-1})`` value for each
    round. Designed for use after a completed run when only plots need to
    be regenerated.

    Parameters
    ----------
    polyswyftSettings : PolySwyftSettings
        Active run settings; ``root`` and key naming are read from here.
    network : PolySwyftNetwork
        Template network used to instantiate fresh modules before loading
        each round's state-dict.
    polyset : pypolychord.PolyChordSettings
        PolyChord settings; only ``file_root`` is used.
    until_round : int
        Last round to reload (inclusive).
    only_last_round : bool, default False
        If ``True``, restore only ``until_round - 1`` and ``until_round``
        (sufficient to recompute the final KL). Earlier rounds are skipped.

    Returns
    -------
    tuple of four dicts
        ``(root_storage, network_storage, samples_storage, dkl_storage)``
        keyed by round index.
    """

    network_storage = {}
    root_storage = {}
    samples_storage = {}
    dkl_storage = {}
    root = polyswyftSettings.root

    try:
        obs = network.obs
    except AttributeError as err:
        raise AttributeError("network object does not have an attribute 'obs'") from err

    for rd in range(until_round + 1):
        if only_last_round and rd < until_round - 1:
            continue

        current_root = f"{root}/{polyswyftSettings.child_root}_{rd}"
        root_storage[rd] = current_root

        # load network
        new_network = network.get_new_network()
        new_network.load_state_dict(torch.load(f"{current_root}/{polyswyftSettings.neural_network_file}"))
        new_network.double()  # change to float64 precision of network
        network_storage[rd] = new_network

        # load samples
        params = [rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)]
        if polyswyftSettings.use_livepoint_increasing:
            samples = anesthetic.read_chains(
                root=f"{root_storage[rd]}/{polyswyftSettings.increased_livepoints_fileroot}/{polyset.file_root}"
            )
        else:
            samples = anesthetic.read_chains(root=f"{root_storage[rd]}/{polyset.file_root}")
        labels = samples.get_labels()
        labels[: polyswyftSettings.num_features] = params
        samples.set_labels(labels, inplace=True)
        samples_storage[rd] = samples

        # compute DKL
        if rd > 0:
            if only_last_round and rd < until_round:
                continue
            previous_network = network_storage[rd - 1]
            KDL = compute_KL_divergence(
                polyswyftSettings=polyswyftSettings,
                previous_network=previous_network.eval(),
                current_samples=samples_storage[rd],
                obs=obs,
                previous_samples=samples_storage[rd - 1],
            )
            dkl_storage[rd] = KDL
    return root_storage, network_storage, samples_storage, dkl_storage


def resimulate_deadpoints(deadpoints: np.ndarray, polyswyftSettings: PolySwyftSettings, sim: Simulator, rd: int):
    """Run the simulator at each dead point and write the round's training set.

    Splits ``deadpoints`` across MPI ranks, draws the corresponding ``D``
    via ``sim.sample`` (or the noise-resampler when
    ``polyswyftSettings.use_noise_resampling`` is on), gathers everything
    on rank 0, shuffles row-wise to break the PolyChord ordinal correlation,
    broadcasts back, and persists ``z.npy`` / ``x.npy`` (or whatever the
    ``targetKey`` / ``obsKey`` settings name) under
    ``{root}/{child_root}_{rd}/``.

    Parameters
    ----------
    deadpoints : np.ndarray
        Parameter samples from the previous round, shape
        ``(N, num_features)``.
    polyswyftSettings : PolySwyftSettings
        Active run settings; controls keys, paths, and noise resampling.
    sim : swyft.Simulator
        Forward simulator producing ``D`` conditioned on ``theta``.
    rd : int
        Round index; selects the output subdirectory.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(thetas, Ds)`` after MPI gather and shuffle. Persisted to disk
        on rank 0.

    Raises
    ------
    ImportError
        If ``mpi4py`` is not installed.
    """
    logger = polyswyftSettings.logger
    try:
        from mpi4py import MPI
    except ImportError as err:
        raise ImportError("mpi4py is required for this function!") from err

    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    logger.info(
        f"Simulating joint training dataset ({polyswyftSettings.obsKey}, {polyswyftSettings.targetKey}) using "
        f"deadpoints with "
        f"Simulator!"
    )

    ### simulate joint distribution using deadpoints ###
    if size_gen > 1:
        deadpoints = np.array_split(deadpoints, size_gen, axis=0)
        deadpoints = deadpoints[rank_gen]
    samples = []
    for point in deadpoints:
        cond = {polyswyftSettings.targetKey: point}
        ### noise resampling ###
        if polyswyftSettings.use_noise_resampling and rd > 0:
            resampler = sim.get_resampler(targets=[polyswyftSettings.obsKey])
            for _ in range(polyswyftSettings.n_noise_resampling_samples):
                cond[polyswyftSettings.obsKey] = None
                sample = resampler(cond)
                samples.append(sample)
        else:
            sample = sim.sample(conditions=cond, targets=[polyswyftSettings.obsKey])
            samples.append(sample)

    comm_gen.Barrier()
    if size_gen > 1:
        samples = comm_gen.allgather(samples)
        samples = np.concatenate(samples, axis=0)
        samples = samples.tolist()
    logger.info(f"Total number of samples for training the network: {len(samples)}")
    comm_gen.Barrier()
    samples = reformat_samples(samples)
    logger.info("Simulation done!")
    thetas = torch.empty(size=samples[polyswyftSettings.targetKey].shape)
    Ds = torch.empty(size=samples[polyswyftSettings.obsKey].shape)
    ### shuffle training data to reduce ordinal bias introduced by deadpoints###
    if rank_gen == 0:
        thetas = samples[polyswyftSettings.targetKey]
        Ds = samples[polyswyftSettings.obsKey]
        theta_dim = thetas.shape[1]
        combined = np.concatenate((thetas, Ds), axis=1)
        np.random.shuffle(combined)
        thetas = combined[:, :theta_dim]
        Ds = combined[:, theta_dim:]
    comm_gen.Barrier()
    thetas = comm_gen.bcast(thetas, root=0)
    Ds = comm_gen.bcast(Ds, root=0)
    ### save training data for NRE on disk ###
    if rank_gen == 0:
        np.save(
            arr=thetas,
            file=f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}/{polyswyftSettings.targetKey}.npy",
        )
        np.save(
            arr=Ds, file=f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}/{polyswyftSettings.obsKey}.npy"
        )
    comm_gen.Barrier()
    return thetas, Ds
