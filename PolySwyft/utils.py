import os
from typing import Dict
from typing import Tuple
import sklearn
import anesthetic
import numpy as np
import pandas as pd
import swyft
import torch
from anesthetic import NestedSamples
from pypolychord import PolyChordSettings
from scipy.special import logsumexp
from swyft import collate_output as reformat_samples, Simulator
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


def select_weighted_contour(data: NestedSamples, threshold: float) -> int:
    """find the index of the posterior sample that corresponds iso-contour threshold.
    :param data: An anesthetic NestedSamples object
    :param threshold: A float between 0 and 1
    :return: An integer index
    """
    cumulative_weights = data.get_weights().cumsum()
    cumulative_weights_norm = cumulative_weights / cumulative_weights[-1]
    index = np.searchsorted(cumulative_weights_norm, threshold)
    return index


def compute_KL_divergence(polyswyftSettings: PolySwyft_Settings, previous_network: swyft.SwyftModule,
                          current_samples: anesthetic.Samples, previous_samples: anesthetic.Samples,
                          obs: swyft.Sample) -> Tuple[float, float]:
    """
    Compute the KL divergence between the previous NRE and the current NRE KL(P_{i}||P_{i-i}).
    :param polyswyftSettings: A PolySwyft_Settings object
    :param previous_network: A swyft network object
    :param current_samples: An anesthetic samples object of the current samples
    :param previous_samples: An anesthetic samples object of the previous samples
    :param obs: A swyft sample of the observed data
    :return: A tuple of the KL divergence and the error
    """

    samples = {polyswyftSettings.targetKey: torch.as_tensor(current_samples.iloc[:, :polyswyftSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, samples)
    current_samples["logL_previous"] = predictions.logratios.numpy().squeeze()

    logw = current_samples.logw(polyswyftSettings.n_DKL_estimates)
    logpqs = (current_samples["logL"].values[:, None] - current_samples.logZ(logw).values - current_samples[
                                                                                                "logL_previous"].values[
                                                                                            :,
                                                                                            None] +
              previous_samples.logZ(
                  polyswyftSettings.n_DKL_estimates).values)
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()

    return DKL, DKL_err


def compute_KL_divergence_truth(polyswyftSettings: PolySwyft_Settings, previous_network: swyft.SwyftModule,
                                true_posterior: anesthetic.Samples, previous_samples: anesthetic.Samples,
                                obs: swyft.Sample) -> Tuple[float, float]:
    """Compute the KL divergence between the previous NRE and the true posterior KL(P_{true}||P_{i}).
    :param polyswyftSettings: A PolySwyft_Settings object
    :param previous_network: A swyft network object
    :param true_posterior: An anesthetic samples object of the true posterior
    :param previous_samples: An anesthetic samples object of the previous samples
    :param obs: A swyft sample of the observed data
    :return: A tuple of the KL divergence and the error
    """
    swyft_samples = {
        polyswyftSettings.targetKey: torch.as_tensor(true_posterior.iloc[:, :polyswyftSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, swyft_samples)
    true_posterior["logL_previous"] = predictions.logratios.numpy().squeeze()
    # MCMC samples for true samples do not have logw functionality
    samples = true_posterior.iloc[:, :polyswyftSettings.num_features].squeeze()
    true_posterior_logL = polyswyftSettings.model.posterior(obs[polyswyftSettings.obsKey].numpy().squeeze()).logpdf(samples)
    true_prior = polyswyftSettings.model.prior().logpdf(samples)
    true_posterior.logL = true_posterior_logL
    true_posterior["logR"] = true_posterior["logL_previous"]
    logpqs = (true_posterior["logL"].values[:, None] - true_posterior["logR"].values[:, None] - true_prior[:,
                                                                                                None] +
              previous_samples.logZ(
                  polyswyftSettings.n_DKL_estimates).values)
    DKL_estimates = logpqs.mean(axis=0)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return (DKL, DKL_err)


def compute_KL_compression(samples: anesthetic.NestedSamples, polyswyftSettings: PolySwyft_Settings):
    """
    Compute the KL compression of the samples, Prior to Posterior, KL(P||pi).
    :param samples: An anesthetic NestedSamples object
    :param polyswyftSettings: A PolySwyft_Settings object
    :return: A tuple of the KL compression and the error
    """
    logw = samples.logw(polyswyftSettings.n_DKL_estimates)
    logpqs = samples["logL"].values[:, None] - samples.logZ(logw).values
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return DKL, DKL_err


def reload_data_for_plotting(polyswyftSettings: PolySwyft_Settings, network: swyft.SwyftModule, polyset: PolyChordSettings,
                             until_round: int, only_last_round=False) -> \
        Tuple[
            Dict[int, str], Dict[int, swyft.SwyftModule], Dict[int, anesthetic.NestedSamples], Dict[
                int, Tuple[float, float]]]:
    """
    Reload the data for plotting.
    :param polyswyftSettings: A PolySwyft_Settings object
    :param network: A swyft network object
    :param polyset: A PolyChordSettings object
    :param until_round: An integer of the number of rounds to reload (inclusive)
    :param only_last_round: A boolean to only reload the last round until_round
    :return: A tuple of dictionaries of root_storage, network_storage, samples_storage, and dkl_storage
    """

    network_storage = {}
    root_storage = {}
    samples_storage = {}
    dkl_storage = {}
    root = polyswyftSettings.root

    try:
        obs = network.obs
    except AttributeError:
        raise AttributeError("network object does not have an attribute 'obs'")

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
        params = [fr"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)]
        if polyswyftSettings.use_livepoint_increasing:
            samples = anesthetic.read_chains(
                root=f"{root_storage[rd]}/{polyswyftSettings.increased_livepoints_fileroot}/{polyset.file_root}")
        else:
            samples = anesthetic.read_chains(root=f"{root_storage[rd]}/{polyset.file_root}")
        labels = samples.get_labels()
        labels[:polyswyftSettings.num_features] = params
        samples.set_labels(labels, inplace=True)
        samples_storage[rd] = samples

        # compute DKL
        if rd > 0:
            if only_last_round and rd < until_round:
                continue
            previous_network = network_storage[rd - 1]
            KDL = compute_KL_divergence(polyswyftSettings=polyswyftSettings, previous_network=previous_network.eval(),
                                        current_samples=samples_storage[rd], obs=obs,
                                        previous_samples=samples_storage[rd - 1])
            dkl_storage[rd] = KDL
    return root_storage, network_storage, samples_storage, dkl_storage


def random_subset_after_truncation(deadpoints: anesthetic.NestedSamples, logR_cutoff: float,
                                   p: float) -> anesthetic.NestedSamples:
    rest = deadpoints[deadpoints.logL >= logR_cutoff]
    bools = np.random.choice([True, False], size=rest.shape[0], p=[p, 1 - p])
    rest = rest[bools]
    deadpoints = pd.concat([deadpoints, rest], axis=0)
    deadpoints.drop_duplicates(inplace=True)
    return deadpoints


def delete_previous_joint_training_data(until_rd: int, root: str, nreSettings: PolySwyft_Settings):
    """
    Delete previous joint training data.
    :param until_rd: An integer of the round number to delete until
    :param root: A string of the root directory
    """
    for rd in range(until_rd):
        try:
            os.remove(f"{root}/{nreSettings.child_root}_{rd}/{nreSettings.joint_training_data_fileroot}")
        except FileNotFoundError:
            pass
    return



def resimulate_deadpoints(deadpoints: np.ndarray, polyswyftSettings: PolySwyft_Settings,
                          sim: Simulator,rd: int):
    """
    Retrain the network for the next round of NSNRE.
    :param root: A string of the root folder
    :param deadpoints: A tensor of deadpoints
    :param polyswyftSettings: A PolySwyft_Settings object
    :param sim: A swyft simulator object
    :param obs: A swyft sample of the observed data
    :param network: A swyft network object
    :param trainer: A swyft trainer object
    :param rd: An integer of the round number
    :return: A trained swyft network object
    """
    logger = polyswyftSettings.logger
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py is required for this function!")

    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    logger.info(
        f"Simulating joint training dataset ({polyswyftSettings.obsKey}, {polyswyftSettings.targetKey}) using deadpoints with "
        f"Simulator!")

    ### simulate joint distribution using deadpoints ###
    deadpoints = np.array_split(deadpoints, size_gen)
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

    del deadpoints
    comm_gen.Barrier()
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
        joint =  np.array(sklearn.utils.shuffle(list(zip(thetas,Ds))),dtype=object)
        thetas = np.stack(joint[:, 0])
        Ds = np.stack(joint[:, 1])
    comm_gen.Barrier()
    thetas = comm_gen.bcast(thetas, root=0)
    Ds = comm_gen.bcast(Ds, root=0)
    ### save training data for NRE on disk ###
    if rank_gen == 0:
        np.save(arr=thetas, file=f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}/thetas.npy")
        np.save(arr=Ds, file=f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}/Ds.npy")
    comm_gen.Barrier()
