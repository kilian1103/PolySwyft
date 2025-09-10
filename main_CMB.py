import logging
import sys

import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import pypolychord
import swyft
import torch
from cmblike.cmb import CMB
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from PolySwyft.PolySwyft import PolySwyft
from PolySwyft.PolySwyft_Network_CMB import Network
from PolySwyft.PolySwyft_Post_Analysis import plot_analysis_of_NSNRE
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_CMB import Simulator
from PolySwyft.utils import reload_data_for_plotting


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    n_lr = int(sys.argv[1])
    # n_lr = 5
    # n_lr = 1
    lrs = {0: 1.00,
           1: 0.99,
           2: 0.98,
           3: 0.97,
           4: 0.96,
           5: 0.95}

    root = f"CMB_PolySwyft_lr{lrs[n_lr]}"
    polyswyftSettings = PolySwyft_Settings(root)
    polyswyftSettings.learning_rate_decay = 0.95
    # polyswyftSettings.num_summary_features = nsum
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    # cosmopower params
    cp = True
    params = ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']  # cosmopower
    polyswyftSettings.num_features = len(params)
    prior_mins = [0.005, 0.08, 0.01, 0.8, 2.6, 0.5]
    prior_maxs = [0.04, 0.21, 0.16, 1.2, 3.8, 0.9]
    cmbs = CMB(path_to_cp="../cosmopower", parameters=params, prior_maxs=prior_maxs, prior_mins=prior_mins)

    # prepare binning
    l_max = 2508  # read from planck unbinned data
    first_bin_width = 1
    second_bin_width = 30
    divider = 30
    # bins = np.array([np.arange(2, l_max, 1), np.arange(2, l_max, 1)]).T  # 2 to 2508 unbinned
    first_bins = np.array([np.arange(2, divider, first_bin_width), np.arange(2, divider, first_bin_width)]).T  # 2 to 29
    # Correcting bin generation to be non-overlapping [start, end)
    l_starts = np.arange(divider, l_max, second_bin_width)
    l_ends = np.arange(divider + second_bin_width, l_max + second_bin_width, second_bin_width)
    l_ends = np.clip(l_ends, a_min=None, a_max=l_max)  # Ensure last bin doesn't exceed l_max
    second_bins = np.array([l_starts, l_ends[:len(l_starts)]]).T
    bins = np.concatenate([first_bins, second_bins])
    bins[:, 1] += 1
    # bin_centers = bins[:, 0]
    bin_centers = np.concatenate([first_bins[:, 0], np.mean(bins[divider - 2:], axis=1)])
    l = bin_centers.copy()

    # planck noise
    # pnoise, _ = planck_noise().calculate_noise()
    pnoise = None

    # binned planck data, not using real data for now
    # planck = np.loadtxt('data/planck_unbinned.txt', usecols=[1])
    # planck = cmbs.rebin(planck, bins=bins)
    sim = Simulator(polyswyftSettings=polyswyftSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers, p_noise=pnoise,
                    cp=cp)
    # obs = swyft.Sample(x=torch.as_tensor(planck)[None, :])

    # ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']
    theta_true = np.array([0.022, 0.12, 0.055, 0.965, 3.0, 0.67])
    sample_true = sim.sample(conditions={polyswyftSettings.targetKey: theta_true})
    obs = swyft.Sample(x=torch.as_tensor(sample_true[polyswyftSettings.obsKey])[None, :])
    polyswyftSettings.num_features_dataset = obs[polyswyftSettings.obsKey].shape[1]
    # generate dead points
    n_per_core = polyswyftSettings.n_training_samples // size_gen
    seed_everything(polyswyftSettings.seed + rank_gen, workers=True)
    if rank_gen == 0:
        n_per_core += polyswyftSettings.n_training_samples % size_gen
    joints = sim.sample(n_per_core, targets=[polyswyftSettings.obsKey])
    deadpoints = joints[polyswyftSettings.targetKey]
    joints = joints[polyswyftSettings.obsKey]

    comm_gen.Barrier()
    seed_everything(polyswyftSettings.seed, workers=True)
    deadpoints = comm_gen.allgather(deadpoints)
    deadpoints = np.concatenate(deadpoints, axis=0)
    comm_gen.Barrier()

    # preprocess data: log transform and noise normalization
    joints = comm_gen.allgather(joints)
    joints = np.concatenate(joints, axis=0)
    comm_gen.Barrier()
    joints = joints / obs[polyswyftSettings.obsKey][0].numpy()
    log_data_mean = np.mean(np.log10(joints), axis=0)
    log_data_std = np.std(np.log10(joints), axis=0)
    polyswyftSettings.log_data_mean = log_data_mean
    polyswyftSettings.log_data_std = log_data_std
    comm_gen.Barrier()

    polyswyftSettings.wandb_project_name = polyswyftSettings.root
    polyswyftSettings.wandb_kwargs["project"] = polyswyftSettings.wandb_project_name

    network = Network(polyswyftSettings=polyswyftSettings, obs=obs, cmbs=cmbs)

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    def lr_round_scheduler(rd: int) -> float:
        lr = polyswyftSettings.learning_rate_init * (
                    polyswyftSettings.learning_rate_decay ** (polyswyftSettings.early_stopping_patience * rd))
        return lr

    def compress_deadpoints(deadpoints: anesthetic.NestedSamples, rd: int) -> anesthetic.NestedSamples:
        return deadpoints.posterior_points()

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(polyswyftSettings.num_features, nDerived=polyswyftSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = polyswyftSettings.root
    polyset.seed = polyswyftSettings.seed
    polyset.nfail = polyswyftSettings.n_training_samples
    polyset.nlive = polyswyftSettings.num_features * 100
    polySwyft = PolySwyft(polyswyftSettings=polyswyftSettings, sim=sim, obs=obs, deadpoints=deadpoints,
                          network=network, polyset=polyset, callbacks=create_callbacks,
                          lr_round_scheduler=lr_round_scheduler)

    if not polyswyftSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()

    # plot observation
    if rank_gen == 0:
        plt.plot(l, obs[polyswyftSettings.obsKey][0].numpy(), label='Sim. Observation', c="red")
        # for i in range(0, joints.shape[0], 1000):
        # plt.plot(l, joints[i], color='gray', alpha=0.1, label='Sim. Samples')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r"$D_\ell$")
        plt.savefig(f"{polyswyftSettings.root}/obs.pdf", bbox_inches='tight')

    root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(
        polyswyftSettings=polyswyftSettings,
        network=network,
        polyset=polyset,
        until_round=polyswyftSettings.NRE_num_retrain_rounds)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(polyswyftSettings=polyswyftSettings, network_storage=network_storage,
                               samples_storage=samples_storage, dkl_storage=dkl_storage,
                               obs=obs, root=polyswyftSettings.root)
    logger.info('Finished')


if __name__ == '__main__':
    main()
