import logging

import pypolychord
import swyft
import torch
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Post_Analysis import plot_analysis_of_NSNRE
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator_MultiGauss import Simulator
from NSLFI.PolySwyft import PolySwyft
from NSLFI.utils import reload_data_for_plotting


def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    nreSettings = NRE_Settings()
    seed_everything(nreSettings.seed)
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    nreSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    d = nreSettings.num_features_dataset
    n = nreSettings.num_features

    m = torch.randn(d) * 3  # mean vec of dataset
    M = torch.randn(size=(d, n))  # transform matrix of dataset to parameter vee
    C = torch.eye(d)  # cov matrix of dataset
    # C very small, or Sigma very big
    mu = torch.zeros(n)  # mean vec of parameter prior
    Sigma = 100 * torch.eye(n)  # cov matrix of parameter prior
    sim = Simulator(nreSettings=nreSettings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)
    nreSettings.model = sim.model  # lsbi model

    # generate training dat and obs
    obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))
    if rank_gen == 0:
        training_samples = torch.as_tensor(
            sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[
                nreSettings.targetKey])
    else:
        training_samples = torch.empty((nreSettings.n_training_samples, nreSettings.num_features))
    # broadcast samples to all ranks
    training_samples = comm_gen.bcast(training_samples, root=0)
    comm_gen.Barrier()

    #### instantiate swyft network
    network = Network(nreSettings=nreSettings, obs=obs)

    #### create callbacks function for pytorch lightning traineer
    def create_callbacks():
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=nreSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = nreSettings.root
    polyset.seed = nreSettings.seed
    polyset.nfail = nreSettings.nlives_per_dim_constant * nreSettings.n_prior_sampling
    polyset.nprior = nreSettings.n_prior_sampling
    polySwyft = PolySwyft(nreSettings=nreSettings, sim=sim, obs=obs, training_samples=training_samples,
                          network=network, polyset=polyset, callbacks=create_callbacks)

    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()
    else:
        # load data for plotting if data is already generated
        root_storage, network_storage = reload_data_for_plotting(nreSettings=nreSettings, network=network)
        polySwyft.root_storage = root_storage
        polySwyft.network_storage = network_storage

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=polySwyft.network_storage,
                               root_storage=polySwyft.root_storage,
                               sim=sim, obs=obs, polyset=polyset)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
