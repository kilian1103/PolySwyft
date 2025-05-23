import gc
import logging
from scipy.stats import wishart
import anesthetic
from lsbi.model import MixtureModel
import numpy as np
import pypolychord
import swyft
import torch
from anesthetic import MCMCSamples
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from PolySwyft.PolySwyft_Network import Network
from PolySwyft.PolySwyft_Post_Analysis import plot_analysis_of_NSNRE
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_MixGaussMultiPost import Simulator
from PolySwyft.PolySwyft import PolySwyft
from PolySwyft.utils import reload_data_for_plotting

###requires lsbi==0.12.0 for reproducibility
def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    polyswyftSettings = PolySwyft_Settings()
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    n = polyswyftSettings.num_features
    d = polyswyftSettings.num_features_dataset
    a = polyswyftSettings.num_mixture_components

    #prior
    mu_theta = 5*np.random.randn(a,n)
    Sigma = 0.1*np.eye(n)
    Sigma = wishart.rvs(df=n+1, scale=Sigma, size=a)
    #component weights
    logA = np.random.uniform(size=a)
    logA = np.log(logA / np.sum(logA))
    #likelihood
    mu_data = np.zeros(shape=(a, d))
    M = 0.04*np.random.randn(a, d, n)
    C = 4*np.eye(d)

    model = MixtureModel(M=M, C=C, Sigma=Sigma, mu=mu_theta,
                         m=mu_data, logw=logA, n=n, d=d)
    sim = Simulator(polyswyftSettings=polyswyftSettings, model=model)
    polyswyftSettings.model = sim.model  # lsbi model

    # generate training dat and obs
    theta = sim.model.prior().rvs()
    obs = swyft.Sample(x=torch.tensor(sim.model.likelihood(theta).rvs()[None, :]))
    #obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))
    n_per_core = polyswyftSettings.n_training_samples // size_gen
    seed_everything(polyswyftSettings.seed + rank_gen, workers=True)
    if rank_gen == 0:
        n_per_core += polyswyftSettings.n_training_samples % size_gen
    deadpoints = sim.sample(n_per_core, targets=[polyswyftSettings.targetKey])[
        polyswyftSettings.targetKey]
    comm_gen.Barrier()
    seed_everything(polyswyftSettings.seed, workers=True)
    deadpoints = comm_gen.allgather(deadpoints)
    deadpoints = np.concatenate(deadpoints, axis=0)
    comm_gen.Barrier()

    ### generate true posterior for comparison
    cond = {polyswyftSettings.obsKey: obs[polyswyftSettings.obsKey].numpy().squeeze()}
    full_joint = sim.sample(polyswyftSettings.n_weighted_samples, conditions=cond)
    true_logratios = torch.as_tensor(full_joint[polyswyftSettings.contourKey])
    posterior = full_joint[polyswyftSettings.posteriorsKey]
    weights = np.ones(shape=len(posterior))  # direct samples from posterior have weights 1
    params_labels = {i: rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)}
    mcmc_true = MCMCSamples(data=posterior, weights=weights.squeeze(),
        logL=true_logratios, labels=params_labels)

    #### instantiate swyft network
    network = Network(polyswyftSettings=polyswyftSettings, obs=obs)

    def deadpoints_compress(deadpoints: anesthetic.NestedSamples, rd:int):
        return deadpoints.compress()

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]


    def lr_round_scheduler(rd: int)-> float:
        lr = polyswyftSettings.learning_rate_init * (polyswyftSettings.learning_rate_decay ** (polyswyftSettings.early_stopping_patience *rd))
        return lr

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(polyswyftSettings.num_features, nDerived=polyswyftSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = polyswyftSettings.root
    polyset.seed = polyswyftSettings.seed
    polyset.nfail = polyswyftSettings.n_training_samples
    polyset.nlive = polyswyftSettings.num_features * 500
    polySwyft = PolySwyft(polyswyftSettings=polyswyftSettings, sim=sim, obs=obs, deadpoints=deadpoints,
                          network=network, polyset=polyset, callbacks=create_callbacks, lr_round_scheduler=lr_round_scheduler)
    if not polyswyftSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()

    root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(
        polyswyftSettings=polyswyftSettings,
        network=network,
        polyset=polyset,
        until_round=polyswyftSettings.NRE_num_retrain_rounds)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(polyswyftSettings=polyswyftSettings, network_storage=network_storage,
                               samples_storage=samples_storage, dkl_storage=dkl_storage,
                               obs=obs, true_posterior=mcmc_true, root=polyswyftSettings.root)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
