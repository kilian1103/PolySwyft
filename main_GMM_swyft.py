import logging
from scipy.stats import multivariate_normal
import numpy as np
import swyft
import torch
from anesthetic import MCMCSamples
from mpi4py import MPI
import wandb
from pytorch_lightning.loggers import WandbLogger
from lsbi.model import MixtureModel
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import wishart
from PolySwyft.PolySwyft_Network import Network
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_MixGaussMultiPost import Simulator

###requires lsbi==0.12.0 for reproducibility
def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    root = "GMM_Swyft"
    polyswyftSettings = PolySwyft_Settings(root=root)
    polyswyftSettings.seed = 250
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    n = polyswyftSettings.num_features
    d = polyswyftSettings.num_features_dataset
    a = polyswyftSettings.num_mixture_components
    polyswyftSettings.n_training_samples = 160_588 #sum needed for convergence in polyswyft
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
    polyswyftSettings.model = sim.model  # lsbi mode

    # generate training dat and obs
    obs = swyft.Sample(x=sim.model.evidence().rvs()[None, :])

    root = polyswyftSettings.root
    shapes, dtypes = sim.get_shapes_and_dtypes()
    if rank_gen == 0:
        store = swyft.ZarrStore(f"./{root}/zarr_store")
        store.init(polyswyftSettings.n_training_samples, polyswyftSettings.n_weighted_samples, shapes, dtypes)
    comm_gen.Barrier()
    if rank_gen != 0:
        store = swyft.ZarrStore(f"./{root}/zarr_store")
    comm_gen.Barrier()
    seed_everything(polyswyftSettings.seed + rank_gen, workers=True)
    store.simulate(sim, batch_size = polyswyftSettings.n_weighted_samples)
    seed_everything(polyswyftSettings.seed, workers=True)

    ### generate true posterior for comparison
    cond = {polyswyftSettings.obsKey: obs[polyswyftSettings.obsKey].squeeze()}
    full_joint = sim.sample(polyswyftSettings.n_weighted_samples, conditions=cond)
    true_logratios = torch.as_tensor(full_joint[polyswyftSettings.contourKey])
    posterior = full_joint[polyswyftSettings.posteriorsKey]
    weights = np.ones(shape=len(posterior))  # direct samples from posterior have weights 1
    params_labels = {i: rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)}
    mcmc_true = MCMCSamples(
        data=posterior, weights=weights.squeeze(),
        logL=true_logratios, labels=params_labels)

    #### instantiate swyft networ
    network = Network(polyswyftSettings=polyswyftSettings, obs=obs)

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    try:
        polyswyftSettings.wandb_kwargs.pop("finish")
    except KeyError:
        print("Key not found")

    rd=0 #cant run truncation scheme
    polyswyftSettings.wandb_kwargs["name"] = f"{polyswyftSettings.child_root}_{rd}"
    polyswyftSettings.wandb_kwargs["save_dir"] = f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}"
    wandb_logger = WandbLogger(**polyswyftSettings.wandb_kwargs)


    dm = swyft.SwyftDataModule(data=store, **polyswyftSettings.dm_kwargs)
    polyswyftSettings.trainer_kwargs['callbacks'] = create_callbacks()
    trainer = swyft.SwyftTrainer(**polyswyftSettings.trainer_kwargs, logger=wandb_logger)
    trainer.fit(model=network, datamodule=dm)
    if rank_gen == 0:
        wandb.finish()
        torch.save(network.state_dict(), f"{root}/{polyswyftSettings.neural_network_file}")


if __name__ == '__main__':
    execute()
