import os
import logging
import lsbi.model
###requires lsbi==0.9.0 for reproducibility
import numpy as np
import swyft
import torch
from anesthetic import MCMCSamples
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PolySwyft.PolySwyft_Network import Network
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.Swyft_Simulator_MultiGauss import Simulator as Simulator_swyft
from PolySwyft.PolySwyft_Simulator_MultiGauss import Simulator
import wandb
from pytorch_lightning.loggers import WandbLogger


def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    root = "MVG_Swyft"
    polyswyftSettings = PolySwyft_Settings(root=root)
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    d = polyswyftSettings.num_features_dataset
    n = polyswyftSettings.num_features

    m = torch.randn(d).numpy() * 3  # mean vec of dataset
    M = torch.randn(size=(d, n)).numpy()  # transform matrix of dataset to parameter vee
    C = torch.eye(d).numpy()  # cov matrix of dataset
    # C very small, or Sigma very big
    mu = torch.zeros(n).numpy()  # mean vec of parameter prior
    Sigma = 100 * torch.eye(n).numpy()  # cov matrix of parameter prior
    sim = Simulator(polyswyftSettings=polyswyftSettings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)
    polyswyftSettings.model = sim.model  # lsbi model
    # generate training dat and obs
    obs = swyft.Sample(x=sim.model.evidence().rvs()[None, :])

    os.makedirs(polyswyftSettings.root,exist_ok=True)

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
    #swyft.RectBoundSampler()
    #MVG = MultivariateNormalWithPPF(sim.model.prior())
    #sampler = RectBoundSampler(distr=MVG, bounds=None)
    #sampler()

    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]



    def round(obs, rd, bounds = None, model = None):
        root = f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}"
        if rank_gen==0:
            os.makedirs(root,exist_ok=True)
        comm_gen.Barrier()
        sim = Simulator_swyft(polyswyftSettings=polyswyftSettings, bounds=bounds,model=model)
        shapes, dtypes = sim.get_shapes_and_dtypes()
        #sample training dataset with bounds (here with MPI)
        if rank_gen == 0:
            store = swyft.ZarrStore(f"./{root}/zarr_store")
            if rd == 0:
                N = polyswyftSettings.n_training_samples
            else:
                N = 15_000 #average num samples added by polyswyft
            store.init(N, polyswyftSettings.n_weighted_samples, shapes, dtypes)
        comm_gen.Barrier()
        if rank_gen != 0:
            store = swyft.ZarrStore(f"./{root}/zarr_store")
        comm_gen.Barrier()
        seed_everything(polyswyftSettings.seed + rank_gen, workers=True)
        store.simulate(sim, batch_size = polyswyftSettings.n_weighted_samples)
        seed_everything(polyswyftSettings.seed, workers=True)
        comm_gen.Barrier()

        try:
            polyswyftSettings.wandb_kwargs.pop("finish")
        except KeyError:
            print("Key not found")
        polyswyftSettings.wandb_kwargs["name"] = f"{polyswyftSettings.child_root}_{rd}"
        polyswyftSettings.wandb_kwargs["save_dir"] = root
        wandb_logger = WandbLogger(**polyswyftSettings.wandb_kwargs)

        dm = swyft.SwyftDataModule(store,**polyswyftSettings.dm_kwargs)
        polyswyftSettings.trainer_kwargs['callbacks'] = create_callbacks()
        trainer = swyft.SwyftTrainer(**polyswyftSettings.trainer_kwargs, logger=wandb_logger)
        network = Network(polyswyftSettings=polyswyftSettings, obs=obs)
        trainer.fit(network, dm)
        if rank_gen == 0:
            wandb.finish()
            torch.save(network.state_dict(), f"{root}/{polyswyftSettings.neural_network_file}")
        ### resample prior to construct new bounds on trained NRE
        comm_gen.Barrier()
        seed_everything(polyswyftSettings.seed, workers=True)
        prior_samples = sim.sample(polyswyftSettings.n_training_samples, targets=[polyswyftSettings.targetKey])
        comm_gen.Barrier()
        network.eval()
        obs_torch = swyft.Sample({polyswyftSettings.obsKey:  torch.as_tensor(obs[polyswyftSettings.obsKey])})
        prior_samples_torch = swyft.Samples({polyswyftSettings.targetKey: torch.as_tensor(prior_samples[polyswyftSettings.targetKey]) })
        predictions = network(obs_torch, prior_samples_torch)
        new_bounds = swyft.collect_rect_bounds(predictions, polyswyftSettings.targetKey, (polyswyftSettings.num_features,), threshold = 1e-6)
        if rank_gen == 0:
            torch.save(obj=new_bounds, f=f"{root}/new_bounds.pt")
        return predictions, new_bounds


    comm_gen.Barrier()

    prediction_rounds = []
    bounds_rounds = []

    for rd in range(polyswyftSettings.NRE_start_from_round, polyswyftSettings.NRE_num_retrain_rounds+1):
        if polyswyftSettings.NRE_start_from_round > 0:
            load_root  = f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd-1}"
            bounds = torch.load(f=f"{load_root}/new_bounds.pt")
        else:
            bounds = None
        predictions, bounds = round(obs, bounds = bounds, model= polyswyftSettings.model,rd=rd)
        prediction_rounds.append(predictions)
        bounds_rounds.append(bounds)
        print("New bounds:", bounds)
        comm_gen.Barrier()

if __name__ == '__main__':
    execute()
