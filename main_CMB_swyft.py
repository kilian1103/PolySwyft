import wandb
from pytorch_lightning.loggers import WandbLogger
import logging
from cmblike.cmb import CMB
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PolySwyft.PolySwyft_Network_CMB import Network
from PolySwyft.Swyft_Simulator_CMB import Simulator as Simulator_swyft
#from PolySwyft.utils import *
import swyft
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
import numpy as np
import torch
import os

def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    root = "CMB_Swyft"
    polyswyftSettings = PolySwyft_Settings(root=root)
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="w")
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
    #bins = np.array([np.arange(2, l_max, 1), np.arange(2, l_max, 1)]).T  # 2 to 2508 unbinned
    first_bins = np.array([np.arange(2, divider, first_bin_width), np.arange(2, divider, first_bin_width)]).T  # 2 to 29
    second_bins = np.array([np.arange(divider, l_max - second_bin_width, second_bin_width),
                            np.arange(divider + second_bin_width, l_max, second_bin_width)]).T  # 30 to 2508
    last_bin = np.array([[second_bins[-1, 1], l_max]])  # remainder
    bins = np.concatenate([first_bins, second_bins, last_bin])
    #bin_centers = bins[:, 0]
    bin_centers = np.concatenate([first_bins[:, 0], np.mean(bins[divider - 2:], axis=1)])
    l = bin_centers.copy()
    polyswyftSettings.num_features_dataset = len(l)


    # planck noise
    #pnoise, _ = planck_noise().calculate_noise()
    pnoise = None

    # binned planck data, not using real data for now
    #planck = np.loadtxt('data/planck_unbinned.txt', usecols=[1])
    #planck = cmbs.rebin(planck, bins=bins)
    sim = Simulator_swyft(polyswyftSettings=polyswyftSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers,
                          p_noise=pnoise, cp=cp, prior_mins=np.array(prior_mins), prior_maxs=np.array(prior_maxs))
    # obs = swyft.Sample(x=torch.as_tensor(planck)[None, :])

    # ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']
    theta_true = np.array([0.022, 0.12, 0.055, 0.965, 3.0, 0.67])
    sample_true = sim.sample(conditions={polyswyftSettings.targetKey: theta_true})
    obs = swyft.Sample(x=torch.as_tensor(sample_true[polyswyftSettings.obsKey])[None, :])
    torch.save(torch.as_tensor(sample_true[polyswyftSettings.obsKey])[None, :], f"{polyswyftSettings.root}/obs.pt")

    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                    patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    def round(obs, polyswyftSettings, cmbs, bins, bin_centers, p_noise, cp, prior_mins, prior_maxs, rd, bounds=None):
        root = f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd}"
        if rank_gen==0:
            os.makedirs(root,exist_ok=True)
        comm_gen.Barrier()
        sim = Simulator_swyft(polyswyftSettings=polyswyftSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers,
                              p_noise=p_noise, cp=cp, bounds=bounds, prior_mins=np.array(prior_mins), prior_maxs=np.array(prior_maxs))
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
        network = Network(polyswyftSettings=polyswyftSettings, obs=obs, cmbs=cmbs)
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
        predictions, bounds = round(obs, polyswyftSettings, cmbs, bins, bin_centers, pnoise, cp,
                                    prior_mins, prior_maxs, rd, bounds=bounds)
        prediction_rounds.append(predictions)
        bounds_rounds.append(bounds)
        print("New bounds:", bounds)
        comm_gen.Barrier()





if __name__ == '__main__':
    main()
