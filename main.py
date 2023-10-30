import logging

import numpy as np
import pypolychord
import swyft
import torch
from mpi4py import MPI
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
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
    np.random.seed(nreSettings.seed)
    torch.manual_seed(nreSettings.seed)
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    nreSettings.logger = logger
    logger.info('Started')
    # TODO merge prior framework, so far simulator has scipy, polychord has hypercube
    # instantiate swyft simulator
    sim = Simulator(nreSettings=nreSettings)
    # generate training dat aand obs
    obs = swyft.Sample(x=np.array(nreSettings.num_features_dataset * [0]))
    if rank_gen == 0:
        training_samples = torch.as_tensor(
            sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[
                nreSettings.targetKey])
    else:
        training_samples = torch.empty((nreSettings.n_training_samples, nreSettings.num_features))
    # broadcast samples to all ranks
    training_samples = comm_gen.bcast(training_samples, root=0)
    comm_gen.Barrier()
    untrained_network = Network(nreSettings=nreSettings)
    untrained_network_wrapped = NRE_PolyChord(network=untrained_network, obs=obs, nreSettings=nreSettings)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                            patience=nreSettings.early_stopping_patience, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
    trainer = swyft.SwyftTrainer(accelerator=nreSettings.device, devices=1, max_epochs=nreSettings.max_epochs,
                                 precision=64,
                                 enable_progress_bar=True,
                                 default_root_dir=nreSettings.root,
                                 callbacks=[early_stopping_callback, lr_monitor,
                                            checkpoint_callback])
    polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
    polyset.file_root = nreSettings.file_root
    polyset.base_dir = nreSettings.root
    polyset.seed = nreSettings.seed
    polyset.nfail = nreSettings.nlive_scan_run_per_feature * nreSettings.n_training_samples
    polyset.nprior = nreSettings.n_training_samples
    polyset.nlive = nreSettings.nlive_scan_run_per_feature * nreSettings.num_features
    polySwyft = PolySwyft(nreSettings=nreSettings, sim=sim, obs=obs, training_samples=training_samples,
                          untrained_network_wrapped=untrained_network_wrapped, trainer=trainer)
    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()
    else:
        # load data for plotting if data is already generated
        root_storage, network_storage = reload_data_for_plotting(nreSettings=nreSettings)
        polySwyft.root_storage = root_storage
        polySwyft.network_storage = network_storage

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=polySwyft.network_storage,
                               root_storage=polySwyft.root_storage,
                               sim=sim, obs=obs)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
