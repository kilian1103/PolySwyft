import logging
import pickle
from typing import Callable
from PolySwyft.PolySwyft_Dataloader import PolySwyftDataModule
import pypolychord
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
import torch
import anesthetic
from PolySwyft.utils import compute_KL_divergence, resimulate_deadpoints, select_weighted_contour, reload_data_for_plotting, compute_KL_compression
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
import swyft
import numpy as np
from pypolychord import PolyChordSettings
try:
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py is required for PolySwyft!")
comm_gen = MPI.COMM_WORLD
rank_gen = comm_gen.Get_rank()
size_gen = comm_gen.Get_size()

class PolySwyft:
    def __init__(self, polyswyftSettings: PolySwyft_Settings, sim: swyft.Simulator,
                 obs: swyft.Sample, deadpoints: np.ndarray,
                 network: swyft.SwyftModule, polyset: PolyChordSettings,
                 callbacks: Callable, lr_round_scheduler: Callable = None, deadpoints_processing: Callable = None):
        """
        Initialize the PolySwyft object.
        :param polyswyftSettings: A PolySwyft_Settings object
        :param sim: A swyft simulator object
        :param obs: A swyft sample of the observed data
        :param deadpoints_samplse: An array of the deadpoints samplses
        :param network: A swyft network object
        :param polyset: A PolyChordSettings object
        :param callbacks: A callable object for instantiating the new callbacks of the pl.trainer
        """
        self.polyswyftSettings = polyswyftSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.callbacks = callbacks
        self.lr_round_scheduler = lr_round_scheduler
        self.deadpoints_processing = deadpoints_processing
        self.deadpoints_samples = deadpoints
        self.network_model = network
        self.network_storage = dict()
        self.root_storage = dict()
        self.dkl_storage = dict()
        self.dkl_compression_storage = dict()
        self.samples_storage = dict()
        self.current_key = "current"
        self.previous_key = "previous"

    def execute_NSNRE_cycle(self):
        """
        Execute the sequential nested sampling neural ratio estimation cycle.
        :return:
        """
        self.logger = logging.getLogger(self.polyswyftSettings.logger_name)

        ### create root folder ###
        os.makedirs(self.polyswyftSettings.root,exist_ok=True)

        ### save settings
        with open(f'{self.polyswyftSettings.root}/settings.pkl', 'wb') as file:
            pickle.dump(self.polyswyftSettings, file)

        ### reload data if necessary to resume run ###
        if self.polyswyftSettings.NRE_start_from_round > 0:
            if (self.polyswyftSettings.NRE_start_from_round > self.polyswyftSettings.NRE_num_retrain_rounds and
                    self.polyswyftSettings.cyclic_rounds):
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")

            root = f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_{self.polyswyftSettings.NRE_start_from_round-1}"
            deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
            self.deadpoints_samples = deadpoints.iloc[:, :self.polyswyftSettings.num_features].to_numpy()
            self.network_model.load_state_dict(torch.load(f"{root}/{self.polyswyftSettings.neural_network_file}"))
            self.network_storage[self.previous_key] = self.network_model
            self.samples_storage[self.previous_key] = deadpoints

        ### execute main cycle ###
        if self.polyswyftSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()


    def _cyclic_rounds(self):
        for rd in range(self.polyswyftSettings.NRE_start_from_round, self.polyswyftSettings.NRE_num_retrain_rounds + 1):
            self._cycle(rd)

    def _cyclic_kl(self):
        DKL_info = (100, 100)
        DKL, DKL_std = DKL_info
        rd = self.polyswyftSettings.NRE_start_from_round
        while abs(DKL) >= self.polyswyftSettings.termination_abs_dkl:
            self._cycle(rd)
            DKL, DKL_std = self.dkl_storage[rd]
            rd += 1
        self.polyswyftSettings.NRE_num_retrain_rounds = rd - 1

    def _cycle(self, rd):

        ### start NRE training section ###
        self.logger.info("training network round: " + str(rd))
        root = f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_{rd}"
        ### create root folder ###
        os.makedirs(root,exist_ok=True)

        ### setup wandb ###
        if self.polyswyftSettings.activate_wandb:
            self.polyswyftSettings.wandb_kwargs["name"] = f"{self.polyswyftSettings.child_root}_{rd}"
            self.polyswyftSettings.wandb_kwargs["save_dir"] = f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_{rd}"
            wandb_logger = WandbLogger(**self.polyswyftSettings.wandb_kwargs)
            self.polyswyftSettings.trainer_kwargs["logger"] = wandb_logger

        ### setup trainer ###
        self.polyswyftSettings.trainer_kwargs["default_root_dir"] = root
        self.polyswyftSettings.trainer_kwargs["callbacks"] = self.callbacks()
        trainer = swyft.SwyftTrainer(**self.polyswyftSettings.trainer_kwargs)

        ### setup network and train network###
        if self.polyswyftSettings.continual_learning_mode:
            network = self.network_model
        else:
            network = self.network_model.get_new_network()

        ### continue lr rate at last point
        if self.lr_round_scheduler is not None:
            learning_rate = self.lr_round_scheduler(rd) #between rounds
            self.network_model.optimizer_init.optim_args = dict(lr=learning_rate)

        resimulate_deadpoints(deadpoints=self.deadpoints_samples, polyswyftSettings=self.polyswyftSettings, sim=self.sim, rd=rd)

        dm = PolySwyftDataModule(polyswyftSettings=self.polyswyftSettings,rd=rd,
                                 **self.polyswyftSettings.dm_kwargs)
        network.train()
        trainer.fit(network, dm)
        comm_gen.Barrier()
        if self.polyswyftSettings.activate_wandb and rank_gen == 0:
            wandb.finish()

        ### save network on disk ###
        if rank_gen == 0:
            torch.save(network.state_dict(), f"{root}/{self.polyswyftSettings.neural_network_file}")
            torch.save(network.optimizers().state_dict(), f"{root}/{self.polyswyftSettings.optimizer_file}")

        comm_gen.Barrier()

        ### load network on disk (to sync across nodes) ###
        if self.polyswyftSettings.continual_learning_mode:
            self.network_model.load_state_dict(torch.load(f"{root}/{self.polyswyftSettings.neural_network_file}"))

        ### save network and root in memory###
        comm_gen.Barrier()
        network.eval()

        ### start polychord section ###
        ### run PolyChord ###
        self.logger.info("Using PolyChord with trained NRE to generate deadpoints for the next round!")
        self.polyset.base_dir = root
        comm_gen.barrier()

        pypolychord.run_polychord(loglikelihood=network.logLikelihood,
                                  nDims=self.polyswyftSettings.num_features,
                                  nDerived=self.polyswyftSettings.nderived, settings=self.polyset,
                                  prior=network.prior, dumper=network.dumper)
        comm_gen.Barrier()

        ### load deadpoints and compute KL divergence and reassign to training samples ###
        deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
        comm_gen.Barrier()

        ### polychord round 2 section ###
        if self.polyswyftSettings.use_livepoint_increasing:

            ### choose contour to increase livepoints ###
            index = select_weighted_contour(deadpoints,
                                            threshold=1 - self.polyswyftSettings.livepoint_increase_posterior_contour)
            logL = deadpoints.iloc[index, :].logL

            try:
                os.makedirs(f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}")
            except OSError:
                self.logger.info("root folder already exists!")

            ### run polychord round 2 ###
            self.polyset.base_dir = f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}"
            self.polyset.nlives = {logL: self.polyswyftSettings.n_increased_livepoints}
            comm_gen.Barrier()
            pypolychord.run_polychord(loglikelihood=network.logLikelihood,
                                      nDims=self.polyswyftSettings.num_features,
                                      nDerived=self.polyswyftSettings.nderived, settings=self.polyset,
                                      prior=network.prior, dumper=network.dumper)
            comm_gen.Barrier()
            self.polyset.nlives = {}
            deadpoints = anesthetic.read_chains(
                root=f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}/{self.polyset.file_root}")
            comm_gen.Barrier()

        #### compute KL divergences
        self.root_storage[rd] = root
        self.samples_storage[self.current_key] = deadpoints
        self.network_storage[self.current_key] = network
        #### compute KL(P_i||P_(i-1))
        if rd > 0:
            previous_network = self.network_storage[self.previous_key]
            KDL = compute_KL_divergence(polyswyftSettings=self.polyswyftSettings, previous_network=previous_network.eval(),
                                        current_samples=self.samples_storage[self.current_key], obs=self.obs,
                                        previous_samples=self.samples_storage[self.previous_key])
            self.dkl_storage[rd] = KDL
            self.logger.info(f"Round {rd}: KL(P_i||P_(i-1)) = {KDL[0]} +/- {KDL[1]}")
            print(f"Round {rd}: KL(P_i||P_(i-1)) = {KDL[0]} +/- {KDL[1]}")


        ### compute KL(P_i||Pi)
        DKL = compute_KL_compression(self.samples_storage[self.current_key], self.polyswyftSettings)
        self.dkl_compression_storage[rd] = DKL
        self.logger.info(f"Round {rd}: KL(P_i||Pi) = {DKL[0]} +/- {DKL[1]}")
        print(f"Round {rd}: KL(P_i||Pi) = {DKL[0]} +/- {DKL[1]}")

        self.samples_storage[self.previous_key] = deadpoints
        self.network_storage[self.previous_key] = network

        #### optional deadpoints post processing
        if self.deadpoints_processing is not None:
            deadpoints = self.deadpoints_processing(deadpoints, rd)

        #prepare data for next round
        deadpoints_samples = deadpoints.iloc[:, :self.polyswyftSettings.num_features].to_numpy()
        comm_gen.Barrier()
        self.logger.info(f"Number of deadpoints for next rd {rd + 1}: {deadpoints_samples.shape[0]}")
        self.deadpoints_samples = deadpoints_samples
        return

    def _reload_data(self):
        root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(
            polyswyftSettings=self.polyswyftSettings,
            network=self.network_model,
            polyset=self.polyset,
            until_round=self.polyswyftSettings.NRE_start_from_round - 1,
            only_last_round=True)
        self.root_storage = root_storage
        self.network_storage = network_storage
        self.samples_storage = samples_storage
        self.dkl_storage = dkl_storage

        del self.network_storage[self.polyswyftSettings.NRE_start_from_round - 2]
        del self.samples_storage[self.polyswyftSettings.NRE_start_from_round - 2]
