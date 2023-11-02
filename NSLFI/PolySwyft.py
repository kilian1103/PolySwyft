import logging

import anesthetic
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI
from pypolychord import PolyChordSettings

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import compute_KL_divergence


class PolySwyft:
    def __init__(self, nreSettings: NRE_Settings, sim: swyft.Simulator,
                 obs: swyft.Sample, training_samples: torch.Tensor,
                 network_wrapped: NRE_PolyChord, trainer: swyft.SwyftTrainer, polyset: PolyChordSettings,
                 dm: swyft.SwyftDataModule):
        self.nreSettings = nreSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.training_samples = training_samples
        self.network_wrapped = network_wrapped
        self.trainer = trainer
        self.dm = dm
        self.network_storage = dict()
        self.root_storage = dict()
        self.dkl_storage = list()

    def execute_NSNRE_cycle(self):
        # retrain NRE and sample new samples with NS loop
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()
        logger = logging.getLogger(self.nreSettings.logger_name)
        if self.nreSettings.NRE_start_from_round > 0:
            if self.nreSettings.NRE_start_from_round > self.nreSettings.NRE_num_retrain_rounds:
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")
            ### only execute this code when previous rounds are already trained ###
            for i in range(0, self.nreSettings.NRE_start_from_round):
                root = f"{self.nreSettings.root}_round_{i}"
                network = self.network_wrapped.get_new_network()
                network.load_state_dict(torch.load(f"{root}/{self.nreSettings.neural_network_file}"))
                network.double()  # change to float64 precision of network
                self.network_storage[f"round_{i}"] = network
                self.root_storage[f"round_{i}"] = root
                if i > 0:
                    deadpoints = anesthetic.read_chains(root=f"{root}/{self.nreSettings.file_root}")
                    DKL = compute_KL_divergence(nreSettings=self.nreSettings, network_storage=self.network_storage,
                                                current_samples=deadpoints.copy(), rd=i, obs=self.obs)
                    self.dkl_storage.append(DKL)
            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            self.training_samples = deadpoints

        ### main cycle ###
        for rd in range(self.nreSettings.NRE_start_from_round, self.nreSettings.NRE_num_retrain_rounds + 1):
            ### start NRE training section ###
            root = f"{self.nreSettings.root}_round_{rd}"
            if rank_gen == 0:
                logger.info("retraining round: " + str(rd))
                if self.nreSettings.activate_wandb:
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project=self.nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
                network = retrain_next_round(root=root, training_data=self.training_samples,
                                             nreSettings=self.nreSettings, sim=self.sim, obs=self.obs,
                                             untrained_network=self.network_wrapped.network,
                                             trainer=self.trainer, dm=self.dm)
                self.trainer.reset_train_dataloader()
                self.trainer.reset_val_dataloader()
                self.trainer.reset_test_dataloader()
            else:
                network = self.network_wrapped.get_new_network()
            comm_gen.Barrier()
            ### load saved network and save it in network_storage ###
            network.load_state_dict(torch.load(f"{root}/{self.nreSettings.neural_network_file}"))
            network.double()  # change to float64 precision of network
            self.network_storage[f"round_{rd}"] = network
            self.root_storage[f"round_{rd}"] = root
            logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")

            ### start polychord section ###
            ### Run PolyChord ###
            self.polyset.base_dir = root
            self.network_wrapped.set_network(network=network)
            pypolychord.run_polychord(loglikelihood=self.network_wrapped.logLikelihood,
                                      nDims=self.nreSettings.num_features,
                                      nDerived=self.nreSettings.nderived, settings=self.polyset,
                                      prior=self.network_wrapped.prior, dumper=self.network_wrapped.dumper)
            comm_gen.Barrier()

            ### load deadpoints and compute KL divergence and reassign to training samples ###
            deadpoints = anesthetic.read_chains(root=f"{root}/{self.nreSettings.file_root}")
            if rd >= 1:
                DKL = compute_KL_divergence(nreSettings=self.nreSettings, network_storage=self.network_storage,
                                            current_samples=deadpoints, rd=rd, obs=self.obs)
                self.dkl_storage.append(DKL)
                logger.info(f"DKL of rd {rd} is: {DKL}")
            comm_gen.Barrier()
            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
            self.training_samples = deadpoints
            network = self.network_wrapped.get_new_network()
            self.network_wrapped.set_network(network=network)
