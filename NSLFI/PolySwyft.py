import copy
import logging

import anesthetic
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI
from pypolychord import PolyChordSettings

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import compute_KL_divergence


class PolySwyft:
    def __init__(self, nreSettings: NRE_Settings, sim: swyft.Simulator,
                 obs: swyft.Sample, training_samples: torch.Tensor,
                 network: swyft.SwyftModule, polyset: PolyChordSettings, dm: swyft.SwyftDataModule,
                 trainer: swyft.SwyftTrainer):
        self.nreSettings = nreSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.dm = dm
        self.trainer = trainer
        self.training_samples = training_samples
        self.network = network
        self.network_storage = dict()
        self.root_storage = dict()
        self.dkl_storage = list()

    def execute_NSNRE_cycle(self):
        # retrain NRE and sample new samples with NS loop
        self.logger = logging.getLogger(self.nreSettings.logger_name)
        if self.nreSettings.NRE_start_from_round > 0:
            if (self.nreSettings.NRE_start_from_round > self.nreSettings.NRE_num_retrain_rounds and
                    self.nreSettings.cyclic_rounds):
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")
            ### only execute this code when previous rounds are already trained ###
            for i in range(0, self.nreSettings.NRE_start_from_round):
                root = f"{self.nreSettings.root}_round_{i}"
                new_network = self.network.get_new_network()
                new_network.load_state_dict(torch.load(f"{root}/{self.nreSettings.neural_network_file}"))
                new_network.double()  # change to float64 precision of network
                self.network_storage[f"round_{i}"] = new_network
                self.root_storage[f"round_{i}"] = root
                deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
                if i > 0:
                    previous_network = self.network_storage[f"round_{i - 1}"]
                    DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                                current_samples=deadpoints.copy(), obs=self.obs)
                    self.dkl_storage.append(DKL)

            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            self.training_samples = deadpoints

        ### main cycle ###
        if self.nreSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()

    def _cyclic_rounds(self):
        DKL = 10
        for rd in range(self.nreSettings.NRE_start_from_round, self.nreSettings.NRE_num_retrain_rounds + 1):
            _ = self._cycle(DKL, rd)

    def _cyclic_kl(self):
        DKL_info = (10, 10)
        DKL, DKL_std = DKL_info
        rd = self.nreSettings.NRE_start_from_round
        while abs(DKL) >= self.nreSettings.termination_abs_dkl:
            DKL, DKL_std = self._cycle(DKL_info, rd)
            rd += 1
        self.nreSettings.NRE_num_retrain_rounds = rd - 1

    def _cycle(self, DKL, rd):
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()

        ### start NRE training section ###
        root = f"{self.nreSettings.root}_round_{rd}"
        if rank_gen == 0:
            self.logger.info("retraining round: " + str(rd))
            if self.nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=self.nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
            new_trainer = copy.deepcopy(self.trainer)
            new_network = self.network.get_new_network()
            new_network = retrain_next_round(root=root, training_data=self.training_samples,
                                             nreSettings=self.nreSettings, sim=self.sim, obs=self.obs,
                                             network=new_network, dm=self.dm,
                                             trainer=new_trainer)
        else:
            new_network = self.network.get_new_network()
        comm_gen.Barrier()
        ### load saved network and save it in network_storage ###
        new_network.load_state_dict(torch.load(f"{root}/{self.nreSettings.neural_network_file}"))
        new_network.double()  # change to float64 precision of network
        self.network_storage[f"round_{rd}"] = new_network
        self.root_storage[f"round_{rd}"] = root
        self.logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")

        ### start polychord section ###
        ### Run PolyChord ###
        self.polyset.base_dir = root
        self.polyset.nlive = self.nreSettings.nlives_per_dim_dic[rd]
        self.network.set_network(network=new_network)
        pypolychord.run_polychord(loglikelihood=self.network.logLikelihood,
                                  nDims=self.nreSettings.num_features,
                                  nDerived=self.nreSettings.nderived, settings=self.polyset,
                                  prior=self.network.prior, dumper=self.network.dumper)
        comm_gen.Barrier()

        ### load deadpoints and compute KL divergence and reassign to training samples ###
        deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
        if rd >= 1:
            previous_network = self.network_storage[f"round_{rd - 1}"]
            DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                        current_samples=deadpoints, obs=self.obs)
            self.dkl_storage.append(DKL)
            self.logger.info(f"DKL of rd {rd} is: {DKL}")
        comm_gen.Barrier()
        deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        self.logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
        self.training_samples = deadpoints
        return DKL
