"""Core orchestrator for the NSNRE cycle.

The :class:`PolySwyft` class drives the five-phase loop described in
section 3.2 of the paper: simulate using the latest dead points, retrain
the NRE on the cumulative dataset, run PolyChord on the trained
log-ratio, compute KL diagnostics, and either terminate or continue.

All on-disk artefacts are written under ``polyswyftSettings.root``::

    {root}/
        settings.pkl                  # pickled PolySwyftSettings
        round_0/                      # per-round subdirectory
            z.npy, x.npy              # training dataset (theta, D)
            NRE_network.pt            # trained NRE state-dict
            optimizer_file.pt         # optimiser state
            samples.txt, *.dead, ...  # PolyChord output chains
            enhanced_run/             # optional second-pass chains
        round_1/
        ...
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING, Callable

import anesthetic
import numpy as np
import swyft
import torch

from polyswyft.dataloader import PolySwyftDataModule
from polyswyft.network import PolySwyftNetwork
from polyswyft.settings import PolySwyftSettings
from polyswyft.utils import (
    compute_KL_compression,
    compute_KL_divergence,
    resimulate_deadpoints,
    select_weighted_contour,
)

if TYPE_CHECKING:
    from pypolychord import PolyChordSettings


class PolySwyft:
    """Orchestrator for the sequential NSNRE cycle.

    Holds references to the simulator, observation, NRE template, and
    PolyChord settings, then drives the round-by-round retrain/sample loop
    via :meth:`execute_NSNRE_cycle`.

    Diagnostics produced during the cycle are stored as instance dicts
    keyed by round index (or by ``"current"`` / ``"previous"``):

    Attributes
    ----------
    network_storage : dict
        Networks indexed by ``"current"`` / ``"previous"`` for the latest
        two rounds.
    samples_storage : dict
        Anesthetic ``NestedSamples`` indexed by ``"current"`` / ``"previous"``.
    root_storage : dict[int, str]
        Per-round output directory paths.
    dkl_storage : dict[int, tuple[float, float]]
        ``KL(P_i || P_{i-1})`` (mean, std) per round, populated from round 1.
    dkl_compression_storage : dict[int, tuple[float, float]]
        ``KL(P_i || pi)`` (mean, std) per round.
    """

    def __init__(
        self,
        polyswyftSettings: PolySwyftSettings,
        sim: swyft.Simulator,
        obs: swyft.Sample,
        deadpoints: np.ndarray,
        network: PolySwyftNetwork,
        polyset: PolyChordSettings,
        callbacks: Callable,
        lr_round_scheduler: Callable = None,
        deadpoints_processing: Callable = None,
    ):
        """Initialise a PolySwyft orchestrator.

        Parameters
        ----------
        polyswyftSettings : PolySwyftSettings
            Run configuration; controls round count, training hyper-params,
            and output paths.
        sim : swyft.Simulator
            Forward simulator producing joint ``(theta, D)`` pairs.
        obs : swyft.Sample
            Observed data ``D_obs`` PolyChord conditions on each round.
        deadpoints : np.ndarray
            Initial parameter samples for round 0; typically prior draws of
            shape ``(n_training_samples, num_features)``.
        network : PolySwyftNetwork
            NRE template; ``network.get_new_network()`` is called at the
            start of every round.
        polyset : PolyChordSettings
            PolyChord run settings (``nlive``, ``precision_criterion``,
            etc.). ``base_dir`` is overwritten per round.
        callbacks : callable
            Zero-arg factory returning a fresh list of Lightning callbacks
            per round (e.g. ``EarlyStopping``, ``ModelCheckpoint``).
        lr_round_scheduler : callable, optional
            Map ``rd -> learning_rate`` invoked at the start of each round
            to update the optimiser's ``lr``. ``None`` keeps the network
            default.
        deadpoints_processing : callable, optional
            Map ``(deadpoints_samples, rd) -> deadpoints_samples`` applied
            after each PolyChord run, e.g. to truncate or reweight the
            dead-point set fed into the next round.

        Raises
        ------
        ImportError
            If ``mpi4py`` is not installed; PolySwyft requires MPI for the
            full cycle.
        """
        try:
            from mpi4py import MPI
        except ImportError as err:
            raise ImportError("mpi4py is required for PolySwyft. Install it with: pip install polyswyft[mpi]") from err
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()

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

    def execute_NSNRE_cycle(self) -> None:
        """Run the full NSNRE cycle.

        Creates ``self.polyswyftSettings.root`` if absent, pickles the
        settings, optionally resumes from round
        ``polyswyftSettings.NRE_start_from_round - 1`` artefacts, then
        dispatches to either :meth:`_cyclic_rounds` (fixed budget) or
        :meth:`_cyclic_kl` (adaptive termination on
        ``KL(P_i || P_{i-1})``) depending on
        ``polyswyftSettings.cyclic_rounds``.

        Side effects
        ------------
        Writes per-round artefacts under
        ``{polyswyftSettings.root}/{child_root}_{rd}/`` and populates
        ``self.network_storage``, ``self.samples_storage``,
        ``self.root_storage``, ``self.dkl_storage``,
        ``self.dkl_compression_storage``.

        Raises
        ------
        ValueError
            When resuming with ``NRE_start_from_round > NRE_num_retrain_rounds``
            in cyclic-rounds mode.
        """
        self.logger = logging.getLogger(self.polyswyftSettings.logger_name)

        ### create root folder ###
        os.makedirs(self.polyswyftSettings.root, exist_ok=True)

        ### save settings
        with open(f"{self.polyswyftSettings.root}/settings.pkl", "wb") as file:
            pickle.dump(self.polyswyftSettings, file)

        ### reload data if necessary to resume run ###
        if self.polyswyftSettings.NRE_start_from_round > 0:
            if (
                self.polyswyftSettings.NRE_start_from_round > self.polyswyftSettings.NRE_num_retrain_rounds
                and self.polyswyftSettings.cyclic_rounds
            ):
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")

            root = (
                f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_"
                f"{self.polyswyftSettings.NRE_start_from_round - 1}"
            )
            deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
            self.deadpoints_samples = deadpoints.iloc[:, : self.polyswyftSettings.num_features].to_numpy()
            network = self.network_model.get_new_network()
            network.load_state_dict(torch.load(f"{root}/{self.polyswyftSettings.neural_network_file}"))
            network.double()
            self.root_storage[self.polyswyftSettings.NRE_start_from_round - 1] = root
            self.network_storage[self.previous_key] = network
            self.samples_storage[self.previous_key] = deadpoints

        ### execute main cycle ###
        if self.polyswyftSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()

    def _cyclic_rounds(self) -> None:
        """Run a fixed number of NSNRE rounds (``NRE_num_retrain_rounds``)."""
        for rd in range(self.polyswyftSettings.NRE_start_from_round, self.polyswyftSettings.NRE_num_retrain_rounds + 1):
            self._cycle(rd)

    def _cyclic_kl(self) -> None:
        """Run rounds adaptively until ``|KL(P_i || P_{i-1})| < termination_abs_dkl``.

        Experimental — see ``polyswyftSettings.cyclic_rounds`` docstring.
        """
        DKL_info = (100, 100)
        DKL, DKL_std = DKL_info
        rd = self.polyswyftSettings.NRE_start_from_round
        while abs(DKL) >= self.polyswyftSettings.termination_abs_dkl:
            self._cycle(rd)
            DKL, DKL_std = self.dkl_storage[rd]
            rd += 1
        self.polyswyftSettings.NRE_num_retrain_rounds = rd - 1

    def _cycle(self, rd: int) -> None:
        """Execute one NSNRE round: simulate, train NRE, run PolyChord, score KL.

        Steps (in order):
        1. ``resimulate_deadpoints`` materialises ``(theta, D)`` for the
           current dead-point set.
        2. A fresh network is instantiated; under
           ``continual_learning_mode`` it is warm-started from the previous
           round's checkpoint.
        3. ``swyft.SwyftTrainer.fit`` trains on the cumulative dataset via
           :class:`PolySwyftDataModule`.
        4. ``pypolychord.run_polychord`` is invoked with
           ``network.logRatio`` as the likelihood callback.
        5. Optionally, a second PolyChord pass with boosted live points
           runs at the chosen posterior contour.
        6. ``KL(P_i || P_{i-1})`` and ``KL(P_i || pi)`` are computed and
           cached in ``self.dkl_storage`` / ``self.dkl_compression_storage``.

        Parameters
        ----------
        rd : int
            Round index (zero-based).
        """

        ### start NRE training section ###
        self.logger.info("training network round: " + str(rd))
        root = f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_{rd}"
        ### create root folder ###
        os.makedirs(root, exist_ok=True)

        ### generate training data using deadpoints
        resimulate_deadpoints(
            deadpoints=self.deadpoints_samples, polyswyftSettings=self.polyswyftSettings, sim=self.sim, rd=rd
        )

        ### setup wandb ###
        if self.polyswyftSettings.activate_wandb:
            from pytorch_lightning.loggers import WandbLogger

            self.polyswyftSettings.wandb_kwargs["name"] = f"{self.polyswyftSettings.child_root}_{rd}"
            self.polyswyftSettings.wandb_kwargs["save_dir"] = (
                f"{self.polyswyftSettings.root}/{self.polyswyftSettings.child_root}_{rd}"
            )
            wandb_logger = WandbLogger(**self.polyswyftSettings.wandb_kwargs)
            self.polyswyftSettings.trainer_kwargs["logger"] = wandb_logger

        ### setup trainer ###
        self.polyswyftSettings.trainer_kwargs["default_root_dir"] = root
        self.polyswyftSettings.trainer_kwargs["callbacks"] = self.callbacks()
        trainer = swyft.SwyftTrainer(**self.polyswyftSettings.trainer_kwargs)

        ### setup network
        network = self.network_model.get_new_network()
        if self.polyswyftSettings.continual_learning_mode and rd > 0:
            prev_root = self.root_storage[rd - 1]
            network.load_state_dict(torch.load(f"{prev_root}/{self.polyswyftSettings.neural_network_file}"))

        ### continue lr rate at last point
        if callable(self.lr_round_scheduler):
            learning_rate = self.lr_round_scheduler(rd)  # between rounds
            network.optimizer_init.optim_args = dict(lr=learning_rate)

        ### train network
        dm = PolySwyftDataModule(polyswyftSettings=self.polyswyftSettings, rd=rd, **self.polyswyftSettings.dm_kwargs)
        network.train()
        trainer.fit(network, dm)
        self._comm.Barrier()
        if self.polyswyftSettings.activate_wandb and self._rank == 0:
            import wandb

            wandb.finish()

        ### save network on disk ###
        if self._rank == 0:
            torch.save(network.state_dict(), f"{root}/{self.polyswyftSettings.neural_network_file}")
            torch.save(network.optimizers().state_dict(), f"{root}/{self.polyswyftSettings.optimizer_file}")

        self._comm.Barrier()

        ### load network on disk (to sync across nodes) ###
        if self.polyswyftSettings.continual_learning_mode:
            network.load_state_dict(torch.load(f"{root}/{self.polyswyftSettings.neural_network_file}"))
        self._comm.Barrier()

        ### prepare network for inference ###
        network.eval()

        ### start polychord section ###
        ### run PolyChord ###
        import pypolychord

        self.logger.info("Using PolyChord with trained NRE to generate deadpoints for the next round!")
        self.polyset.base_dir = root
        self._comm.barrier()

        pypolychord.run_polychord(
            loglikelihood=network.logRatio,
            nDims=self.polyswyftSettings.num_features,
            nDerived=self.polyswyftSettings.nderived,
            settings=self.polyset,
            prior=network.prior,
            dumper=network.dumper,
        )
        self._comm.Barrier()

        deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
        self._comm.Barrier()

        ### polychord round 2 section (optional for dynamic nested sampling)###
        if self.polyswyftSettings.use_livepoint_increasing:
            ### choose contour to increase livepoints ###
            index = select_weighted_contour(
                deadpoints, threshold=1 - self.polyswyftSettings.livepoint_increase_posterior_contour
            )
            logL = deadpoints.iloc[index, :].logL

            os.makedirs(f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}", exist_ok=True)

            ### run polychord round 2 ###
            self.polyset.base_dir = f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}"
            self.polyset.nlives = {logL: self.polyswyftSettings.n_increased_livepoints}
            self._comm.Barrier()
            pypolychord.run_polychord(
                loglikelihood=network.logRatio,
                nDims=self.polyswyftSettings.num_features,
                nDerived=self.polyswyftSettings.nderived,
                settings=self.polyset,
                prior=network.prior,
                dumper=network.dumper,
            )
            self._comm.Barrier()
            self.polyset.nlives = {}
            deadpoints = anesthetic.read_chains(
                root=f"{root}/{self.polyswyftSettings.increased_livepoints_fileroot}/{self.polyset.file_root}"
            )
            self._comm.Barrier()

        #### compute KL divergences
        self.root_storage[rd] = root
        self.samples_storage[self.current_key] = deadpoints
        self.network_storage[self.current_key] = network

        #### compute KL(P_i||P_(i-1))
        if rd > 0:
            previous_network = self.network_storage[self.previous_key]
            KDL = compute_KL_divergence(
                polyswyftSettings=self.polyswyftSettings,
                previous_network=previous_network.eval(),
                current_samples=self.samples_storage[self.current_key],
                obs=self.obs,
                previous_samples=self.samples_storage[self.previous_key],
            )
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
        if callable(self.deadpoints_processing):
            deadpoints = self.deadpoints_processing(deadpoints, rd)

        # prepare data for next round
        deadpoints_samples = deadpoints.iloc[:, : self.polyswyftSettings.num_features].to_numpy()
        self._comm.Barrier()
        self.logger.info(f"Number of deadpoints for next rd {rd + 1}: {deadpoints_samples.shape[0]}")
        self.deadpoints_samples = deadpoints_samples
        return
