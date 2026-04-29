"""Multi-round dataset and Lightning DataModule used during NRE training.

The NSNRE cycle accumulates a *cumulative* training set across rounds:
round ``i`` trains on the union of round-0 prior samples plus dead-measure
samples from rounds ``1..i``. This module reads each round's on-disk
``{targetKey}.npy`` / ``{obsKey}.npy`` artefacts via memory-mapped numpy
arrays so the whole history fits in RAM regardless of round count.
"""

import os
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from polyswyft.settings import PolySwyftSettings


class PolySwyftSequential(Dataset):
    """Cumulative-across-rounds ``torch.utils.data.Dataset`` for NSNRE training.

    Loads ``{root}/{child_root}_{k}/{targetKey}.npy`` and
    ``{root}/{child_root}_{k}/{obsKey}.npy`` for ``k = 0..rd`` using
    ``mmap_mode='r'``, then exposes a flat index over the union. The
    per-round directory prefix matches the one written by
    :class:`polyswyft.core.PolySwyft` and ``polyswyft.utils.resimulate_deadpoints``.
    Each ``__getitem__`` call
    returns one ``(theta, D)`` pair as a dict with the keys defined in
    ``PolySwyftSettings.targetKey`` / ``obsKey``.

    The relative weight of round ``k`` in the cumulative training set is
    ``N_k / sum_j N_j`` where ``N_k`` is the number of samples written in
    that round. This implicit weighting matches the ω-weighted dead-measure
    mixture described in section 3.1 of the paper.

    Parameters
    ----------
    polyswyftSettings : PolySwyftSettings
        Active run settings.
    rd : int
        Last round to include (inclusive). Round 0 always loaded.
    index_subset : sequence[int], optional
        Optional integer indices that restrict the dataset to a subset of
        the cumulative pool (used by Lightning's ``random_split`` train/val
        partitioning).
    on_after_load_sample : callable, optional
        Hook ``sample -> sample`` invoked after every ``__getitem__``;
        useful for on-the-fly augmentation.
    """

    def __init__(
        self,
        polyswyftSettings: PolySwyftSettings,
        rd: int,
        index_subset: Optional[list[int]] = None,
        on_after_load_sample: Optional[Callable] = None,
    ):
        self.polyswyftSettings = polyswyftSettings
        self.rd = rd
        self.on_after_load_sample = on_after_load_sample
        self.round_data: dict[int, dict[str, np.ndarray]] = {}
        self.index_map: list[tuple[int, int]] = []

        for round_id in range(self.rd + 1):
            round_dir = os.path.join(
                self.polyswyftSettings.root,
                f"{self.polyswyftSettings.child_root}_{round_id}",
            )
            theta_path = os.path.join(round_dir, f"{self.polyswyftSettings.targetKey}.npy")
            obs_path = os.path.join(round_dir, f"{self.polyswyftSettings.obsKey}.npy")

            thetas = np.load(theta_path, mmap_mode="r")
            obs = np.load(obs_path, mmap_mode="r")

            assert thetas.shape[0] == obs.shape[0], f"Mismatch in round {round_id}"

            self.round_data[round_id] = {self.polyswyftSettings.targetKey: thetas, self.polyswyftSettings.obsKey: obs}

            self.index_map.extend([(round_id, i) for i in range(thetas.shape[0])])

        if index_subset is not None:
            self.index_map = [self.index_map[i] for i in index_subset]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rd, i = self.index_map[idx]
        data = self.round_data[rd]

        sample = {
            self.polyswyftSettings.targetKey: torch.tensor(data[self.polyswyftSettings.targetKey][i]).float(),
            self.polyswyftSettings.obsKey: torch.tensor(data[self.polyswyftSettings.obsKey][i]).float(),
        }

        if self.on_after_load_sample:
            sample = self.on_after_load_sample(sample)

        return sample


class PolySwyftDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping :class:`PolySwyftSequential`.

    Splits the cumulative dataset into train/val/test partitions using
    ``torch.utils.data.random_split``. Either ``lengths`` (absolute counts)
    or ``fractions`` (proportions summing to ~1) must be provided;
    ``fractions`` are normalised and converted to integer lengths internally.

    Parameters
    ----------
    polyswyftSettings : PolySwyftSettings
        Active run settings.
    rd : int
        Round index passed through to :class:`PolySwyftSequential`.
    lengths : sequence[int], optional
        Absolute split sizes ``[n_train, n_val, n_test]``. Mutually
        exclusive with ``fractions``.
    fractions : sequence[float], optional
        Relative split sizes; rounded to integer counts. Mutually exclusive
        with ``lengths``.
    batch_size : int, default 64
        Mini-batch size for the train/val/test DataLoaders.
    num_workers : int, default 0
        Worker processes for DataLoader; ``0`` keeps loading on the main
        process (safer with MPI).
    shuffle : bool, default True
        Whether to shuffle the train DataLoader. Validation/test loaders
        are never shuffled.
    on_after_load_sample : callable, optional
        Forwarded to :class:`PolySwyftSequential`.

    Raises
    ------
    ValueError
        If neither ``lengths`` nor ``fractions`` is provided, if both are
        provided simultaneously, or if ``fractions`` contains negative
        values or sums to zero.
    """

    def __init__(
        self,
        polyswyftSettings: PolySwyftSettings,
        rd: int,
        lengths: Optional[list[int]] = None,
        fractions: Optional[list[float]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        on_after_load_sample: Optional[Callable] = None,
    ):
        super().__init__()
        if lengths is not None and fractions is not None:
            raise ValueError("Pass either `lengths` or `fractions`, not both.")
        if fractions is not None:
            arr = np.asarray(fractions, dtype=float)
            if (arr < 0).any() or not arr.sum() > 0:
                raise ValueError("`fractions` must be non-negative with a positive sum.")
        self.polyswyftSettings = polyswyftSettings
        self.rd = rd
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.lengths = lengths
        self.fractions = fractions
        self.on_after_load_sample = on_after_load_sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Materialise the cumulative dataset and apply the train/val/test split.

        Called automatically by Lightning before ``trainer.fit``. Resolves
        ``fractions`` to integer ``lengths`` if needed, then assigns
        ``dataset_train``, ``dataset_val``, ``dataset_test``.
        """
        full_dataset = PolySwyftSequential(
            polyswyftSettings=self.polyswyftSettings,
            rd=self.rd,
            on_after_load_sample=self.on_after_load_sample,
        )

        total_len = len(full_dataset)

        if self.lengths is None and self.fractions is not None:
            fractions = np.array(self.fractions, dtype=float)
            fractions /= fractions.sum()
            counts = np.floor(total_len * fractions).astype(int)
            counts[0] += total_len - np.sum(counts)
            self.lengths = counts.tolist()

        if self.lengths is not None:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(full_dataset, self.lengths)
        else:
            raise ValueError("Must provide either `lengths` or `fractions`.")

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffled per ``self.shuffle``)."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (never shuffled)."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader (never shuffled)."""
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
