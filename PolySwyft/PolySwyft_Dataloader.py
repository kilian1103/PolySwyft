from torch.utils.data import Dataset
import torch
import os
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Callable, Optional
class PolySwyftSequential(Dataset):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, rd, index_subset=None, on_after_load_sample: Optional[Callable] =None):
        self.polyswyftSettings = polyswyftSettings
        self.rd = rd
        self.on_after_load_sample = on_after_load_sample
        self.round_data = {}
        self.index_map = []

        for round_id in range(self.rd + 1):
            round_dir = os.path.join(self.polyswyftSettings.root, f"round_{round_id}")
            theta_path = os.path.join(round_dir, f"{self.polyswyftSettings.targetKey}.npy")
            obs_path = os.path.join(round_dir, f"{self.polyswyftSettings.obsKey}.npy")

            thetas = np.load(theta_path, mmap_mode='r')  # memory-efficient numpy mmap
            obs = np.load(obs_path, mmap_mode='r')

            assert thetas.shape[0] == obs.shape[0], f"Mismatch in round {round_id}"

            self.round_data[round_id] = {
                self.polyswyftSettings.targetKey: thetas,
                self.polyswyftSettings.obsKey: obs
            }

            self.index_map.extend([(round_id, i) for i in range(thetas.shape[0])])

        if index_subset is not None:
            self.index_map = [self.index_map[i] for i in index_subset]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
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
    def __init__(
            self,
            polyswyftSettings: PolySwyft_Settings,
            rd: int,
            lengths=None,
            fractions=None,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            on_after_load_sample=None,
    ):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.rd = rd
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.lengths = lengths
        self.fractions = fractions
        self.on_after_load_sample = on_after_load_sample

    def setup(self, stage=None):
        full_dataset = PolySwyftSequential(
            polyswyftSettings=self.polyswyftSettings,
            rd=self.rd,
            on_after_load_sample=self.on_after_load_sample,
        )

        total_len = len(full_dataset)

        if self.lengths is None and self.fractions is not None:
            fractions = np.array(self.fractions)
            fractions /= fractions.sum()
            counts = np.floor(total_len * fractions).astype(int)
            counts[0] += total_len - np.sum(counts)  # ensure sum == total_len
            self.lengths = counts.tolist()

        if self.lengths is not None:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(full_dataset, self.lengths)
        else:
            raise ValueError("Must provide either `lengths` or `fractions`.")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )