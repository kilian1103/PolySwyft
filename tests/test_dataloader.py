"""Tests for PolySwyft.PolySwyft_Dataloader."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from polyswyft.dataloader import PolySwyftDataModule, PolySwyftSequential


# =============================================================================
# PolySwyftSequential
# =============================================================================
class TestPolySwyftSequential:
    def test_length_multi_round(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=1)
        # 50 samples per round x 2 rounds = 100
        assert len(ds) == 100

    def test_length_single_round(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)
        assert len(ds) == 50

    def test_getitem_returns_dict(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {round_data_on_disk.targetKey, round_data_on_disk.obsKey}

    def test_getitem_tensor_shapes(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)
        sample = ds[0]
        assert sample[round_data_on_disk.targetKey].shape == (round_data_on_disk.num_features,)
        assert sample[round_data_on_disk.obsKey].shape == (round_data_on_disk.num_features_dataset,)

    def test_getitem_tensor_dtype_float(self, round_data_on_disk):
        """Dataset converts via .float(), so dtype must be float32."""
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)
        sample = ds[0]
        assert sample[round_data_on_disk.targetKey].dtype == torch.float32
        assert sample[round_data_on_disk.obsKey].dtype == torch.float32

    def test_index_map_start(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=1)
        assert ds.index_map[0] == (0, 0)

    def test_index_map_second_round_offset(self, round_data_on_disk):
        """Sample 50 should be the first sample of round 1."""
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=1)
        assert ds.index_map[50] == (1, 0)

    def test_index_map_last(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=1)
        assert ds.index_map[-1] == (1, 49)

    def test_index_subset(self, round_data_on_disk):
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=1, index_subset=[0, 1, 50])
        assert len(ds) == 3
        assert ds.index_map == [(0, 0), (0, 1), (1, 0)]

    def test_on_after_load_sample_callback(self, round_data_on_disk):
        def transform(sample):
            sample["marker"] = torch.tensor([42.0])
            return sample

        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0, on_after_load_sample=transform)
        sample = ds[0]
        assert "marker" in sample
        assert torch.equal(sample["marker"], torch.tensor([42.0]))

    def test_data_round_trip(self, round_data_on_disk):
        """Samples from the dataset must match the on-disk .npy contents."""
        ds = PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)
        expected_thetas = np.load(f"{round_data_on_disk.root}/round_0/{round_data_on_disk.targetKey}.npy")
        sample_0 = ds[0]
        np.testing.assert_allclose(
            sample_0[round_data_on_disk.targetKey].numpy(),
            expected_thetas[0].astype(np.float32),
        )

    def test_mismatched_rounds_raises_assertion(self, round_data_on_disk, tmp_path):
        """If theta and obs files have different sample counts the __init__ should assert."""
        root = round_data_on_disk.root
        # Overwrite round_0/x.npy with a shorter file.
        np.save(
            f"{root}/round_0/{round_data_on_disk.obsKey}.npy",
            np.zeros((10, round_data_on_disk.num_features_dataset)),
        )
        with pytest.raises(AssertionError, match="Mismatch"):
            PolySwyftSequential(polyswyftSettings=round_data_on_disk, rd=0)


# =============================================================================
# PolySwyftDataModule
# =============================================================================
class TestPolySwyftDataModule:
    def test_setup_with_explicit_lengths(self, round_data_on_disk):
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            lengths=[70, 20, 10],
            batch_size=8,
        )
        dm.setup()
        assert len(dm.dataset_train) == 70
        assert len(dm.dataset_val) == 20
        assert len(dm.dataset_test) == 10

    def test_setup_with_fractions(self, round_data_on_disk):
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            fractions=[0.8, 0.1, 0.1],
            batch_size=8,
        )
        dm.setup()
        total = len(dm.dataset_train) + len(dm.dataset_val) + len(dm.dataset_test)
        assert total == 100  # 50 * 2 rounds
        # 0.8 * 100 = 80 (+ any remainder goes to train)
        assert len(dm.dataset_train) >= 80

    def test_setup_fractions_normalized(self, round_data_on_disk):
        """Unnormalized fractions [2, 1, 1] should behave like [0.5, 0.25, 0.25]."""
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            fractions=[2, 1, 1],
            batch_size=8,
        )
        dm.setup()
        total = len(dm.dataset_train) + len(dm.dataset_val) + len(dm.dataset_test)
        assert total == 100
        assert len(dm.dataset_train) >= 50

    def test_setup_no_lengths_no_fractions_raises(self, round_data_on_disk):
        dm = PolySwyftDataModule(polyswyftSettings=round_data_on_disk, rd=0, batch_size=8)
        with pytest.raises(ValueError, match="lengths.*fractions"):
            dm.setup()

    def test_init_both_lengths_and_fractions_raises(self, round_data_on_disk):
        with pytest.raises(ValueError, match="lengths.*fractions"):
            PolySwyftDataModule(
                polyswyftSettings=round_data_on_disk,
                rd=0,
                lengths=[70, 20, 10],
                fractions=[0.8, 0.1, 0.1],
                batch_size=8,
            )

    def test_init_zero_fractions_raises(self, round_data_on_disk):
        with pytest.raises(ValueError, match="fractions"):
            PolySwyftDataModule(
                polyswyftSettings=round_data_on_disk,
                rd=0,
                fractions=[0.0, 0.0, 0.0],
                batch_size=8,
            )

    def test_init_negative_fraction_raises(self, round_data_on_disk):
        with pytest.raises(ValueError, match="fractions"):
            PolySwyftDataModule(
                polyswyftSettings=round_data_on_disk,
                rd=0,
                fractions=[-0.1, 0.5, 0.6],
                batch_size=8,
            )

    def test_train_dataloader_yields_batches(self, round_data_on_disk):
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            fractions=[0.8, 0.1, 0.1],
            batch_size=8,
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert round_data_on_disk.targetKey in batch
        assert round_data_on_disk.obsKey in batch
        assert batch[round_data_on_disk.targetKey].shape[1] == round_data_on_disk.num_features
        assert batch[round_data_on_disk.obsKey].shape[1] == round_data_on_disk.num_features_dataset

    def test_val_and_test_dataloaders_return_dataloaders(self, round_data_on_disk):
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            fractions=[0.8, 0.1, 0.1],
            batch_size=8,
        )
        dm.setup()
        assert isinstance(dm.val_dataloader(), DataLoader)
        assert isinstance(dm.test_dataloader(), DataLoader)

    def test_dm_honors_batch_size(self, round_data_on_disk):
        dm = PolySwyftDataModule(
            polyswyftSettings=round_data_on_disk,
            rd=1,
            fractions=[0.8, 0.1, 0.1],
            batch_size=16,
        )
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch[round_data_on_disk.targetKey].shape[0] <= 16
