"""Integration tests for MPI code paths — require mpi4py and mpirun.

Run with:
    mpirun -n 1 pytest tests/test_mpi_integration.py -v
    mpirun -n 2 pytest tests/test_mpi_integration.py -v
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import swyft

MPI = pytest.importorskip("mpi4py.MPI")

from polyswyft.settings import PolySwyftSettings  # noqa: E402
from polyswyft.utils import resimulate_deadpoints  # noqa: E402

pytestmark = pytest.mark.integration


def _make_settings(root, num_features=2, num_features_dataset=4):
    settings = PolySwyftSettings(root=root)
    settings.num_features = num_features
    settings.num_features_dataset = num_features_dataset
    settings.logger = logging.getLogger("mpi_integration_test")
    settings.use_noise_resampling = False
    return settings


def _make_mock_simulator(settings):
    """Mock simulator returning real swyft.Sample objects."""
    d = settings.num_features_dataset
    rng = np.random.default_rng(42 + MPI.COMM_WORLD.Get_rank())
    sim = MagicMock()

    def _sample(conditions=None, targets=None):
        return swyft.Sample(
            **{
                settings.targetKey: conditions[settings.targetKey].ravel(),
                settings.obsKey: rng.standard_normal(d),
            }
        )

    sim.sample.side_effect = _sample
    return sim


def _shared_tmpdir():
    """Create a tmpdir on rank 0 and broadcast the path to all ranks."""
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        path = tempfile.mkdtemp(prefix="polyswyft_mpi_test_")
    else:
        path = None
    path = comm.bcast(path, root=0)
    return path


class TestResimulateDeadpointsIntegration:
    def test_returns_correct_shapes(self):
        """All ranks get thetas and Ds with correct shapes."""
        comm = MPI.COMM_WORLD
        root = _shared_tmpdir()
        settings = _make_settings(root)
        n_deadpoints = 20
        deadpoints = np.random.default_rng(0).standard_normal((n_deadpoints, settings.num_features))

        rd_dir = f"{root}/{settings.child_root}_0"
        os.makedirs(rd_dir, exist_ok=True)
        comm.Barrier()

        sim = _make_mock_simulator(settings)
        thetas, Ds = resimulate_deadpoints(deadpoints, settings, sim, rd=0)

        assert thetas.shape == (n_deadpoints, settings.num_features)
        assert Ds.shape == (n_deadpoints, settings.num_features_dataset)

    def test_all_ranks_get_same_data(self):
        """After bcast, all ranks must have identical thetas and Ds."""
        comm = MPI.COMM_WORLD
        root = _shared_tmpdir()
        settings = _make_settings(root)
        n_deadpoints = 20
        deadpoints = np.random.default_rng(0).standard_normal((n_deadpoints, settings.num_features))

        rd_dir = f"{root}/{settings.child_root}_0"
        os.makedirs(rd_dir, exist_ok=True)
        comm.Barrier()

        sim = _make_mock_simulator(settings)
        thetas, Ds = resimulate_deadpoints(deadpoints, settings, sim, rd=0)

        # Gather results from all ranks onto rank 0
        all_thetas = comm.gather(thetas, root=0)
        all_Ds = comm.gather(Ds, root=0)

        if comm.Get_rank() == 0:
            for i in range(1, len(all_thetas)):
                np.testing.assert_array_equal(all_thetas[0], all_thetas[i])
                np.testing.assert_array_equal(all_Ds[0], all_Ds[i])

    def test_file_only_on_rank_zero(self):
        """Only rank 0 should create .npy files."""
        comm = MPI.COMM_WORLD
        root = _shared_tmpdir()
        settings = _make_settings(root)
        n_deadpoints = 20
        deadpoints = np.random.default_rng(0).standard_normal((n_deadpoints, settings.num_features))

        rd_dir = f"{root}/{settings.child_root}_0"
        os.makedirs(rd_dir, exist_ok=True)
        comm.Barrier()

        sim = _make_mock_simulator(settings)
        resimulate_deadpoints(deadpoints, settings, sim, rd=0)

        comm.Barrier()
        if comm.Get_rank() == 0:
            assert os.path.exists(f"{rd_dir}/{settings.targetKey}.npy")
            assert os.path.exists(f"{rd_dir}/{settings.obsKey}.npy")
