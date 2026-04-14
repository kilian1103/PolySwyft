"""Mock-based unit tests for MPI code paths in polyswyft.

These tests run without mpi4py installed by mocking the MPI module.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import swyft

from polyswyft.settings import PolySwyftSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_comm(rank=0, size=1):
    """Return a MagicMock mimicking MPI.COMM_WORLD with given rank/size."""
    comm = MagicMock()
    comm.Get_rank.return_value = rank
    comm.Get_size.return_value = size
    comm.Barrier.return_value = None
    comm.barrier.return_value = None
    # Default passthrough behaviour (overridden in multi-rank tests)
    comm.bcast.side_effect = lambda data, root=0: data
    comm.allgather.side_effect = lambda data: [data]
    return comm


def _make_mock_mpi(comm):
    """Return mock mpi4py and MPI modules wired to *comm*."""
    mock_MPI = MagicMock()
    mock_MPI.COMM_WORLD = comm
    mock_mpi4py = MagicMock()
    mock_mpi4py.MPI = mock_MPI
    return mock_mpi4py, mock_MPI


def _make_settings(tmp_path, num_features=2, num_features_dataset=4):
    """Minimal PolySwyftSettings for MPI tests."""
    import logging

    settings = PolySwyftSettings(root=str(tmp_path / "mpi_test"))
    settings.num_features = num_features
    settings.num_features_dataset = num_features_dataset
    settings.logger = logging.getLogger("mpi_test")
    settings.use_noise_resampling = False
    settings.n_noise_resampling_samples = 3
    return settings


def _make_mock_simulator(settings):
    """Return a mock simulator whose .sample() returns real swyft.Samples."""
    d = settings.num_features_dataset
    rng = np.random.default_rng(42)

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


# ============================================================================
# resimulate_deadpoints — ImportError
# ============================================================================
class TestResimulateDeadpointsImportError:
    def test_raises_import_error_when_mpi4py_missing(self, tmp_path):
        settings = _make_settings(tmp_path)
        deadpoints = np.zeros((5, settings.num_features))
        sim = MagicMock()

        with patch.dict(sys.modules, {"mpi4py": None, "mpi4py.MPI": None}):
            from polyswyft.utils import resimulate_deadpoints

            with pytest.raises(ImportError, match="mpi4py"):
                resimulate_deadpoints(deadpoints, settings, sim, rd=0)


# ============================================================================
# resimulate_deadpoints — single rank (size=1, rank=0)
# ============================================================================
class TestResimulateDeadpointsSingleRank:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.settings = _make_settings(tmp_path)
        self.n_deadpoints = 10
        self.deadpoints = np.random.default_rng(0).standard_normal((self.n_deadpoints, self.settings.num_features))
        self.sim = _make_mock_simulator(self.settings)
        self.comm = _make_mock_comm(rank=0, size=1)
        self.mock_mpi4py, self.mock_MPI = _make_mock_mpi(self.comm)

        # Create round directory
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_0"
        os.makedirs(rd_dir, exist_ok=True)

    def _call(self, rd=0):
        with patch.dict(
            sys.modules,
            {"mpi4py": self.mock_mpi4py, "mpi4py.MPI": self.mock_MPI},
        ):
            from polyswyft.utils import resimulate_deadpoints

            return resimulate_deadpoints(self.deadpoints, self.settings, self.sim, rd=rd)

    def test_returns_correct_shapes(self):
        thetas, Ds = self._call()
        assert thetas.shape == (self.n_deadpoints, self.settings.num_features)
        assert Ds.shape == (self.n_deadpoints, self.settings.num_features_dataset)

    def test_no_allgather_when_size_one(self):
        self._call()
        self.comm.allgather.assert_not_called()

    def test_files_saved_on_rank_zero(self):
        self._call()
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_0"
        assert os.path.exists(f"{rd_dir}/{self.settings.targetKey}.npy")
        assert os.path.exists(f"{rd_dir}/{self.settings.obsKey}.npy")

    def test_barrier_call_count(self):
        self._call()
        assert self.comm.Barrier.call_count == 4
        self.comm.barrier.assert_not_called()

    def test_bcast_called(self):
        self._call()
        assert self.comm.bcast.call_count == 2


# ============================================================================
# resimulate_deadpoints — multi rank (size=2)
# ============================================================================
class TestResimulateDeadpointsMultiRank:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.settings = _make_settings(tmp_path)
        self.n_deadpoints = 10
        self.deadpoints = np.random.default_rng(0).standard_normal((self.n_deadpoints, self.settings.num_features))
        self.sim = _make_mock_simulator(self.settings)

        # Create round directory
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_0"
        os.makedirs(rd_dir, exist_ok=True)

    def _make_comm_and_mpi(self, rank, size=2):
        comm = _make_mock_comm(rank=rank, size=size)

        # allgather: simulate collecting from all ranks.
        # In our single-process test we only have the local chunk, so
        # we duplicate it to mimic two ranks producing data.
        def _allgather(data):
            return [data, data]

        comm.allgather.side_effect = _allgather
        return comm, *_make_mock_mpi(comm)

    def _call(self, rank, rd=0):
        comm, mock_mpi4py, mock_MPI = self._make_comm_and_mpi(rank)
        with patch.dict(
            sys.modules,
            {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI},
        ):
            from polyswyft.utils import resimulate_deadpoints

            thetas, Ds = resimulate_deadpoints(self.deadpoints, self.settings, self.sim, rd=rd)
        return thetas, Ds, comm

    def test_allgather_called_when_size_gt_1(self):
        _, _, comm = self._call(rank=0)
        comm.allgather.assert_called_once()

    def test_rank_zero_saves_files(self):
        self._call(rank=0)
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_0"
        assert os.path.exists(f"{rd_dir}/{self.settings.targetKey}.npy")
        assert os.path.exists(f"{rd_dir}/{self.settings.obsKey}.npy")

    def test_non_rank_zero_does_not_save(self):
        self._call(rank=1)
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_0"
        assert not os.path.exists(f"{rd_dir}/{self.settings.targetKey}.npy")
        assert not os.path.exists(f"{rd_dir}/{self.settings.obsKey}.npy")

    def test_bcast_called_on_all_ranks(self):
        _, _, comm = self._call(rank=1)
        assert comm.bcast.call_count == 2

    def test_noise_resampling_path(self):
        self.settings.use_noise_resampling = True
        self.settings.n_noise_resampling_samples = 3

        # Setup resampler mock
        resampler = MagicMock()
        d = self.settings.num_features_dataset
        rng = np.random.default_rng(99)

        def _resample(cond):
            return swyft.Sample(
                **{
                    self.settings.targetKey: cond[self.settings.targetKey].ravel(),
                    self.settings.obsKey: rng.standard_normal(d),
                }
            )

        resampler.side_effect = _resample
        self.sim.get_resampler.return_value = resampler

        # rd > 0 triggers noise resampling
        rd_dir = f"{self.settings.root}/{self.settings.child_root}_1"
        os.makedirs(rd_dir, exist_ok=True)

        thetas, Ds, comm = self._call(rank=0, rd=1)
        # get_resampler called once per deadpoint (5 for rank 0 with size=2)
        assert self.sim.get_resampler.call_count == 5
        assert thetas.shape[0] > 0


# ============================================================================
# PolySwyft.__init__ — MPI
# ============================================================================
class TestPolySwyftInitMPI:
    def test_raises_import_error_when_mpi4py_missing(self):
        with patch.dict(sys.modules, {"mpi4py": None, "mpi4py.MPI": None}):
            from polyswyft.core import PolySwyft

            with pytest.raises(ImportError, match="mpi4py"):
                PolySwyft(
                    polyswyftSettings=MagicMock(),
                    sim=MagicMock(),
                    obs=MagicMock(),
                    deadpoints=np.zeros((5, 2)),
                    network=MagicMock(),
                    polyset=MagicMock(),
                    callbacks=MagicMock(),
                )

    def test_stores_comm_and_rank(self):
        comm = _make_mock_comm(rank=3, size=4)
        mock_mpi4py, mock_MPI = _make_mock_mpi(comm)

        with patch.dict(
            sys.modules,
            {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI},
        ):
            from polyswyft.core import PolySwyft

            ps = PolySwyft(
                polyswyftSettings=MagicMock(),
                sim=MagicMock(),
                obs=MagicMock(),
                deadpoints=np.zeros((5, 2)),
                network=MagicMock(),
                polyset=MagicMock(),
                callbacks=MagicMock(),
            )
            assert ps._rank == 3
            assert ps._comm is comm
