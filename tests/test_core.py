"""Control-flow tests for the PolySwyft orchestrator that need no MPI."""

from types import SimpleNamespace

import polyswyft.core as core
from polyswyft.core import PolySwyft


def _orchestrator(settings, **attrs):
    """Build a PolySwyft without running __init__ (which imports mpi4py)."""
    ps = PolySwyft.__new__(PolySwyft)
    ps.polyswyftSettings = settings
    ps.network_storage, ps.root_storage, ps.samples_storage = {}, {}, {}
    ps.dkl_storage, ps.dkl_compression_storage = {}, {}
    ps.current_key, ps.previous_key = "current", "previous"
    ps.deadpoints_processing = None
    for key, value in attrs.items():
        setattr(ps, key, value)
    return ps


def test_cyclic_kl_cold_start_requires_both_criteria(polyswyft_settings):
    polyswyft_settings.cyclic_rounds = False
    polyswyft_settings.NRE_start_from_round = 0
    polyswyft_settings.termination_abs_dkl = 0.2
    polyswyft_settings.termination_min_compression = 1.0
    ps = _orchestrator(polyswyft_settings)
    kl = {1: 5.0, 2: 0.1, 3: 0.05}  # rd=2 is already below the KL threshold ...
    compression = {0: 0.5, 1: 0.5, 2: 0.5, 3: 2.0}  # ... but only rd=3 is compressed enough

    def fake_cycle(rd):
        if rd > 0:
            ps.dkl_storage[rd] = (kl[rd], 0.0)
        ps.dkl_compression_storage[rd] = (compression[rd], 0.0)

    ps._cycle = fake_cycle
    ps._cyclic_kl()  # round 0 has no KL(P_0 || P_-1): must not KeyError
    assert polyswyft_settings.NRE_num_retrain_rounds == 3


def test_resume_reads_enhanced_chains_and_applies_deadpoints_processing(
    polyswyft_settings, network, fake_nested_samples, monkeypatch, tmp_path
):
    polyswyft_settings.root = str(tmp_path / "resume_root")
    polyswyft_settings.NRE_start_from_round = 1
    polyswyft_settings.use_livepoint_increasing = True
    seen = {}

    def fake_read_chains(root):
        seen["root"] = root
        return fake_nested_samples.copy()

    def processing(samples, rd):
        seen["rd"] = rd
        return samples.iloc[:7]

    monkeypatch.setattr(core.anesthetic, "read_chains", fake_read_chains)
    monkeypatch.setattr(core.torch, "load", lambda *a, **k: network.state_dict())
    monkeypatch.setattr(core.pickle, "dump", lambda *a, **k: None)

    ps = _orchestrator(
        polyswyft_settings,
        polyset=SimpleNamespace(file_root="samples", seed=1),
        network_model=network,
        deadpoints_processing=processing,
    )
    ps._cyclic_rounds = lambda: None
    ps.execute_NSNRE_cycle()

    assert seen["root"].endswith(f"round_0/{polyswyft_settings.increased_livepoints_fileroot}/samples")
    assert seen["rd"] == 0
    assert ps.deadpoints_samples.shape == (7, polyswyft_settings.num_features)
    # stored chains stay unprocessed, exactly as at the end of ``_cycle``
    assert len(ps.samples_storage["previous"]) == len(fake_nested_samples)
