# CLAUDE.md

PolySwyft is a sequential simulation-based inference framework that combines nested sampling (PolyChord) with neural ratio estimation (swyft) to recover posterior distributions when the likelihood is intractable.

## Core Algorithm (NSNRE Cycle)

1. **Initialise** -- sample from prior to create initial training dataset
2. **Simulate** -- forward-simulate joint pairs (theta, D) using deadpoints
3. **Train NRE** -- train binary classifier on joint vs disjoint pairs (cumulative dataset across rounds)
4. **Run NS** -- run PolyChord using the trained NRE log-ratio as the likelihood, producing new deadpoints
5. **Terminate** -- stop when KL(P_i || P_{i-1}) ~ 0 with sufficient compression KL(P_i || prior) > C_comp; otherwise go to step 2

Key equations:
- Ratio estimate: r(theta, D) = L/Z = P/prior (likelihood-to-evidence ratio)
- Prior correction: r = r*/Z_NS where Z_NS normalises for training on dead measure != prior
- Posterior: P = (r*/Z_NS) * prior

## Package Structure

- `polyswyft/core.py` -- `PolySwyft` class orchestrating the NSNRE cycle
- `polyswyft/network.py` -- `PolySwyftNetwork` ABC (implement: forward, prior, logRatio, get_new_network)
- `polyswyft/settings.py` -- `PolySwyftSettings` configuration
- `polyswyft/dataloader.py` -- `PolySwyftDataModule` + `PolySwyftSequential` (multi-round data loading)
- `polyswyft/utils.py` -- KL divergence computation, deadpoint resimulation, plotting reload
- `examples/` -- MVG, GMM, CMB examples (not pip-installed)
- `tests/` -- unit tests

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. When creating a PR, always add an entry under `## [Unreleased]` (create the section if it doesn't exist, directly below the `# Changelog` header). Use these subsections as needed: `### Added`, `### Fixed`, `### Changed`, `### Removed`. On version bump, `version-bump.yml` runs `git-cliff` to automatically convert `[Unreleased]` into the versioned section and regenerate the full file — so PR authors only ever write to `[Unreleased]`.

## Development

```bash
pip install "polyswyft[dev,examples]"    # install with test + example deps
pytest -m "not integration and not slow" -v  # run unit tests
ruff check . && ruff format --check .    # lint
```

- PolyChord must be installed separately (build from source)
- MPI (`mpi4py`) required for full PolySwyft runs; unit tests do not require MPI
- MPI integration tests (require `mpi4py` + `mpirun`):
  ```bash
  mpirun -n 1 pytest tests/test_mpi_integration.py -m integration -v
  mpirun -n 2 pytest tests/test_mpi_integration.py -m integration -v
  ```
- Version is auto-bumped on merge to master via `.github/workflows/version-bump.yml` (conventional commits: `feat:` -> minor, `fix:` -> patch, `BREAKING CHANGE`/`feat!:` -> major)
- Version lives in both `pyproject.toml` and `polyswyft/__init__.py`
