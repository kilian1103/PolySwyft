# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-04-15

### Fixed

- Split branch and tag push in version-bump workflow to correctly trigger publish workflow

## [0.2.2] - 2026-04-15

### Fixed

- Add PyPI version badge to README
- Automate GitHub Release creation on publish via softprops/action-gh-release

## [0.2.1] - 2026-04-15

### Fixed

- Quote all extras specs in docs to prevent zsh glob expansion
- Revert CI install to local source build

## [0.2.0] - 2026-04-15

### Added

- PyPI publish workflow using OIDC Trusted Publisher
- GitHub environment protection rules for publish job
- PyPI package classifiers (Python 3.9/3.10/3.11, Apache License, OS Independent)

### Fixed

- Update all install references to use PyPI package (`pip install polyswyft`)
- Add uv install instructions to README
- Add PolyChord third-party license notice to README
- Update ImportError message to point to `pip install "polyswyft[mpi]"`
- Clean up requirements.txt

## [0.1.3] - 2026-04-14

### Added

- MPI test coverage

## [0.1.0] - 2025-12-10

### Added

- Initial release of PolySwyft as an installable Python package
- Sequential Neural Ratio Estimation (NSNRE) combining PolyChord and swyft
- Multi-round training with progressive improvement
- KL-divergence convergence monitoring
- Modular design with `PolySwyftNetwork` ABC and `PolySwyftSettings`
- Examples for Multivariate Gaussian, Gaussian Mixture Model, and CMB cosmology
- Weights & Biases experiment tracking support
- MPI parallelization support via mpi4py
