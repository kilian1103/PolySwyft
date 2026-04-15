# Nested Sampling Neural Ratio Estimator (NSNRE)

<div align="center">

[![CI](https://github.com/kilian1103/PolySwyft/actions/workflows/ci.yml/badge.svg)](https://github.com/kilian1103/PolySwyft/actions/workflows/ci.yml)
[![Lint](https://github.com/kilian1103/PolySwyft/actions/workflows/lint.yml/badge.svg)](https://github.com/kilian1103/PolySwyft/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2512.08316-b31b1b.svg)](https://arxiv.org/abs/2512.08316)

</div>

## PolySwyft

**PolySwyft** is an implementation of a Nested Sampling Neural-Ratio-Estimator (NSNRE) that combines
[PolyChord](https://github.com/PolyChord/PolyChordLite) and [swyft](https://github.com/undark-lab/swyft) in a
PyTorch-based framework.

### Key Features

- **Sequential Neural Ratio Estimation**: Advanced nested sampling with neural networks, MPI compatibility
- **Multi-round Training**: Progressive improvement through multiple training rounds
- **Comprehensive Diagnostics**: Built-in KL-divergence and convergence monitoring
- **Modular Design**: Subclass `PolySwyftNetwork` and `swyft.Simulator` for your own problems
- **Experiment Tracking**: Integrated Weights & Biases support

## Installation

### Install the core package

```bash
# pip
pip install polyswyft

# uv
uv add polyswyft
```

This installs the `polyswyft` package with its core dependencies. You can then import:

```python
from polyswyft import PolySwyft, PolySwyftNetwork, PolySwyftSettings
```

### Optional dependencies

```bash
# pip
pip install "polyswyft[mpi]"       # MPI support (mpi4py)
pip install "polyswyft[examples]"  # lsbi for MVG/GMM examples
pip install "polyswyft[cmb]"       # cmblike for CMB example
pip install "polyswyft[wandb]"     # Weights & Biases tracking
pip install "polyswyft[dev]"       # pytest for running tests
pip install "polyswyft[all]"       # everything

# uv
uv add "polyswyft[mpi]"
uv add "polyswyft[examples]"
uv add "polyswyft[all]"            # etc.
```

### PolyChordLite

PolyChord must be installed separately:

```bash
git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
make
pip install .
cd .. && rm -rf PolyChordLite
```

### CMB Likelihood Library (for CMB example only)

```bash
git clone https://github.com/htjb/cmb-likelihood.git
cd cmb-likelihood
pip install .
cd .. && rm -rf cmb-likelihood
```

## Quick Start

The repository has two parts:

1. **`polyswyft/`** — the pip-installable core algorithm
2. **`examples/`** — paper examples demonstrating how to use the core algorithm

To use PolySwyft for your own problem, you need to:

1. **Subclass `swyft.Simulator`** — define your forward model (prior, likelihood)
2. **Subclass `PolySwyftNetwork`** — define your network architecture, prior transform, and log-ratio function
3. **Wire them together** with `PolySwyftSettings` and `PolySwyft`

See `examples/mvg/` for the simplest example, or `main.ipynb` for a step-by-step notebook walkthrough.

### Running the examples

Each example domain (MVG, GMM, CMB) has three scripts:

```bash
# Multivariate Gaussian example
cd examples/mvg
mpirun -n 4 python run_polyswyft.py   # PolySwyft (NSNRE)
mpirun -n 4 python run_swyft.py       # swyft baseline (TNRE)
mpirun -n 4 python run_polychord.py   # PolyChord baseline (exact likelihood)
```

> **Important**: Ensure your parameter dimensions follow the shape `(n, dim(theta))` and `(n, dim(D))` respectively,
> where `n` is the number of samples in your Simulator class. This is particularly important when `dim(theta)` and/or
> `dim(D) = 1`.

## Project Structure

```
polyswyft/                  # core pip-installable package
    __init__.py             # public API
    core.py                 # PolySwyft class (NSNRE cycle)
    network.py              # PolySwyftNetwork ABC
    settings.py             # PolySwyftSettings
    dataloader.py           # PolySwyftDataModule + PolySwyftSequential
    utils.py                # KL divergence, deadpoints, plotting reload

examples/                   # paper examples (not pip-installed)
    mvg/                    # Multivariate Gaussian
        simulator.py        # swyft.Simulator subclass (lsbi LinearModel)
        network.py          # PolySwyftNetwork subclass
        run_polyswyft.py    # main PolySwyft run script
        run_swyft.py        # swyft TNRE baseline
        run_polychord.py    # PolyChord baseline
    gmm/                    # Gaussian Mixture Model
        ...                 # same structure
    cmb/                    # CMB Cosmology
        ...                 # same structure
    plotting.py             # shared plotting utilities

tests/                      # test suite
main.ipynb                  # notebook walkthrough
```

### Core Components

**`PolySwyftNetwork`** (`polyswyft/network.py`) — Abstract base class extending `swyft.SwyftModule` with PolyChord compatibility. Implement these methods:

- `forward(A, B)` — swyft forward pass computing log-ratios
- `prior(cube)` — unit cube to prior transform for PolyChord
- `logRatio(theta)` — NRE log-ratio for PolyChord (replaces the standard log-likelihood)
- `get_new_network()` — factory for fresh network instances

**`PolySwyft`** (`polyswyft/core.py`) — Orchestrates the sequential NSNRE cycle: train NRE, run PolyChord with the trained ratio, use deadpoints for next round.

**`PolySwyftDataModule`** (`polyswyft/dataloader.py`) — Multi-round data management. Handles progressive training across multiple rounds with efficient batch selection.

**`PolySwyftSettings`** (`polyswyft/settings.py`) — Configuration for all algorithm parameters.

### Output Structure

When PolySwyft runs, it creates a structured output directory:

```
project_root/
    round_0/
        NRE_network.pt    # neural network weights
        optimizer_file.pt  # optimizer state
        x.npy              # data samples
        z.npy              # parameter samples with (z,x) ~ p(theta,D)
        samples.*          # nested sampling output (anesthetic compatible)
        wandb/             # Weights & Biases logs
    round_1/
        ...
    settings.pkl           # polyswyft settings pickle file
```

Each `round_i/` folder contains the trained network, optimizer state, deadpoints from nested sampling, and joint samples used for training. Round 0 contains full prior samples; subsequent rounds contain samples created via deadpoints from previous rounds. The dataloader randomly selects batches from all rounds up to and including the current round.

## Contributors

### Main Contributor

- **Kilian Scheutwinkel** - _Lead Developer & Researcher_

### Project Supervisors

- **Will Handley** - _Project Supervisor_
- **Christoph Weniger** - _Project Supervisor_
- **Eloy de Lera Acedo** - _Project Supervisor_

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

> **Third-party notice:** PolySwyft depends on [PolyChord](https://github.com/PolyChord/PolyChordLite), which is licensed separately under the PolyChord License Agreement for non-commercial, academic, and research use only. Commercial use of PolyChord requires a separate license from its authors. See the `THIRD-PARTY DEPENDENCY NOTICE` section at the bottom of [LICENSE](LICENSE) for the full notice.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to
discuss what you would like to change.

## Citation

If you use PolySwyft in your research, please cite:

```bibtex
@article{scheutwinkel_polyswyft_2025,
    title = {{PolySwyft}: sequential simulation-based nested sampling},
    shorttitle = {{PolySwyft}},
    url = {http://arxiv.org/abs/2512.08316},
    doi = {10.48550/arXiv.2512.08316},
    abstract = {We present PolySwyft, a novel, non-amortised simulation-based inference framework that unites the strengths of nested sampling (NS) and neural ratio estimation (NRE) to tackle challenging posterior distributions when the likelihood is intractable but a forward simulator is available. By nesting rounds of NRE within the exploration of NS, and employing a principled KL-divergence criterion to adaptively terminate sampling, PolySwyft achieves faster convergence on complex, multimodal targets while rigorously preserving Bayesian validity. On a suite of toy problems with analytically known posteriors of a dim(theta,D)=(5,100) multivariate Gaussian and multivariate correlated Gaussian mixture model, we demonstrate that PolySwyft recovers all modes and credible regions with fewer simulator calls than swyft's TNRE. As a real-world application, we infer cosmological parameters dim(theta,D)=(6,111) from CMB power spectra using CosmoPower. PolySwyft is released as open-source software, offering a flexible toolkit for efficient, accurate inference across the astrophysical sciences and beyond.},
    urldate = {2025-12-10},
    publisher = {arXiv},
    author = {Scheutwinkel, Kilian H. and Handley, Will and Weniger, Christoph and Acedo, Eloy de Lera},
    month = dec,
    year = {2025},
    note = {arXiv:2512.08316 [astro-ph]},
    keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
}
```

## Support

For questions, issues, or contributions, please:

- Open an issue on GitHub
- Contact the main contributor: [Kilian Scheutwinkel](mailto:khs40@cantab.ac.uk)
