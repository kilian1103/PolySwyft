# 🧠 Nested Sampling Neural Ratio Estimator (NSNRE)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org)

</div>

## 🚀 PolySwyft

**PolySwyft** is a cutting-edge implementation of a Nested Sampling Neural-Ratio-Estimator (NSNRE) that combines the
power of [PolyChord](https://github.com/PolyChord/PolyChordLite) and [swyft](https://github.com/undark-lab/swyft) in a
PyTorch-based framework.

### ✨ Key Features

- 🔄 **Sequential Neural Ratio Estimation**: Advanced nested sampling with neural networks, MPI compatibility
- 🎯 **Multi-round Training**: Progressive improvement through multiple training rounds
- 📊 **Comprehensive Diagnostics**: Built-in KL-divergence and convergence monitoring
- 🔧 **Modular Design**: Easy to customize and extend for different problems
- 📈 **Experiment Tracking**: Integrated Weights & Biases support

### 📋 Quick Start

This repository contains self-contained examples that can be executed via the `main_xxx.py` files.

For a more customizable experience, check out the **`main.ipynb`** notebook, which provides the same workflow with
enhanced modularity for easier experimentation.

> ⚠️ **Important**: Ensure your parameter dimensions follow the shape `(n, dim(θ))` and `(n, dim(D))` respectively,
> where `n` is the number of samples in your Simulator class. This is particularly important when `dim(θ)` and/or
`dim(D) = 1`.

## 🛠️ Installation

### Prerequisites

Before proceeding, ensure you have a working Conda environment:

```bash
conda create -n polyswyft python=3.11
conda activate polyswyft
```

### Step-by-Step Installation

The following steps will guide you through installing all required dependencies. **Version compatibility is critical**
for proper functionality.

#### 1️⃣ Core Dependencies

Install the essential Python packages with specific versions:

```bash
pip install swyft==0.4.4 typing wandb anesthetic lsbi==0.9.0 mpi4py cosmopower jupyter notebook numpy==1.26.4 scipy==1.10.1
```

#### 2️⃣ PolyChordLite Installation

Clone and install PolyChordLite from the official repository:

```bash
git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
make
pip install .
cd ..
rm -r PolyChordLite
```

> ⚠️ **macOS Users**: PolyChord installation on macOS may encounter issues. Consider using LLMs (Gemini, ChatGPT,
> Claude) for troubleshooting specific macOS-related problems.

#### 3️⃣ CMB Likelihood Library (Optional)

Install the CMB likelihood library:

```bash
git clone https://github.com/htjb/cmb-likelihood.git
cd cmb-likelihood
pip install .
cd ..
rm -r cmb-likelihood
```

#### 4️⃣ Weights & Biases Setup

Configure [Weights & Biases](https://wandb.ai/) for experiment tracking (free for academic use):

```bash
# Create environment file
vim .env.example
# Add your WandB API key
echo "WANDB_API_KEY=your_api_key_here" >> .env.example
mv .env.example .env
```

## 🏗️ Code Architecture

The PolySwyft framework is built with a modular architecture consisting of four core components:

### 🔧 Core Components

#### 1️⃣ **Simulator Class** (`PolySwyft_Simulator_MultiGauss.py`)

- **Purpose**: Standard `swyft`-compatible simulator
- **Functionality**: Takes parameter sets and returns simulated data
- **Usage**: Define your forward model here

#### 2️⃣ **Network Class** (`PolySwyft_Network.py`)

- **Purpose**: Extends `swyft.SwyftModule` with PolyChord compatibility
- **Key Methods**:
    - `prior()`: Converts hypercube parameters to physical space
    - `loglikelihood()`: Computes log-ratio of data given parameters
    - `dumper()`: Optional runtime monitoring for PolyChord
- **Integration**: Seamlessly bridges neural networks with nested sampling

#### 3️⃣ **Data Loader** (`PolySwyft_Dataloader.py`)

- **Purpose**: Multi-round data management
- **Features**: Handles progressive training across multiple rounds
- **Optimization**: Efficient batch selection from all previous rounds

#### 4️⃣ **Utilities** (`utils.py`)

- **Purpose**: PolySwyft-specific helper functions
- **Features**:
    - KL-divergence diagnostics
    - Conditional sampling using deadpoints
    - Convergence monitoring tools

### 📁 Output Structure

When PolySwyft runs, it automatically creates a structured output directory:

```
project_root/
├── round_0/
│   ├── NRE_network.pt    # neural network weights
│   ├── optimizer_file.pt # optimizer state
│   ├── x.npy             # data samples
│   ├── z.npy             # parameter samples  with (z,x) ~ p(θ,D)
│   ├── samples.*         # output from nested sampling, anesthetic compatible
│   └── wandb             # Weights & Biases logs
├── round_1/
│   └── ...
└── settings.pkl          # polyswyft settings pickle file
```

**Each `round_i/` folder contains**:

- 🧠 **Neural Network**: Trained model at round `i`
- ⚙️ **Optimizer State**: Training state for resuming
- 💀 **Deadpoints**: Samples generated using network at round `i`
- 📊 **Joint Samples**: `p(θ,D)_i` used for training network at round `i` (saved as `.npy` files)

> **Note**: Round `i=0` contains full prior samples, while subsequent rounds contain samples created via deadpoints from
> previous rounds. The dataloader randomly selects batches from all rounds up to and including the current round.

## 👥 Contributors

### Main Contributor

- **Kilian Scheutwinkel** - _Lead Developer & Researcher_

### Project Supervisors

- **Will Handley** - _Project Supervisor_
- **Christoph Weniger** - _Project Supervisor_

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to
discuss what you would like to change.

## 📚 Citation

If you use PolySwyft in your research, please cite:

```bibtex
@software{polyswyft2025,
  title={PolySwyft: sequential simulation-based nested sampling},
  author={Scheutwinkel, Kilian and Handley, Will and Weniger, Christoph},
  year={2025},
  url={https://github.com/kilian1103/PolySwyft}
}
```

## 📞 Support

For questions, issues, or contributions, please:

- Open an issue on GitHub
- Contact the main contributor: [Kilian Scheutwinkel](mailto:hikarukilian@gmail.com)

---

<div align="center">
  <p><strong>Made with ❤️ for the scientific community</strong></p>
  <p>⭐ Star this repository if you find it useful!</p>
</div>
