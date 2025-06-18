# Nested sampling Neural Ratio estimator (NSNRE)

## PolySwyft

An implementation of a Nested Sampling Neural-Ratio-Estimator (NSNRE) using ``pypolychord``
and ``swyft`` named ``PolySwyft``in ``pytorch``.

This repository contains a self-contained example of ``PolySwyft`` that is executed via the ``main_xxx.py`` files.

For anyone wanting to have a more customisable version of the code, please refer to the notebook
``main.ipynb``. This notebook contains the same workflow as the ``main_xxx.py`` files, but is more modular and allows
for easier experimentation.

## Prerequisites

Before proceeding, ensure you have a working Conda environment.

```bash
conda create -n venv python=3.11
```

## Installation

The following steps will guide you through the installation of the required Python packages and libraries for this
project. The versioning for certain packages is critical for compatibility.
The environment is a minimally viable Conda environment that can be used to run the main.ipynb noetbook in this
repository.

### 1. Initial Package Installation

First, install the core Python dependencies with specific versions using `pip`:

```bash
pip install swyft==0.4.4 typing wandb anesthetic scikit-learn lsbi==0.9.0 mpi4py cosmopower jupyter notebook numpy==1.26.4 scipy==1.10.1
```

### 2. Install PolyChordLite

Next, you will need to clone and install PolyChordLite from its GitHub repository:

```bash
git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
make
pip install .
cd ..
rm -r PolyChordLite
```

Note: Installing PolyChord on macOS may encounter issues, use LLMs such as ChatGPT, Claude, or Gemini to help you

### 3. Install CMB Likelihood

Finally, clone and install the cmb-likelihood library:

```bash
git clone https://github.com/htjb/cmb-likelihood.git
cd cmb-likelihood
pip install .
cd ..
rm -r cmb-likelihood
```

### 4. Setup WandB

As I use [Weights & Biases (WandB)](https://wandb.ai/) for experiment tracking, you need to set up your WandB account
and login.
It is free to use for academic purposes. Once set up you need to setup the following environment variable in the ``
.env.example`` file.

```bash
vim .env.example
### setup WANDB_API_KEY
mv .env.example .env
```

## Code Structure

The code is structured as follows:

#### 1. A swyft-compatible Simulator class e.g. ``PolySwyft_Simulator_MultiGauss.py``.

- This is a standard ``swyft`` simulator class that takes in a set of parameters and returns a set of simulated data.

#### 2. A PolySwyft-compatible Network class e.g. ``PolySwyft_Network.py``.

- This is a ``Network(swyft.SwyftModule)`` class that extends its usual ``torch.nn.Module`` functionality of a
  ``forward`` method with ``PolyChord``compatible methods.
- For this class, one has to implement a ``prior``, ``loglikelihood``and ``dumper`` (optional) function that are methods
  needed for ``PolyChord``.
- The ``prior`` function receives a set of hypercube parameters in unit-space and returns a set of prior samples in the
  physical space.
- The ``loglikelihood`` function receives a set of physical parameters and returns the log-ratio of the data given the
  parameters.
- The ``dumper`` function is optional to monitor runtime progress of ``PolyChord``.

#### 3. A

``PolySwyft_Dataloader.py`` file that implement the dataloading accross multiple rounds for retraining the network.

#### 4. A ``utils.py``file to implement the

``PolySwyft``-specific functions e.g. KL-divergence diagnostics functions and

conditional sampling using deadpoints.

#### 5. When ``PolySwyft`` is run, it will automatically create a root folder in the current working directory.

This folder contains the following structure:

``round_i``folder containing:

- The neural network at round ``i``.
- Optimiser state at round ``i``.
- Deadpoints that were generated using the network at round ``i``.
- The joint samples ``p(\theta,D)_i`` used to train the network at round ``i`` saved as ``.npy``files. For instance, in
  round ``i=0``, the folder contains the full prior samples, and in ``i=1``the new joint samples created via the
  deadpoints at round ``i=0`` are stored. The dataloader randomly selects batches from all rounds until round ``i`` (
  inclusive)  to train the network at round ``i``.

