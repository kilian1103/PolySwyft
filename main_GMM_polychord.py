import logging

import numpy as np
import swyft
from lsbi.model import MixtureModel
from mpi4py import MPI
from pypolychord import PolyChordSettings, run_polychord
from pytorch_lightning import seed_everything
from scipy.stats import wishart

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_MixGaussMultiPost import Simulator


###requires lsbi==0.12.0 for reproducibility
def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    root = "GMM_PolyChord"
    polyswyftSettings = PolySwyft_Settings(root=root)
    polyswyftSettings.seed = 250
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    n = polyswyftSettings.num_features
    d = polyswyftSettings.num_features_dataset
    a = polyswyftSettings.num_mixture_components
    polyswyftSettings.n_training_samples = 160_588  # sum needed for convergence in polyswyft
    # prior
    mu_theta = 5 * np.random.randn(a, n)
    Sigma = 0.1 * np.eye(n)
    Sigma = wishart.rvs(df=n + 1, scale=Sigma, size=a)
    # component weights
    logA = np.random.uniform(size=a)
    logA = np.log(logA / np.sum(logA))
    # likelihood
    mu_data = np.zeros(shape=(a, d))
    M = 0.04 * np.random.randn(a, d, n)
    C = 4 * np.eye(d)
    model = MixtureModel(M=M, C=C, Sigma=Sigma, mu=mu_theta,
                         m=mu_data, logw=logA, n=n, d=d)
    sim = Simulator(polyswyftSettings=polyswyftSettings, model=model)
    polyswyftSettings.model = sim.model  # lsbi mode

    # generate training dat and obs
    obs = swyft.Sample(x=sim.model.evidence().rvs()[None, :])

    nlive = n * 500
    polyset = PolyChordSettings(nDims=n, nDerived=0, nlive=nlive, base_dir=root)

    def loglikelihood(theta):
        return sim.model.likelihood(theta).logpdf(obs[polyswyftSettings.obsKey].squeeze()), []

    def prior(cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        theta = sim.model.prior().bijector(x=cube)
        return theta

    def dumper(live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))

    run_polychord(loglikelihood=loglikelihood,
                  nDims=n,
                  nDerived=0, settings=polyset,
                  prior=prior, dumper=dumper)


if __name__ == '__main__':
    execute()
