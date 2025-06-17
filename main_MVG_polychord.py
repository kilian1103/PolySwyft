import logging

import numpy as np
import swyft
import torch
from mpi4py import MPI
from pypolychord import PolyChordSettings, run_polychord
from pytorch_lightning import seed_everything

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_MultiGauss import Simulator


###requires lsbi==0.9.0 for reproducibility
def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    root = "MVG_PolyChord"
    polyswyftSettings = PolySwyft_Settings(root)
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    d = polyswyftSettings.num_features_dataset
    n = polyswyftSettings.num_features

    m = torch.randn(d) * 3  # mean vec of dataset
    M = torch.randn(size=(d, n))  # transform matrix of dataset to parameter vee
    C = torch.eye(d)  # cov matrix of dataset
    # C very small, or Sigma very big
    mu = torch.zeros(n)  # mean vec of parameter prior
    Sigma = 100 * torch.eye(n)  # cov matrix of parameter prior
    sim = Simulator(polyswyftSettings=polyswyftSettings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)
    polyswyftSettings.model = sim.model  # lsbi model

    # generate training dat and obs
    obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))

    nlive = n * 100
    polyset = PolyChordSettings(nDims=n, nDerived=0, nlive=nlive, base_dir=root)

    def loglikelihood(theta):
        return sim.model.likelihood(theta).logpdf(obs[polyswyftSettings.obsKey].numpy().squeeze()), []

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
