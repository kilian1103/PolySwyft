import logging

import numpy as np
from cmblike.cmb import CMB
from cmblike.noise import planck_noise
from mpi4py import MPI
from pypolychord import run_polychord, PolyChordSettings
from pytorch_lightning import seed_everything

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_CMB import Simulator


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    root = "CMB_PolyChord"
    polyswyftSettings = PolySwyft_Settings(root)
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="a")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    # cosmopower params
    cp = True
    params = ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']  # cosmopower
    polyswyftSettings.num_features = len(params)
    prior_mins = [0.005, 0.08, 0.01, 0.8, 2.6, 0.5]
    prior_maxs = [0.04, 0.21, 0.16, 1.2, 3.8, 0.9]
    cmbs = CMB(path_to_cp="../cosmopower", parameters=params, prior_maxs=prior_maxs, prior_mins=prior_mins)

    # prepare binning
    l_max = 2508  # read from planck unbinned data
    first_bin_width = 1
    second_bin_width = 30
    divider = 30
    # bins = np.array([np.arange(2, l_max, 1), np.arange(2, l_max, 1)]).T  # 2 to 2508 unbinned
    first_bins = np.array([np.arange(2, divider, first_bin_width), np.arange(2, divider, first_bin_width)]).T  # 2 to 29
    second_bins = np.array([np.arange(divider, l_max - second_bin_width, second_bin_width),
                            np.arange(divider + second_bin_width, l_max, second_bin_width)]).T  # 30 to 2508
    last_bin = np.array([[second_bins[-1, 1], l_max]])  # remainder
    bins = np.concatenate([first_bins, second_bins, last_bin])
    # bin_centers = bins[:, 0]
    bin_centers = np.concatenate([first_bins[:, 0], np.mean(bins[divider - 2:], axis=1)])
    l = bin_centers.copy()
    polyswyftSettings.num_features_dataset = len(l)

    # planck noise
    # pnoise, _ = planck_noise().calculate_noise()
    pnoise = None

    # binned planck data, not using real data for now
    # planck = np.loadtxt('data/planck_unbinned.txt', usecols=[1])
    # planck = cmbs.rebin(planck, bins=bins)#

    sim = Simulator(polyswyftSettings=polyswyftSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers, p_noise=pnoise,
                    cp=cp)
    # obs = swyft.Sample(x=torch.as_tensor(planck)[None, :])

    # ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']
    theta_true = np.array([0.022, 0.12, 0.055, 0.965, 3.0, 0.67])
    Dell = sim.sample(conditions={polyswyftSettings.targetKey: theta_true})
    Cell = Dell[polyswyftSettings.obsKey] / sim.conversion

    nlive = polyswyftSettings.num_features * 100
    polyset = PolyChordSettings(nDims=polyswyftSettings.num_features, nDerived=0, nlive=nlive, base_dir=root)

    loglikelihood = cmbs.get_likelihood(data=Cell, l=l, bins=bins, noise=None, cp=True)

    def prior(cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        return cmbs.prior(cube=cube)

    def dumper(live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))

    run_polychord(loglikelihood=loglikelihood,
                  nDims=polyswyftSettings.num_features,
                  nDerived=0, settings=polyset,
                  prior=prior, dumper=dumper)


if __name__ == '__main__':
    main()
