import logging

import numpy as np
from cmblike.cmb import CMB
from mpi4py import MPI
from pypolychord import PolyChordSettings, run_polychord
from pytorch_lightning import seed_everything

from examples.cmb.simulator import Simulator
from polyswyft.settings import PolySwyftSettings


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    root = "CMB_PolyChord"
    polyswyftSettings = PolySwyftSettings(root)
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
    first_bins = np.array([np.arange(2, divider, first_bin_width), np.arange(2, divider, first_bin_width)]).T  # 2 to 29
    l_starts = np.arange(divider, l_max, second_bin_width)
    l_ends = np.arange(divider + second_bin_width, l_max + second_bin_width, second_bin_width)
    l_ends = np.clip(l_ends, a_min=None, a_max=l_max)
    second_bins = np.array([l_starts, l_ends[:len(l_starts)]]).T
    bins = np.concatenate([first_bins, second_bins])
    bins[:, 1] += 1  # Make upper edge exclusive for arange, e.g. [2,3) for l=2
    dof_per_bin = []
    for l_min, l_max_exclusive in bins:
        ells_in_bin = np.arange(l_min, l_max_exclusive)
        dof = np.sum(2 * ells_in_bin + 1)
        dof_per_bin.append(dof)

    dof_per_bin = np.array(dof_per_bin)

    bin_centers = np.concatenate([first_bins[:, 0], np.mean(bins[divider - 2:], axis=1)])
    l = bin_centers.copy()
    polyswyftSettings.num_features_dataset = len(l)

    # planck noise
    pnoise = None

    sim = Simulator(polyswyftSettings=polyswyftSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers, p_noise=pnoise,
                    cp=cp)

    # ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']
    theta_true = np.array([0.022, 0.12, 0.055, 0.965, 3.0, 0.67])
    Dell = sim.sample(conditions={polyswyftSettings.targetKey: theta_true})
    Cell = Dell[polyswyftSettings.obsKey] / sim.conversion

    nlive = polyswyftSettings.num_features * 100
    polyset = PolyChordSettings(nDims=polyswyftSettings.num_features, nDerived=0, nlive=nlive, base_dir=root)

    loglikelihood = cmbs.get_likelihood(data=Cell, l=l, bins=bins, noise=None, cp=True, dof_per_bin=dof_per_bin)

    def prior(cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        return cmbs.prior(cube=cube)

    def dumper(live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print(f"Last dead point: {dead[-1]}")

    run_polychord(loglikelihood=loglikelihood,
                  nDims=polyswyftSettings.num_features,
                  nDerived=0, settings=polyset,
                  prior=prior, dumper=dumper)


if __name__ == '__main__':
    main()
