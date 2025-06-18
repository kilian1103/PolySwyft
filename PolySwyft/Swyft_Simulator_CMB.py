import numpy as np
import swyft
#from cmblike.cmb import CMB
import scipy.stats
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, cmbs, bins, bin_centers, p_noise, prior_mins, prior_maxs,
                 cp=False, bounds=None):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings

        self.prior_rect_sampler = swyft.RectBoundSampler([scipy.stats.uniform(loc=prior_mins,
                                                                  scale= prior_maxs-prior_mins)],
                                                            bounds=bounds)
        self.cmbs = cmbs
        self.bins = bins
        self.bin_centers = bin_centers
        self.p_noise = p_noise
        self.conversion = self.bin_centers * (self.bin_centers + 1) / (2 * np.pi)
        self.cp = cp

    def prior(self):
        return self.prior_rect_sampler()

    def likelihood(self, theta):
        cltheory, sample = self.cmbs.get_samples(theta, self.bins, noise=self.p_noise, cp=self.cp)
        return sample * self.conversion

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)
