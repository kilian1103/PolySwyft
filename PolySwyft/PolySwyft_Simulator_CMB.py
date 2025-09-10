import numpy as np
import swyft
from cmblike.cmb import CMB

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, cmbs: CMB, bin_centers, p_noise, bins=None, cp=False):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.cmbs = cmbs
        self.bins = bins
        self.bin_centers = bin_centers
        self.p_noise = p_noise
        self.cp = cp

    def prior(self):
        cube = np.random.uniform(0, 1, self.polyswyftSettings.num_features)
        theta = self.cmbs.prior(cube=cube)
        return theta

    def likelihood(self, theta):
        cltheory, sample = self.cmbs.get_samples(theta, self.bins, noise=self.p_noise, cp=self.cp, dell_conversion=True)
        return sample

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)
