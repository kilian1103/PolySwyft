import scipy.stats
import swyft
from lsbi.model import LinearModel

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, bounds=None, model: LinearModel = None):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        #self.C_inv = torch.inverse(C)
        self.model = model
        self.prior_rect_sampler = swyft.RectBoundSampler(
            [scipy.stats.norm(loc=self.model.prior().mean, scale=10)]
        , bounds=bounds)

    def prior(self):
        return self.prior_rect_sampler()

    def likelihood(self, z):
        return self.model.likelihood(z).rvs()

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)