import numpy as np
import swyft
from lsbi.model import MixtureModel

from polyswyft.settings import PolySwyftSettings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyftSettings, model: MixtureModel):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.n = self.polyswyftSettings.num_features
        self.d = self.polyswyftSettings.num_features_dataset
        self.a = self.polyswyftSettings.num_mixture_components
        self.model = model

    def logratio(self, x, z):
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def posterior(self, x):
        post = self.model.posterior(x)
        # lsbi may squeeze scalar dims: force shape (n,) even when n == 1.
        return np.atleast_1d(post.rvs())

    def prior(self):
        # lsbi may squeeze scalar dims: force shape (n,) even when n == 1.
        return np.atleast_1d(self.model.prior().rvs())

    def likelihood(self, z):
        # lsbi may squeeze scalar dims: force shape (d,) even when d == 1.
        return np.atleast_1d(self.model.likelihood(z).rvs())

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)
        # posterior
        post = graph.node(self.polyswyftSettings.posteriorsKey, self.posterior, x)
        # logratio
        l = graph.node(self.polyswyftSettings.contourKey, self.logratio, x, post)
