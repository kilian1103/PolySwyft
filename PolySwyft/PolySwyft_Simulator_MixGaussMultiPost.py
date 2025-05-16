import swyft
from lsbi.model import MixtureModel

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, model: MixtureModel):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.n = self.polyswyftSettings.num_features
        self.d = self.polyswyftSettings.num_features_dataset
        self.a = self.polyswyftSettings.num_mixture_components
        self.model =  model



    def logratio(self, x, z):
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def posterior(self, x):
        post = self.model.posterior(x)
        return post.rvs()

    def prior(self):
        return self.model.prior().rvs()

    def likelihood(self, z):
        return self.model.likelihood(z).rvs()

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)
        # posterior
        post = graph.node(self.polyswyftSettings.posteriorsKey, self.posterior, x)
        # logratio
        l = graph.node(self.polyswyftSettings.contourKey, self.logratio, x, post)
