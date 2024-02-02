import swyft
import torch
from lsbi.model import LinearModel

from NSLFI.NRE_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings):
        super().__init__()
        self.nreSettings = nreSettings
        self.n = self.nreSettings.num_features
        self.d = self.nreSettings.num_features_dataset
        self.m = torch.randn(self.d) * 3  # mean vec of dataset
        self.M = torch.randn(size=(self.d, self.n))  # transform matrix of dataset to parameter vee
        self.C = torch.eye(self.d)  # cov matrix of dataset
        # C very small, or Sigma very big
        self.mu = torch.zeros(self.n)  # mean vec of parameter prior
        self.Sigma = 100 * torch.eye(self.n)  # cov matrix of parameter prior
        self.Sigma_inv = torch.inverse(self.Sigma)
        self.C_inv = torch.inverse(self.C)
        self.model = LinearModel(M=self.M, C=self.C, Sigma=self.Sigma, mu=self.mu, m=self.m, n=self.n, d=self.d)
        # self.z_sampler = stats.multivariate_normal(mean=self.mu, cov=self.Sigma).rvs
        self.z_sampler = self.model.prior().rvs

    def xgivenz(self, z):
        # return stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).rvs()
        return self.model.likelihood(z).rvs()

    def logratio(self, x, z):
        # loglike = stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).logpdf(x)
        # logevidence = stats.multivariate_normal(mean=(self.m + self.M @ self.mu),
        #                                         cov=(self.C + self.M @ self.Sigma @ self.M.T)).logpdf(x)
        # logratio = loglike - logevidence
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def zgivenx(self, x):
        """Posterior weights"""
        # D0 = self.m + self.M @ self.mu
        # mean = self.mu + self.Sigma @ self.M.T @ np.linalg.inv(self.C) @ (torch.as_tensor(x).float() - D0)
        # post = stats.multivariate_normal(mean=mean, cov=(self.Sigma_inv + self.M.T @ self.C_inv @
        #                                                  self.M).inverse())
        #
        # return post.rvs()
        return self.model.posterior(x).rvs()

    def build(self, graph):
        z = graph.node(self.nreSettings.targetKey, self.z_sampler)
        x = graph.node(self.nreSettings.obsKey, self.xgivenz, z)
        post = graph.node(self.nreSettings.posteriorsKey, self.zgivenx, x)
        l = graph.node(self.nreSettings.contourKey, self.logratio, x, post)
