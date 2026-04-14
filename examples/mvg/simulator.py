from __future__ import annotations

from typing import Union

import numpy as np
import swyft
from lsbi.model import LinearModel

from polyswyft.settings import PolySwyftSettings

_ArrayLike = Union[np.ndarray, "torch.Tensor"]  # torch imported lazily at call-sites


def _to_numpy(x) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a numpy ndarray."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Simulator(swyft.Simulator):
    def __init__(
        self,
        polyswyftSettings: PolySwyftSettings,
        m: _ArrayLike,
        M: _ArrayLike,
        C: _ArrayLike,
        mu: _ArrayLike,
        Sigma: _ArrayLike,
    ):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.n = self.polyswyftSettings.num_features
        self.d = self.polyswyftSettings.num_features_dataset
        self.model = LinearModel(
            M=_to_numpy(M),
            C=_to_numpy(C),
            Sigma=_to_numpy(Sigma),
            mu=_to_numpy(mu),
            m=_to_numpy(m),
            n=self.n,
            d=self.d,
        )

    def logratio(self, x, z):
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def posterior(self, x):
        post = self.model.posterior(x)
        # lsbi squeezes scalar dims: force shape (n,) even when n == 1.
        return np.atleast_1d(post.rvs())

    def prior(self):
        # lsbi squeezes scalar dims: force shape (n,) even when n == 1.
        return np.atleast_1d(self.model.prior().rvs())

    def likelihood(self, z):
        # lsbi squeezes scalar dims: force shape (d,) even when d == 1.
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
