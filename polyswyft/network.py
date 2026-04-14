from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import swyft


class PolySwyftNetwork(swyft.SwyftModule):
    """Abstract base class for PolySwyft networks.

    Subclass this and implement the abstract methods to define your own
    network architecture, prior, and likelihood for use with PolySwyft.

    Example::

        class MyNetwork(PolySwyftNetwork):
            def __init__(self, settings, obs):
                super().__init__()
                self.settings = settings
                self.obs = obs
                self.optimizer_init = swyft.OptimizerInit(...)
                self.network = swyft.LogRatioEstimator_Ndim(...)

            def forward(self, A, B):
                return self.network(A["x"], B["z"])

            def prior(self, cube):
                return ...  # unit cube -> prior cube

            def logRatio(self, theta):
                ...  # return (logR, [])

            def get_new_network(self):
                return MyNetwork(self.settings, self.obs)
    """

    @abstractmethod
    def forward(self, A, B):
        """Swyft forward pass computing log-ratios."""
        ...

    @abstractmethod
    def prior(self, cube) -> np.ndarray:
        """Transform the unit cube to the prior for PolyChord."""
        ...

    @abstractmethod
    def logRatio(self, theta: np.ndarray) -> tuple[Any, list]:
        """Compute the NRE log-ratio for PolyChord."""
        ...

    @abstractmethod
    def get_new_network(self) -> PolySwyftNetwork:
        """Create a fresh instance of this network (same config, new weights)."""
        ...

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """Dumper function for PolyChord runtime progress. Override for custom behavior."""
        print(f"Last dead point: {dead[-1]}")
