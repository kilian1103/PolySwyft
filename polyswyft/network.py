"""Abstract base class users subclass to plug a neural ratio estimator into
:class:`polyswyft.core.PolySwyft`.

Concrete subclasses must implement four hooks: :meth:`forward` (the
log-ratio computation used by swyft during training), :meth:`prior` (the
unit-cube → prior bijection used by PolyChord), :meth:`logRatio` (the
likelihood callback PolyChord queries), and :meth:`get_new_network`
(re-instantiation for fresh-weight rounds). See ``examples/mvg/network.py``
for a full reference implementation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import swyft


class PolySwyftNetwork(swyft.SwyftModule):
    """Abstract base class for the NRE consumed by PolySwyft.

    A subclass must own a swyft log-ratio estimator (typically
    ``swyft.LogRatioEstimator_Ndim`` or ``LogRatioEstimator_1dim``) plus an
    ``swyft.OptimizerInit``. The four abstract hooks below adapt that swyft
    network to PolyChord's calling convention.

    Example
    -------
    ::

        class MyNetwork(PolySwyftNetwork):
            def __init__(self, settings, obs):
                super().__init__()
                self.settings = settings
                self.obs = obs
                self.optimizer_init = swyft.OptimizerInit(
                    torch.optim.Adam, dict(lr=settings.learning_rate_init),
                    torch.optim.lr_scheduler.ExponentialLR,
                    dict(gamma=settings.learning_rate_decay),
                )
                self.network = swyft.LogRatioEstimator_Ndim(
                    num_features=settings.num_features_dataset,
                    marginals=(tuple(range(settings.num_features)),),
                    varnames=settings.targetKey,
                    dropout=settings.dropout, hidden_features=128, Lmax=0,
                )

            def forward(self, A, B):
                return self.network(A[self.settings.obsKey],
                                    B[self.settings.targetKey])

            def prior(self, cube):
                return self.settings.model.prior().bijector(x=cube)

            def logRatio(self, theta):
                ...
    """

    @abstractmethod
    def forward(self, A: dict, B: dict) -> Any:
        """Swyft forward pass returning the network log-ratio.

        Called by ``swyft.SwyftTrainer`` during training/validation.

        Parameters
        ----------
        A : dict
            Observation batch keyed by ``settings.obsKey`` containing
            tensors of shape ``(batch, num_features_dataset)``.
        B : dict
            Parameter batch keyed by ``settings.targetKey`` containing
            tensors of shape ``(batch, num_features)``.

        Returns
        -------
        swyft.LogRatioSamples
            The wrapped log-ratio estimate produced by the underlying
            ``swyft.LogRatioEstimator_*``.
        """
        ...

    @abstractmethod
    def prior(self, cube: np.ndarray) -> np.ndarray:
        """Map PolyChord's unit cube sample to the prior space.

        Parameters
        ----------
        cube : np.ndarray
            Unit-cube sample of shape ``(num_features,)`` with each
            component in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Parameter vector ``theta`` of shape ``(num_features,)`` drawn
            from the prior corresponding to ``cube``.
        """
        ...

    @abstractmethod
    def logRatio(self, theta: np.ndarray) -> tuple[Any, list]:
        """PolyChord likelihood callback returning the NRE log-ratio.

        Evaluated once per PolyChord live-point proposal. PolyChord uses
        the returned scalar as ``log L`` for nested sampling; under the
        likelihood-to-evidence-ratio identity this is equivalent up to a
        constant ``log Z_NS`` per round (see paper eq. 11).

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector of shape ``(num_features,)``.

        Returns
        -------
        tuple[float, list]
            ``(log_r, derived)`` where ``log_r`` is the scalar log-ratio
            and ``derived`` is a list of derived parameters (empty when
            ``settings.nderived == 0``).
        """
        ...

    @abstractmethod
    def get_new_network(self) -> PolySwyftNetwork:
        """Return a freshly-constructed instance with the same configuration.

        Used at the start of each NSNRE round to obtain a network with
        re-initialised weights. The returned instance must share
        ``settings`` and ``obs`` with ``self`` but must not share
        parameters.

        Returns
        -------
        PolySwyftNetwork
            A new instance ready to load a state-dict (when
            ``continual_learning_mode=True``) or to train from scratch.
        """
        ...

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """PolyChord dumper hook called periodically during nested sampling.

        Default implementation prints the most recent dead point. Override
        to integrate with custom progress logging.

        Parameters
        ----------
        live : np.ndarray
            Current live-point set, shape ``(n_live, num_features + nderived + 2)``.
        dead : np.ndarray
            Dead-point set accumulated so far, same column layout as
            ``live``.
        logweights : np.ndarray
            Log nested-sampling weights of the dead points.
        logZ : float
            Current evidence estimate.
        logZerr : float
            Current evidence uncertainty.
        """
        print(f"Last dead point: {dead[-1]}")
