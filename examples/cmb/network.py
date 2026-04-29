"""Reference :class:`PolySwyftNetwork` for the CMB example.

Wraps ``swyft.LogRatioEstimator_Ndim`` over the full 6-d cosmological
parameter joint. The prior is uniform on each parameter (see Table 1 in
the paper); bounds are encoded in ``cmblike.cmb.CMB``. Spectra are
binned and pre-processed (normalisation, log-transform, standardisation)
before being fed into the swyft network.
"""

import numpy as np
import swyft
import torch
from cmblike.cmb import CMB

from polyswyft.network import PolySwyftNetwork
from polyswyft.settings import PolySwyftSettings


class Network(PolySwyftNetwork):
    def __init__(self, polyswyftSettings: PolySwyftSettings, obs: swyft.Sample, cmbs: CMB):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.obs = obs
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr=self.polyswyftSettings.learning_rate_init),
                                                  torch.optim.lr_scheduler.ExponentialLR,
                                                  dict(gamma=self.polyswyftSettings.learning_rate_decay))
        self.network = swyft.LogRatioEstimator_Ndim(num_features=self.polyswyftSettings.num_features_dataset,
                                                    marginals=(
                                                        tuple(dim for dim in
                                                              range(self.polyswyftSettings.num_features)),),
                                                    varnames=self.polyswyftSettings.targetKey,
                                                    dropout=self.polyswyftSettings.dropout, hidden_features=128, Lmax=0,
                                                    num_blocks=2)
        self.cmbs = cmbs

    def forward(self, A, B):
        x_ = self.preprocess(A[self.polyswyftSettings.obsKey])
        return self.network(x_, B[self.polyswyftSettings.targetKey])

    def prior(self, cube) -> np.ndarray:
        theta = self.cmbs.prior(cube=cube)
        return theta

    def logRatio(self, theta: np.ndarray):
        theta = torch.tensor(theta)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        x_ = self.preprocess(self.obs[self.polyswyftSettings.obsKey])
        prediction = self.network(x_, theta)
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def get_new_network(self):
        return Network(polyswyftSettings=self.polyswyftSettings, obs=self.obs, cmbs=self.cmbs)

    def preprocess(self, x) -> swyft.Sample:
        x_ = x / self.obs[self.polyswyftSettings.obsKey]
        x_ = np.log10(x_)
        x_ = (x_ - self.polyswyftSettings.log_data_mean) / self.polyswyftSettings.log_data_std
        return x_
