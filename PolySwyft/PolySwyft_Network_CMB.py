from typing import Tuple, List, Any

import numpy as np
import swyft
import torch
from cmblike.cmb import CMB

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Network(swyft.SwyftModule):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, obs: swyft.Sample, cmbs: CMB):
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
        # self.summarizer = torch.nn.Sequential(torch.nn.Linear(self.polyswyftSettings.num_features_dataset, 32),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(32, 32),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(32, 16),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(16, self.polyswyftSettings.num_summary_features)
        #                                       )

    def forward(self, A, B):
        # s = self.summarizer(A[self.polyswyftSettings.obsKey])
        x_ = self.preprocess(A[self.polyswyftSettings.obsKey])
        return self.network(x_, B[self.polyswyftSettings.targetKey])

    def prior(self, cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        theta = self.cmbs.prior(cube=cube)
        return theta

    def logLikelihood(self, theta: np.ndarray) -> Tuple[Any, List]:
        """Computes the loglikelihood ("NRE") of the given theta."""
        theta = torch.tensor(theta)
        # check if list of datapoints or single datapoint
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        # s = self.summarizer(self.obs[self.polyswyftSettings.obsKey])
        x_ = self.preprocess(self.obs[self.polyswyftSettings.obsKey])
        prediction = self.network(x_, theta)
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))

    def get_new_network(self):
        return Network(polyswyftSettings=self.polyswyftSettings, obs=self.obs, cmbs=self.cmbs)

    def preprocess(self, x) -> swyft.Sample:
        x_ = x / self.obs[self.polyswyftSettings.obsKey]
        x_ = np.log10(x_)
        x_ = (x_ - self.polyswyftSettings.log_data_mean) / self.polyswyftSettings.log_data_std
        return x_
