import numpy as np
import swyft
import torch

from polyswyft.network import PolySwyftNetwork
from polyswyft.settings import PolySwyftSettings


class Network(PolySwyftNetwork):
    def __init__(self, polyswyftSettings: PolySwyftSettings, obs: swyft.Sample):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.obs = obs
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr=self.polyswyftSettings.learning_rate_init),
                                                  torch.optim.lr_scheduler.ExponentialLR,
                                                  dict(gamma=self.polyswyftSettings.learning_rate_decay))
        self.network = swyft.LogRatioEstimator_Ndim(num_features=self.polyswyftSettings.num_features_dataset, marginals=(
            tuple(dim for dim in range(self.polyswyftSettings.num_features)),),
                                                    varnames=self.polyswyftSettings.targetKey,
                                                    dropout=self.polyswyftSettings.dropout, hidden_features=128, Lmax=0)

    def forward(self, A, B):
        return self.network(A[self.polyswyftSettings.obsKey], B[self.polyswyftSettings.targetKey])

    def prior(self, cube) -> np.ndarray:
        theta = self.polyswyftSettings.model.prior().bijector(x=cube)
        return theta

    def logRatio(self, theta: np.ndarray):
        theta = torch.tensor(theta)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        prediction = self.network(self.obs[self.polyswyftSettings.obsKey], theta)
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def get_new_network(self):
        return Network(polyswyftSettings=self.polyswyftSettings, obs=self.obs)
