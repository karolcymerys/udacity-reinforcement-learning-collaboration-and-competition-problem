import math
from typing import Union

import torch
from torch import nn


class ActorNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int, seed: int = 0) -> None:
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=observation_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_size),
            nn.Tanh()
        )

        self.mlp[0].weight.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        self.mlp[3].weight.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        self.mlp[5].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, observations: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        return self.mlp(observations)


class CriticNetwork(nn.Module):

    def __init__(self, observation_size: int, action_size: int, hidden_size: int, seed: int = 0) -> None:
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.v1 = nn.Sequential(
            nn.Linear(in_features=observation_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.v2 = nn.Sequential(
            nn.Linear(in_features=hidden_size + action_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )

        self.v1[0].weight.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        self.v2[0].weight.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        self.v2[2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        x = self.v1(observations)
        return self.v2(torch.cat([x, actions], dim=1))
