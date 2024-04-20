import copy
from typing import Tuple, Union

import torch


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3

    def __init__(self,
                 size: Tuple[int, int],
                 mu: float = 0.,
                 theta: float = 0.15,
                 sigma: float = 0.2,
                 device: str = 'cpu',
                 noise_factor_decay: float = 1.0
                 ) -> None:
        self.mu = mu * torch.ones(*size, device=device)
        self.theta = theta
        self.sigma = sigma
        self.state = mu * torch.ones(*size, device=device)
        self.device = device
        self.noise_factor = 1.0
        self.noise_factor_decay = noise_factor_decay

    def step(self) -> None:
        self.state = copy.copy(self.mu)
        self.noise_factor = max(0.01, self.noise_factor * self.noise_factor_decay)

    def sample(self) -> Union[torch.FloatType, torch.cuda.FloatTensor]:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(*self.mu.shape, device=self.device)
        self.state = x + dx
        return self.noise_factor * self.state
