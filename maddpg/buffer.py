import random
from collections import deque
from typing import Union, Tuple

import torch


class ReplayBuffer:

    def __init__(self,
                 buffer_size: int,
                 device: str,
                 seed: int = 0) -> None:
        self.device = device
        random.seed(seed)
        self.buffer_size = buffer_size

        self.observations = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.target_observations = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.choices = set()

    def add(self,
            observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            rewards: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            target_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            dones: Union[torch.IntTensor, torch.cuda.IntTensor]) -> None:
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.target_observations.append(target_observations)
        self.dones.append(dones)
        self.choices.add((self.__len__() - 1) % self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.IntTensor, torch.cuda.IntTensor]
    ]:
        selected_indices = random.sample(self.choices,
                                         self.__len__() if self.__len__() < batch_size else batch_size)

        return (
            torch.stack([self.observations[index] for index in selected_indices]).float().to(self.device),
            torch.stack([self.actions[index] for index in selected_indices]).float().to(self.device),
            torch.stack([self.rewards[index] for index in selected_indices]).float().to(self.device),
            torch.stack([self.target_observations[index] for index in selected_indices]).float().to(self.device),
            torch.stack([self.dones[index] for index in selected_indices]).int().to(self.device)
        )

    def __len__(self) -> int:
        return len(self.observations)
