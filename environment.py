from typing import Union

import torch
from unityagents import UnityEnvironment, BrainInfo


class ActionResult:
    def __init__(self,
                 states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                 rewards: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                 dones: Union[torch.IntTensor, torch.cuda.IntTensor]) -> None:
        self.states = states
        self.rewards = rewards
        self.dones = dones

    @staticmethod
    def from_brain_info(brain_info: BrainInfo, device: str):
        return ActionResult(
            torch.from_numpy(brain_info.vector_observations).float().to(device),
            torch.tensor(brain_info.rewards).float().to(device),
            torch.tensor(brain_info.local_done).int().to(device)
        )


class TennisEnvironment:
    def __init__(self,
                 filename: str = './Tennis.x86_64',
                 seed: int = 0,
                 device='cpu',
                 training_mode: bool = False) -> None:
        self.filename = filename
        self.seed = seed
        self.unity_env = UnityEnvironment(filename, seed=seed)
        self.brain_name = self.unity_env.brain_names[0]
        self.device = device
        self.training_mode = training_mode
        self.agents_no = None

    def action_size(self) -> int:
        return self.unity_env.brains[self.brain_name].vector_action_space_size

    def state_size(self) -> int:
        brain_details = self.unity_env.brains[self.brain_name]
        return brain_details.vector_observation_space_size * brain_details.num_stacked_vector_observations

    def agents_size(self) -> int:
        if not self.agents_no:
            self.agents_no = len(self.unity_env.reset(train_mode=self.training_mode)[self.brain_name].agents)
        return self.agents_no

    def reset(self) -> ActionResult:
        result = self.unity_env.reset(train_mode=self.training_mode)[self.brain_name]
        if not self.agents_no:
            self.agents_no = len(result.agents)
        return ActionResult.from_brain_info(result, self.device)

    def step(self, action: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> ActionResult:
        actions = action.detach().cpu().data.numpy()
        return ActionResult.from_brain_info(self.unity_env.step(actions)[self.brain_name], self.device)
