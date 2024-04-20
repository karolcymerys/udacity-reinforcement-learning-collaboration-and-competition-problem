from copy import copy
from typing import Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from environment import TennisEnvironment
from maddpg.buffer import ReplayBuffer
from maddpg.ddpg_agent import DDPGAgent
from maddpg.utils import OUNoise


class MADDPG:
    def __init__(self,
                 env: TennisEnvironment,
                 gamma: float = 0.95,
                 buffer_size: int = 100_000,
                 hidden_size: int = 128,
                 action_boundaries: Tuple[float, float] = (-1, 1),
                 device: str = 'cpu') -> None:
        self.env = env
        self.observation_size = self.env.observation_size()
        self.action_size = self.env.action_size()
        self.agent_size = self.env.agents_size()
        self.agents = [
            DDPGAgent(
                self.observation_size,
                hidden_size,
                self.action_size,
                self.action_size * self.agent_size,
                action_boundaries,
                gamma,
                device=device
            )
            for _ in range(self.agent_size)
        ]
        self.replay_buffer = ReplayBuffer(buffer_size, device=device)
        self.device = device

    def train(self,
              max_episodes: int = 100_000,
              max_t: int = 100_000,
              minibatch_size: int = 1024) -> None:
        noise_sampler = OUNoise((1, self.action_size), device=self.device)

        scores = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        with (tqdm(range(1, max_episodes + 1)) as episodes):
            for episode_i in episodes:
                observations = self.env.reset().observations
                total_rewards = np.zeros(self.agent_size)
                for t in range(1, max_t + 1):
                    actions = torch.cat([agent.act(observations[agent_idx, :].view(1, -1), noise_sampler)
                                         for agent_idx, agent in enumerate(self.agents)])

                    results = self.env.step(actions)
                    rewards = results.rewards
                    target_observations = results.observations
                    dones = results.dones

                    self.replay_buffer.add(observations, actions, rewards, target_observations, dones)
                    total_rewards += rewards.detach().cpu().data.numpy()

                    if len(self.replay_buffer) >= minibatch_size:
                        self.__train_agents(minibatch_size)

                    if torch.any(dones == 1):
                        break

                    observations = target_observations

                scores.append(total_rewards)
                episodes.set_postfix({
                    'Current Avg reward': scores[-1],
                    'Avg reward': np.mean(scores[-100:], axis=0),
                })

                if np.max(np.mean(scores[-100:], axis=1)) >= 0.5:
                    print(f'Goal reached at {episode_i}th episode.')
                    break

                noise_sampler.step()
                np_scores = np.array(scores)
                for agent_id in range(self.agent_size):
                    plt.plot(np.arange(len(scores)), np_scores[:, agent_id], label=f'Agent {agent_id}')
                plt.pause(1e-5)

    def __train_agents(self, minibatch_size: int) -> None:

        def swap_rows(_actions_src: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                      agent_id: int) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
            # Critic networks assume that on first row there is action for its own agent
            _actions_target = copy(_actions_src)
            _actions_target[:, 0, :] = _actions_src[:, agent_id, :]
            actions_agent_idx = 0
            for idx in range(1, _actions_target.shape[1]):
                if actions_agent_idx == agent_id:
                    actions_agent_idx += 1

                _actions_target[:, idx, :] = _actions_src[:, actions_agent_idx, :]
                actions_agent_idx += 1
            return _actions_target

        for _ in range(10):
            for agent_idx, agent in enumerate(self.agents):
                observations, actions, rewards, target_observations, dones = self.replay_buffer.sample(minibatch_size)

                target_actions = torch.stack(
                    [self.agents[local_agent_id].act_target(target_observations[:, local_agent_id, :])
                     for local_agent_id in range(self.agent_size)], dim=1)

                agent.train(
                    observations[:, agent_idx, :],
                    swap_rows(actions, agent_idx),
                    rewards[:, agent_idx].view(minibatch_size, 1),
                    target_observations[:, agent_idx, :],
                    swap_rows(target_actions, agent_idx),
                    dones[:, agent_idx].view(minibatch_size, 1)
                )


    def test(self):
        pass
