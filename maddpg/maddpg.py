from typing import Tuple

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
                 gamma: float = 0.99,
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
                self.agent_size,
                action_boundaries,
                gamma,
                device=device
            )
            for _ in range(self.agent_size)
        ]
        self.replay_buffer = ReplayBuffer(buffer_size, device=device)
        self.device = device

    def train(self,
              max_episodes: int = 5_000,
              max_t: int = 5_000,
              minibatch_size: int = 256,
              optimize_every_timestamps: int = 1,
              optimization_loops: int = 2) -> None:
        scores = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        noise_sampler = OUNoise((1, self.action_size), device=self.device, noise_factor_decay=0.999)
        for agent in self.agents:
            agent.hard_update()

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

                    if len(self.replay_buffer) >= minibatch_size and t % optimize_every_timestamps == 0:
                        for _ in range(optimization_loops):
                            self.__train_agents(minibatch_size)
                        self.__soft_update()

                    if torch.any(dones == 1):
                        break

                    observations = target_observations

                scores.append(total_rewards)
                np_scores = np.array(scores)
                episodes.set_postfix({
                    'Current reward': np_scores[-1, :],
                    'Avg reward over 100 last episodes': np.mean(np_scores[-100:, :], axis=0),
                    'Max of Avg reward over 100 last episodes': np.max(np.mean(np_scores[-100:, :], axis=0))
                })

                plt.plot(np.arange(len(scores)), np.max(np_scores[:, :], axis=1))
                plt.pause(1e-5)

                if np.max(np.mean(np_scores[-100:, :], axis=0)) >= 0.5:
                    print(f'Goal reached at {episode_i}th episode.')
                    break

                noise_sampler.step()

    def __train_agents(self, minibatch_size: int) -> None:
        for agent_idx, agent in enumerate(self.agents):
            observations, actions, rewards, target_observations, dones = self.replay_buffer.sample(minibatch_size)
            batch_size = observations.shape[0]

            other_agents_actions = torch.stack([actions[:, local_agent_id, :]
                                                for local_agent_id in range(self.agent_size) if
                                                local_agent_id != agent_idx], dim=1)

            other_agents_target_actions = torch.stack(
                [self.agents[local_agent_id].act_target(target_observations[:, local_agent_id, :])
                 for local_agent_id in range(self.agent_size) if local_agent_id != agent_idx], dim=1)

            agent.train(
                observations.view(batch_size, -1),
                observations[:, agent_idx, :],
                actions[:, agent_idx, :],
                other_agents_actions,
                rewards[:, agent_idx],
                target_observations.view(batch_size, -1),
                target_observations[:, agent_idx, :],
                other_agents_target_actions,
                dones[:, agent_idx]
            )

    def __soft_update(self):
        for agent in self.agents:
            agent.soft_update()

    def save_weights(self):
        for agent_idx, agent in enumerate(self.agents):
            agent.save_weights('maddpg', agent_idx)

    def test(self):
        pass
