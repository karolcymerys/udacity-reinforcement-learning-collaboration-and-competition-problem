from typing import Tuple, Union

import torch
from torch.nn import SmoothL1Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from maddpg.model import ActorNetwork, CriticNetwork
from maddpg.utils import OUNoise


class DDPGAgent:
    def __init__(self,
                 observation_size: int,
                 hidden_size: int,
                 action_size: int,
                 critic_action_size: int,
                 action_boundaries: Tuple[float, float],
                 gamma: float,
                 actor_lr: float = 1e-2,
                 critic_lr: float = 1e-2,
                 weight_decay: float = 1e-4,
                 tau: float = 1e-2,
                 device: str = 'cpu') -> None:
        self.actor = ActorNetwork(observation_size, action_size, hidden_size).to(device)
        self.critic = CriticNetwork(observation_size, critic_action_size, hidden_size).to(device)
        self.target_actor = ActorNetwork(observation_size, action_size, hidden_size).to(device)
        self.target_critic = CriticNetwork(observation_size, critic_action_size, hidden_size).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), critic_lr, weight_decay=weight_decay)

        self.tau = tau
        self.gamma = gamma

        self.action_bounds = action_boundaries

    def act(self,
            observation: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            noise_sampler: Union[OUNoise, None] = None) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        self.actor.eval()
        with torch.no_grad():
            response = self.actor(observation)
            if noise_sampler:
                response += noise_sampler.sample()

            return response.clip(*self.action_bounds)

    def act_target(self, observation: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                   ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        self.target_actor.eval()
        with torch.no_grad():
            return self.target_actor(observation).clip(*self.action_bounds)

    def train(self,
              observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              rewards: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              target_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              target_actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              dones: Union[torch.IntTensor, torch.cuda.IntTensor]) -> None:
        batch_size = observations.shape[0]
        self.actor.train()
        self.critic.train()
        critic_loss_fn = SmoothL1Loss()

        target_q_value = self.target_critic(target_observations, target_actions.view(batch_size, -1))
        expected_y = rewards + self.gamma * (1 - dones) * target_q_value

        y = self.critic(observations, actions.view(batch_size, -1))
        critic_loss = critic_loss_fn(y, expected_y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        response_action = self.actor(observations)
        agents_no = actions.shape[1]
        critic_input = torch.cat([response_action] + [actions[:, local_agent_id, :].detach() for local_agent_id in range(1, agents_no)], dim=1).view(batch_size, -1)

        actor_loss = -self.critic(observations, critic_input).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.__soft_update(self.tau)

    def hard_update(self) -> None:
        self.__soft_update(1.0)

    def __soft_update(self, tau: float) -> None:
        def transfer_weights(src_network: torch.nn.Module, desc_network: torch.nn.Module):
            for src_param, desc_param in zip(src_network.parameters(), desc_network.parameters()):
                desc_param.data.copy_(tau * src_param.data + (1.0 - tau) * desc_param.data)

        transfer_weights(self.actor, self.target_actor)
        transfer_weights(self.critic, self.target_critic)
