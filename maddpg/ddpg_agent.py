from typing import Tuple, Union

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from maddpg.model import ActorNetwork, CriticNetwork
from maddpg.utils import OUNoise


class DDPGAgent:
    def __init__(self,
                 observation_size: int,
                 hidden_size: int,
                 action_size: int,
                 actor_size: int,
                 action_boundaries: Tuple[float, float],
                 gamma: float,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 tau: float = 1e-2,
                 device: str = 'cpu') -> None:
        self.actor = ActorNetwork(observation_size, action_size, hidden_size).to(device)
        self.critic = CriticNetwork(actor_size*observation_size, action_size, hidden_size, actor_size).to(device)
        self.target_actor = ActorNetwork(observation_size, action_size, hidden_size).to(device)
        self.target_critic = CriticNetwork(actor_size*observation_size, action_size, hidden_size, actor_size).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), critic_lr)

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
              global_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              local_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              other_agents_actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              rewards: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              global_target_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              local_target_observations: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              other_agents_target_actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
              dones: Union[torch.IntTensor, torch.cuda.IntTensor]) -> None:
        batch_size = local_observations.shape[0]
        self.actor.train()
        self.critic.train()
        critic_loss_fn = MSELoss()

        target_action = self.target_actor(local_target_observations)
        target_q_value = self.target_critic(
            global_target_observations,
            target_action,
            other_agents_target_actions.view(batch_size, -1))
        expected_y = rewards.view(batch_size, -1) + self.gamma * (1 - dones.view(batch_size, -1)) * target_q_value

        y = self.critic(global_observations, actions, other_agents_actions.view(batch_size, -1))
        critic_loss = critic_loss_fn(y, expected_y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        local_actions = self.actor(local_observations)
        actor_loss = -self.critic(global_observations, local_actions, other_agents_actions.view(batch_size, -1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def hard_update(self) -> None:
        self.__soft_update(1.0)

    def soft_update(self) -> None:
        self.__soft_update(self.tau)

    def __soft_update(self, tau: float) -> None:
        def transfer_weights(src_network: torch.nn.Module, desc_network: torch.nn.Module):
            for src_param, desc_param in zip(src_network.parameters(), desc_network.parameters()):
                desc_param.data.copy_(tau * src_param.data + (1.0 - tau) * desc_param.data)

        transfer_weights(self.actor, self.target_actor)
        transfer_weights(self.critic, self.target_critic)

    def save_weights(self, prefix: str, agent_id: int) -> None:
        torch.save(self.actor.state_dict(), f'{prefix}_actor_{agent_id}.pth')
        torch.save(self.critic.state_dict(), f'{prefix}_critic_{agent_id}.pth')

    def load_weights(self, prefix: str, agent_id: int) -> None:
        self.actor.load_state_dict(torch.load(f'{prefix}_actor_{agent_id}.pth'))
        self.critic.load_state_dict(torch.load(f'{prefix}_critic_{agent_id}.pth'))
