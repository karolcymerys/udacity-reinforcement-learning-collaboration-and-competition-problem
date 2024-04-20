import torch

from maddpg.maddpg import MADDPG
from environment import TennisEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def maddpg(env: TennisEnvironment) -> None:
    algorithm = MADDPG(env, device=DEVICE)
    algorithm.train()


if __name__ == '__main__':
    env = TennisEnvironment(device=DEVICE, training_mode=True)
    maddpg(env)
