# Udacity Project: Collaboration and competition

__TODO: Put here animation__

## Description

The goal of this project is to resolve __multi-agent collaborative__ environment.

### Environment

This environment is called _Tennis_ environment. 
It consists of 2 rackets (__agents__), whose goal is to keep ball in play as long as possible.
Environment is considered as resolved in training process when one of the agents gets 
__an average score of +0.5 over 100 consecutive episodes__.


### Observation space

The observation space of environment is built from 24-dimensional vector 
that represents the position and velocity of the ball and the rocket. 
Each agent receives its own local observation

### Actions

An agent is responsible for deciding what action should be taken to get the best score based on provided observation. 
Agent action space consist of 2-dimensional vector with continuous values between -1 and 1. 
They correspond to agent's movement toward (or away from) the net, and jumping.


### Reward

- a reward of +0.1 is provided for agent when it hits the ball over the net  
- a reward of -0.01 is provided for agent when it lets the ball hit the ground or hits the ball out of bounds

## Algorithm Description

All details related to algorithm utilized to resolve this problem can be found in [Report.md file](./Report.md).

## Structure description __TODO__

The project contains following files:

| Filename                       | Description                                                                    |
|--------------------------------|--------------------------------------------------------------------------------|
| `doc`                          | folder that contains docs related files                                        |
| `environment.py`               | wrapper class for _UnityEnvironment_ to simplify interactions with environment |
| `Report.md`                    | doc file that contains utilized algorithm's details                            |  
| `requirements.txt`             | file that contains all required Python dependencies                            |  
| `README.md`                    | doc file that contains project's description                                   | 
| `test.py`                      | Python script that allows to start trained agent                               |
| `train.py`                     | main Python script for training                                                |

## Try it yourself

### Dependencies

- [Python 3.X](https://www.python.org/downloads/)
- [git](https://git-scm.com/downloads)

First clone repository:

```shell
git clone https://github.com/karolcymerys/udacity-reinforcement-learning-collaboration-and-competition-problem.git
```

In order to install all required Python dependencies, please run below command:

```shell
pip install -r requirements.txt
```

Next, install custom _UnityAgents_ and its dependencies:

```shell
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install . 
```

Please note that it might be required to remove packages versions from
`deep-reinforcement-learning/python/requirements.txt` to successfully install these dependencies,
but `protobuf` must be used in version `3.20.0`.

At the end, download _Unity Environment_ for your platform:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Once ZIP package is downloaded, please unzip it and place files in archive to root folder.
Please note, that you may need to adjust `filename` parameter of `TenisEnvironment` class
(it depends on platform you use).

### Training

Feel free to train your own agents. In order to do that:

1. In `train.py` file adjust hyperparameters
2. Run `python train.py` to start training process
3. Once training is completed, then your network is saved in `ddpg_actor_model_weights.py` file __TODO__

### Testing

1. In order to run trained agent, run following command `python test.py`
   (it utilizes model weights from `ddpg_actor_model_weights.py` file)  __TODO__