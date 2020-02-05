from networkforall import ActorNetwork, CriticNetwork
from torch.optim import Adam
import torch
import numpy as np
import random


from OUNoiseModule import OUNoise
from ReplayBufferModule import ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.eps = EPS_START
        self.num_agents = num_agents
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


