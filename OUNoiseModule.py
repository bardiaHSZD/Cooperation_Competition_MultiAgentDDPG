import numpy as np
import torch
import random
import copy

OU_SIGMA = 0.2          # volatility
OU_THETA = 0.15         # speed of mean reversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class OUNoise:


    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state