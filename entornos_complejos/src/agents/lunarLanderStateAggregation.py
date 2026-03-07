import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
import random

class StateAggregationEnv(gym.ObservationWrapper):


    def __init__(self, env, bins, low, high):
        super().__init__(env)
        self.bins = bins
        self.buckets = [np.linspace(j, k, l - 1) for j, k, l in zip(low, high, bins)]
        self.observation_space = gym.spaces.MultiDiscrete(nvec=bins.tolist())


    def observation(self, obs):
        indices = tuple(np.digitize(i, b) for i, b in zip(obs, self.buckets))
        return indices