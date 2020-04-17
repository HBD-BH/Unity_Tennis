import torch
import torch.optim as optim
import numpy as np
import random

from model import Network
from utilities import hard_update

from OUNoise import OUNoise

LR = 1e-3               # Learning rate for the optimizer
WEIGHT_DECAY = 1e-5     # Weight decay for critic optimizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_limits = [-1,1]     # Min, Max of all action values
        self.num_agents = 2             # Two agents for tennis environment

        # Critic input = state_size + size_actions*num_agents = 24+2*2=28
        critic_input = self.state_size + self.action_size #* self.num_agents
        critic_output = 1 # Critic output is just a number

        self.actor_local = Network(state_size, action_size, seed, actor=True).to(device)
        self.critic_local = Network(critic_input, critic_output, seed, actor=False).to(device)
        self.actor_target = Network(state_size, action_size, seed, actor=True).to(device)
        self.critic_target = Network(critic_input, critic_output, seed, actor=False).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        # Initialize target networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # act and act_targets similar to exercises and MADDPG Lab
    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
    
        Params
        ======
            state (array_like): current state
        """
        # Uncomment if state is numpy array instead of tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #state = state.to(device)
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.actor_local(state) + noise*self.noise.noise() 

        #  Transform probability into valid action ranges
        act_min, act_max = self.action_limits
        action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return action

    def act_targets(self, state, noise=0.0):
        """Returns actions for given state as per current target policy.
    
        Params
        ======
            state (array_like): current state
        """
        state = state.to(device)
        self.actor_target.eval()
        
        probs = self.actor_target(state) + noise*self.noise.noise() 

        #  Transform probability into valid action ranges
        act_min, act_max = self.action_limits
        action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return action
