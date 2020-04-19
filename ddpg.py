import torch
import torch.optim as optim
import numpy as np
import random

from model import Actor, Critic
from utilities import hard_update, soft_update

from OUNoise import OUNoise

LR_ACTOR = 1e-3               # Learning rate for the actor's optimizer
LR_CRITIC = 1e-3               # Learning rate for the critic's optimizer
TAU = 1e-3                    # Tau factor for soft update
GAMMA = 0.99            # Discount factor

WEIGHT_DECAY = 0#1e-5     # Weight decay for critic optimizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:

    def __init__(self, state_size, action_size, seed, index=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            index (int): Index assigned to the agent
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_limits = [-1,1]     # Min, Max of all action values
        self.index = index
        self.tau = TAU

        self.actor_local = Actor(state_size, action_size, seed, actor=True).to(device)
        self.critic_local = Critic(state_size, action_size, seed, actor=False).to(device)
        self.actor_target = Actor(state_size, action_size, seed, actor=True).to(device)
        self.critic_target = Critic(state_size, action_size, seed, actor=False).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        # Initialize target networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    # act and act_targets similar to exercises and MADDPG Lab
    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
    
        Params
        ======
            state (array_like): current state
        """
        # Uncomment if state is numpy array instead of tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Put model into evaluation mode
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        with torch.no_grad():
            probs = self.actor_local(state) + noise*self.noise.noise() 

        # Put actor back into training mode
        self.actor_local.train()

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

    def learn(self, experiences, all_actions, all_next_actions, gamma=GAMMA):
        """Update value parameters using given batch of experience tuples. 
        Update according to 
            Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        
        According to the lessons: 
            actor_target  (state)           gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (0..1): discount factor
            all_actions (Tuple[torch.Variable]): all actions 
            all_next_actions (Tuple[torch.Variable]): all next actions 
        """ 

        states, actions, rewards, next_states, dones = experiences

        # ------------------- update critic ------------------- #
        next_actions = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(torch.cat(next_states, next_actions), dim=1)
        Q_targets = rewards[:,self.index] + gamma * Q_targets_next * (1 - dones[:, self.index])
        

        actions = actions.squeeze().float()
        Q_expected = self.critic_local(torch.cat((states,actions), dim=1))

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # ------------------- update actor ------------------- #
        # Create input to agent's actor, detach other agents to save computation time
        actions_expected = [actions if i==self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_expected = torch.cat(actions_expected, dim=1).to(device)

        # Compute actor loss based on expectation from actions_expected
        actor_loss = -self.critic_local(torch.cat((states, actions_expected), dim=1)).mean()
        self.actor_optimizer.zeros_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_soft_update()

    def target_soft_update(self):
        """Soft update model parameters for actor and critic of all MADDPG agents.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        soft_update(self.actor_target, self.actor_local, self.tau)
        soft_update(self.critic_target, self.critic_local, self.tau)

        
        


