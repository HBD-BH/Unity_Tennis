import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from model import Actor, Critic
from utilities import hard_update, soft_update

from OUNoise import OUNoise
from buffer import ReplayBuffer

LR_ACTOR = 1e-3               # Learning rate for the actor's optimizer
LR_CRITIC = 1e-3               # Learning rate for the critic's optimizer
TAU = 6e-2                    # Tau factor for soft update
GAMMA = 0.99            # Discount factor

BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 128        # Minibatch size

WEIGHT_DECAY = 0#1e-5     # Weight decay for critic optimizer
UPDATE_EVERY = 1        # Update weights every {} time steps
N_UPDATES = 1           # Number of successive trainings

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
        self.index = index    # Index of this agent
        self.tau = TAU
        self.num_updates = N_UPDATES
        self.num_agents=2     # Number of agents in the environment
        self.tstep = 0          # Simulation step (module UPDATE_EVERY)
        self.gamma = GAMMA 

        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)

        self.noise = OUNoise(action_size, seed)

        # Initialize target networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # act and act_targets similar to exercises and MADDPG Lab
    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
    
        Params
        ======
            state (array_like): current state
            noise (boolean):    control whether or not noise is added
        """
        # Uncomment if state is numpy array instead of tensor
        state = torch.from_numpy(state).float().to(device)
        # Put model into evaluation mode
        action = np.zeros(self.action_size)
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        with torch.no_grad():
            probs = self.actor_local(state) + noise*self.noise.noise() 

        # Put actor back into training mode
        self.actor_local.train()

        #  Transform probability into valid action ranges
        #act_min, act_max = self.action_limits
        #action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return np.clip(probs, -1, 1)


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
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, use random samples from buffer to learn.
        
        PARAMS
        ======
            state:      current state
            action:     taken action
            reward:     earned reward
            next_state: next state
            done:       Whether episode has finished
            TODO: add beta for prioritized experience replay
        """
        self.tstep += 1

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.tstep = (self.tstep + 1 ) % UPDATE_EVERY

        # If UPDATE_EVErY and enough samples are available in memory, get random subset and learn
        if self.tstep == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(self.num_updates):
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples. 
        Update according to 
            Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        
        According to the lessons: 
            actor_target  (state)           gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            all_actions (Tuple[torch.Variable]): all actions 
            all_next_actions (Tuple[torch.Variable]): all next actions 
        """ 

        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)

        # ------------------- update critic ------------------- #
        print(f"Action in experiences: {actions.size()}")
        if self.index == 0:
            next_actions = torch.cat((next_actions, actions[:,:2].float()), dim=1).to(device)
        else:
            next_actions = torch.cat((actions[:,2:], next_actions), dim=1).to(device)

        # Predicted Q value from Critic target network
        print(f"Using next actions size {next_actions.size()}")
        print(f"Using next states  size {next_states.size()}")
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)

        print(f"Using Q_tar_n size {Q_targets_next.size()}")
        print(f"Using rewards size {rewards.size()}")
        print(f"Using dones size {dones.size()}")
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        print(f"Using Q_tar size {Q_targets.size()}")
        

        #actions = actions.squeeze().float()
        print(f"Using actions size {actions.size()}")
        print(f"Using states size {states.size()}")
        Q_expected = self.critic_local(states,actions)
        print(f"Using Q_exp size {Q_expected.size()}")


        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # ------------------- update actor ------------------- #
        actions_expected = self.actor_local(states)

        if self.index == 0:
            actions_expected = torch.cat((actions_expected, actions[:,2:]), dim=1)
        else: 
            actions_expected = torch.cat((actions[:,:2], actions_predicted), dim=1)

        # Compute actor loss based on expectation from actions_expected
        actor_loss = -self.critic_local(states, actions_expected).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_soft_update()

    def target_soft_update(self):
        """Soft update model parameters for actor and critic of all MADDPG agents.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        soft_update(self.actor_target, self.actor_local, self.tau)
        soft_update(self.critic_target, self.critic_local, self.tau)

