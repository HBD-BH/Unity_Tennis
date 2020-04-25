import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from model import Actor, Critic
from utilities import hard_update, soft_update

from OUNoise import OUNoise
from buffer import ReplayBuffer, PrioritizedReplayBuffer

LR_ACTOR = 1e-4               # Learning rate for the actor's optimizer
LR_CRITIC = 1e-3               # Learning rate for the critic's optimizer
TAU = 3e-3                    # Tau factor for soft update
GAMMA = 0.99            # Discount factor

BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 256        # Minibatch size
USE_PER = True          # Use normal replay buffer or prioritized experience replay

WEIGHT_DECAY = 0#1e-5     # Weight decay for critic optimizer
UPDATE_EVERY = 10        # Update weights every {} time steps
N_UPDATES = 5           # Number of successive trainings
GRAD_CLIPPING = 1       # Gradient clipping

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

        if USE_PER:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # act and act_targets similar to exercises and MADDPG Lab
    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
    
        Params
        ======
            state (array_like): current state
            noise (float):    control whether or not noise is added
        """
        # Uncomment if state is numpy array instead of tensor
        state = torch.from_numpy(state).float().to(device)
        # Put model into evaluation mode
        action = np.zeros(self.action_size)
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        with torch.no_grad():
            probs = self.actor_local(state) + noise*self.noise.noise().to(device)

        # Put actor back into training mode
        self.actor_local.train()

        #  Transform probability into valid action ranges
        #act_min, act_max = self.action_limits
        #action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return np.clip(probs, -1, 1)

    # act and act_targets similar to exercises and MADDPG Lab
    def act_target(self, state, noise=0.0):
        """Returns actions for given state as per current target policy.
    
        Params
        ======
            state (array_like): current state
            noise (float):    control whether or not noise is added
        """
        # Uncomment if state is numpy array instead of tensor
        state = torch.from_numpy(state).float().to(device)
        # Put model into evaluation mode
        action = np.zeros(self.action_size)
        self.actor_target.eval()

        # Get actions for current state, transformed from probabilities
        with torch.no_grad():
            probs = self.actor_target(state) + noise*self.noise.noise().to(device)

        # Put actor back into training mode
        self.actor_target.train()

        #  Transform probability into valid action ranges
        #act_min, act_max = self.action_limits
        #action = (act_max - act_min) * (probs - 0.5) + (act_max + act_min)/2
        return np.clip(probs, -1, 1)

    def step(self, state, action, reward, next_state, done, beta=0):
        """Save experience in replay memory, use random samples from buffer to learn.
        
        PARAMS
        ======
            state:      current state
            action:     taken action
            reward:     earned reward
            next_state: next state
            done:       Whether episode has finished
            beta (float 0..1): PER: to what extend use importance weigths 
                                (0 - no corrections, 1 - full correction)
        """
        self.tstep += 1

        # Save experience in replay memory
        if USE_PER:
            # If we use PER, we use the error from prediction to actual Q-value as priorities
            next_actions = self.act_target(state).to(device)
            # Transfer everything to torch tensors
            actions_agent = torch.from_numpy(action[:,self.index*self.num_agents:self.index*self.num_agents+self.action_size]).float().to(device)
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            action = torch.from_numpy(action).float().to(device)

            next_actions = torch.cat((actions_agent, next_actions), dim=1).to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
            # Predicted Q value from Critic target network
            self.critic_target.eval()
            self.critic_local.eval()
            with torch.no_grad():
                Q_targets_next = self.critic_target(next_state, next_actions)

                Q_targets = reward + self.gamma * Q_targets_next * (1 - done)
                Q_expected = self.critic_local(state,action)
            self.critic_target.train()
            self.critic_local.train()
            
            # Error works as priority
            error = (Q_expected - Q_targets)**2
            self.memory.add(state, action, reward, next_state, done, error)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.tstep = (self.tstep + 1 ) % UPDATE_EVERY

        # If UPDATE_EVERY and enough samples are available in memory, get random subset and learn
        if self.tstep == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(self.num_updates):
                if USE_PER:
                    experiences = self.memory.sample(beta)
                else:
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

        if USE_PER:
            states, actions, rewards, next_states, dones, weights_cur, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)
        # Select correct set of actions from all actions
        # e.g. 2 agents, action_size=2:
        # actions: [-1 1 -1 1]
        # agent0:  [-1 1]
        # agent1:       [-1 1]
        actions_agent = actions[:,self.index*self.num_agents:self.index*self.num_agents+self.action_size].float()

        # ------------------- update critic ------------------- #
        next_actions = torch.cat((actions_agent, next_actions), dim=1).to(device)
        
        # Predicted Q value from Critic target network
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)

        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        Q_expected = self.critic_local(states,actions)
        
        if USE_PER:
            # Update priorities in PER
            loss = (Q_expected - Q_targets)**2
            loss = loss.reshape(weights_cur.shape) * weights_cur
            self.memory.update(indices, loss.data.cpu().numpy())

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        self.critic_optimizer.step()


        # ------------------- update actor ------------------- #
        actions_expected = self.actor_local(states)
        """
        if self.index == 0:
            actions_expected = torch.cat((actions_expected, actions_agent), dim=1)
        else: 
            actions_expected = torch.cat((actions_agent, actions_expected), dim=1)
        """
        actions_expected = torch.cat((actions_agent, actions_expected), dim=1)
        # Compute actor loss based on expectation from actions_expected
        actor_loss = -self.critic_local(states, actions_expected).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_soft_update()
        
    def save(self, filename):
        """Saves the agent to the local workplace

        Params
        ======
            filename (string): where to save the weights
        """

        checkpoint = {'input_size': self.state_size,
              'output_size': self.action_size,
              'actor_hidden_layers': [each.out_features for each in self.actor_local.hidden_layers if each._get_name()!='BatchNorm1d'],
              'actor_state_dict': self.actor_local.state_dict(),
              'critic_hidden_layers': [each.out_features for each in self.critic_local.hidden_layers if each._get_name()!='BatchNorm1d'],
              'critic_state_dict': self.critic_local.state_dict()}

        torch.save(checkpoint, filename)


    def load_weights(self, filename):
        """ Load weights to update agent's actor and critic networks.
        Expected is a format like the one produced by self.save()

        Params
        ======
            filename (string): where to load data from. 
        """
        checkpoint = torch.load(filename)
        if not checkpoint['input_size'] == self.state_size:
            print(f"Error when loading weights from checkpoint {filename}: input size {checkpoint['input_size']} doesn't match state size of agent {self.state_size}")
            return None
        if not checkpoint['output_size'] == self.action_size:
            print(f"Error when loading weights from checkpoint {filename}: output size {checkpoint['output_size']} doesn't match action space size of agent {self.action_size}")
            return None
        my_actor_hidden_layers = [each.out_features for each in self.actor_local.hidden_layers if each._get_name()!='BatchNorm1d']
        if not checkpoint['actor_hidden_layers'] == my_actor_hidden_layers:
            print(f"Error when loading weights from checkpoint {filename}: actor hidden layers {checkpoint['actor_hidden_layers']} don't match agent's actor hidden layers {my_actor_hidden_layers}")
            return None
        my_critic_hidden_layers = [each.out_features for each in self.critic_local.hidden_layers if each._get_name()!='BatchNorm1d']
        if not checkpoint['critic_hidden_layers'] == my_critic_hidden_layers:
            print(f"Error when loading weights from checkpoint {filename}: critic hidden layers {checkpoint['critic_hidden_layers']} don't match agent's critic hidden layers {my_critic_hidden_layers}")
            return None
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_state_dict'])


    def target_soft_update(self):
        """Soft update model parameters for actor and critic of all MADDPG agents.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        soft_update(self.actor_target, self.actor_local, self.tau)
        soft_update(self.critic_target, self.critic_local, self.tau)


