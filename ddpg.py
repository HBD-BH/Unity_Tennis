import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from model import Actor, Critic
from utilities import hard_update, soft_update

from OUNoise import OUNoise
from buffer import ReplayBuffer, PrioritizedReplayBuffer

LR_ACTOR = 1e-3               # Learning rate for the actor's optimizer
LR_CRITIC = 1e-3              # Learning rate for the critic's optimizer
TAU = 6e-2                    # Tau factor for soft update
GAMMA = 0.99                  # Discount factor
ALPHA = 0                     # PER: prioritization (0 = no, 1 = full)

BUFFER_SIZE = int(1e6)        # Replay buffer size
BATCH_SIZE = 128              # Minibatch size

WEIGHT_DECAY = 0#1e-5         # Weight decay for critic optimizer
UPDATE_EVERY = 1              # Update weights every {} time steps
N_UPDATES = 1                 # Number of successive trainings
GRAD_CLIPPING = 1             # Gradient clipping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPG_Agent:

    def __init__(self, state_size, action_size, seed, index=0, num_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):   Dimension of each state
            action_size (int):  Dimension of each action
            seed (int):         Random seed
            index (int):        Index assigned to the agent
            num_agents (int):   Number of agents in the environment
        """

        self.state_size = state_size           # State size
        self.action_size = action_size         # Action size
        self.seed = random.seed(seed)    # Random seed
        self.index = index                     # Index of this agent, not used at the moment
        self.tau = TAU                         # Parameter for soft weight update
        self.num_updates = N_UPDATES           # Number of updates to perform when updating 
        self.num_agents=num_agents             # Number of agents in the environment
        self.tstep = 0                         # Simulation step (modulo (%) UPDATE_EVERY)
        self.gamma = GAMMA                     # Gamma for the reward discount
        self.alpha = ALPHA                     # PER: toggle prioritization (0..1)

        # Set up actor and critic networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Ornstein-Uhlenbeck noise 
        self.noise = OUNoise((1, action_size), seed)

        # Replay buffer 
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.alpha)

    # act and act_targets similar to exercises and MADDPG Lab
    def act(self, states, noise=1.0):
        """Returns actions for given state as per current policy.
    
        Params
        ======
            state [n_agents, state_size]: current state
            noise (float):    control whether or not noise is added
        """
        # Uncomment if state is numpy array instead of tensor
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((1, self.action_size))
        
        # Put model into evaluation mode
        self.actor_local.eval()

        # Get actions for current state, transformed from probabilities
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()

        # Put actor back into training mode
        self.actor_local.train()

        # Ornstein-Uhlenbeck noise addition
        actions += noise*self.noise.sample() 
        
        #  Transform probability into valid action ranges
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, dones, beta):
        """Save experience in replay memory, use random samples from buffer to learn.
        
        PARAMS
        ======
            states:     [n_agents, state_size]  current state
            actions:    [n_agents, action_size] taken action
            rewards:    [n_agents]              earned reward
            next_states:[n_agents, state_size]  next state
            dones:      [n_agents]              Whether episode has finished
            beta:       [0..1]                  PER: toggles correction for importance weights (0 - no corrections, 1 - full correction)
        """
        # ------------------------------------------------------------------
        # Save experience in replay memory - slightly more effort due to Prioritization
        # We need to calculate priorities for the experience tuple. 
        # This is in our case (Q_expected - Q_target)**2
        # -----------------------------------------------------------------
        # Set all networks to evaluation mode
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()

        state = torch.from_numpy(states).float().to(device)
        next_state = torch.from_numpy(next_states).float().to(device)
        action = torch.from_numpy(actions).float().to(device)
        #reward = torch.from_numpy(rewards).float().to(device)
        #done = torch.from_numpy(dones).float().to(device)

        with torch.no_grad(): 
            next_actions = self.actor_target(state)

            own_action = action[:,self.index*self.action_size:(self.index+1)*self.action_size]
            if self.index:
                # Agent 1 
                next_actions_agent = torch.cat((own_action, next_actions), dim=1)
            else:
                # Agent 0: flipped order
                next_actions_agent = torch.cat((next_actions, own_action), dim=1)

            # Predicted Q value from Critic target network
            Q_targets_next = self.critic_target(next_state, next_actions_agent).float()
            #print(f"Type Q_t_n: {type(Q_targets_next)}")
            #print(f"Type gamma: {type(self.gamma)}")
            #print(f"Type dones: {type(dones)}")
            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
            Q_expected = self.critic_local(state,action)

        # Use error between Q_expected and Q_targets as priority in buffer
        error = (Q_expected - Q_targets)**2
        self.memory.add(state, action, rewards, next_state, dones, error)

        # Set all networks back to training mode
        self.actor_target.train()
        self.critic_target.train()
        self.critic_local.train()
        
        # ------------------------------------------------------------------
        # Usual learning procedure
        # -----------------------------------------------------------------
        # Learn every UPDATE_EVERY time steps
        self.tstep = (self.tstep + 1 ) % UPDATE_EVERY

        # If UPDATE_EVERY and enough samples are available in memory, get random subset and learn
        if self.tstep == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(self.num_updates):
                experiences = self.memory.sample(beta)
                self.learn(experiences)

    def reset(self):
        """Reset the noise parameter of the agent."""
        self.noise.reset()

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples. 
        Update according to 
            Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        
        According to the lessons: 
            actor_target  (state)           gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of 
                    states          states visited
                    actions         actions taken by all agents
                    rewards         rewards received
                    next states     all next states
                    dones           whether or not a final state is reached 
                    weights         weights of the experiences
                    indices         indices of the experiences            
        """ 

        # Load experiences from sample
        states, actions, rewards, next_states, dones, weights_cur, indices = experiences

        # ------------------- update critic ------------------- #
        
        # Get next actions via actor network
        next_actions = self.actor_target(next_states)
        
        # Stack action together with action of the agent
        own_actions = actions[:,self.index*self.action_size:(self.index+1)*self.action_size]
        if self.index:
            # Agent 1 
            next_actions_agent = torch.cat((own_actions, next_actions), dim=1)
        else:
            # Agent 0: flipped order
            next_actions_agent = torch.cat((next_actions, own_actions), dim=1)


        # Predicted Q value from Critic target network
        Q_targets_next = self.critic_target(next_states, next_actions_agent)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        Q_expected = self.critic_local(states,actions)

        # Update priorities in ReplayBuffer
        loss = (Q_expected - Q_targets).pow(2).reshape(weights_cur.shape) * weights_cur
        self.memory.update(indices, loss.data.cpu().numpy())
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        self.critic_optimizer.step()


        # ------------------- update actor ------------------- #
        actions_expected = self.actor_local(states)

        # Stack action together with action of the agent
        own_actions = actions[:,self.index*self.action_size:(self.index+1)*self.action_size]
        if self.index:
            # Agent 1:
            actions_expected_agent = torch.cat((own_actions, actions_expected), dim=1)
        else: 
            # Agent 0: flipped order
            actions_expected_agent = torch.cat((actions_expected, own_actions), dim=1)

        # Compute actor loss based on expectation from actions_expected
        actor_loss = -self.critic_local(states, actions_expected_agent).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.target_soft_update(self.critic_local, self.critic_target)
        self.target_soft_update(self.actor_local, self.actor_target)

    def target_soft_update(self, local_model, target_model):
        """Soft update model parameters for actor and critic of all MADDPG agents.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
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


