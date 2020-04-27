import numpy as np
from ddpg import DDPG_Agent
import torch
# import torch.nn as nn # Needed for grad clipping
import torch.nn.functional as F
import random

GRAD_CLIPPING = 1.0     # For gradient clipping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MADDPG_Agent(): 
    def __init__(self, state_size, action_size, num_agents=2, seed=0):
        super(MADDPG_Agent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.num_agents = num_agents             # Two agents for tennis environment
        self.ddpg_agents = [DDPG_Agent(self.state_size, self.action_size, seed, index=i) for i in range(self.num_agents)]

        self.tstep = 0


    #def get_actors(self):
        #"""Get actual actors of all the agents in the MADDPG object"""
        #actors = [ddpg_agent.actor_local for ddpg_agent in self.ddpg_agents]
        #return actors

    #def get_target_actors(self):
        #"""Get target actors of all the agents in the MADDPG object"""
        #target_actors = [ddpg_agent.actor_target for ddpg_agent in self.ddpg_agents]
        #return target_actors

    def act(self, states_all_agents, noise=0.0):
        """Get actions from all agents in the MADDPG object.

        Params
        ======
            states_all_agents (array-like): the states for each agent
            noise: the noise to apply
        """

        # Transform states and next_states into row vectors
        states_all_agents = np.reshape(states_all_agents, (1,self.num_agents*self.state_size))

        # Calculation of actions
        actions = self.ddpg_agents[0].act(states_all_agents, noise)
        for agent in self.ddpg_agents[1:]:
            action = agent.act(states_all_agents,noise)
            actions = np.concatenate((actions, action), axis=0)
        actions = np.reshape(actions, (1, self.num_agents*self.action_size))

        #actions = [agent.act(states_all_agents, noise) for agent in self.ddpg_agents]
        # Transform from list to (1,numagents*action_size) ndarray
        #actions = np.resize(np.asarray(actions, dtype=np.float32), (1,self.num_agents*self.action_size))
        return actions

    '''
    def act_targets(self, states_all_agents, noise=0.0):
        """Get actions by target networks from all agents in the MADDPG object.
Params
        ======
            states_all_agents (array-like): the states for each agent
            noise: the noise to apply
        """

        actions = [agent.act_targets(state, noise) for agent, state in zip(self.ddpg_agents, states_all_agents)]
        return actions
    '''

    def step(self, states, actions, rewards, next_states, done,beta=0):
        """ Save experience in replay memory, and learn new target weights

        Params
        =====
            states:      current states of all agents
            actions:     taken actions of all agents
            rewards:     earned rewards of all agents
            next_states: next states of all agents
            done:       Whether episode has finished
            beta (float 0..1): PER: to what extend use importance weigths 
                                (0 - no corrections, 1 - full correction)
        """
        # Increase t_step
        self.tstep += 1

        # Transform states and next_states into row vectors
        states = np.reshape(states, (1,self.num_agents*self.state_size))
        next_states = np.reshape(next_states, (1,self.num_agents*self.state_size))
        
        # Step for each agent
        for agent in self.ddpg_agents:
            agent.step(states,actions,rewards[agent.index],next_states,done,beta)

    def reset(self):
        """Reset noise parameters of each agent"""

        for agent in self.ddpg_agents:
            agent.reset()
            
    def save(self, filename_root):
        """Saves the agent to the local workplace, one DDPG agent at a time

        Params
        ======
            filename_root (string): where to save the weights. Root name, to which 'agentX.pth' is appended. 
        """
        for i, agent in enumerate(self.ddpg_agents):
            filename_cur = f"{filename_root}_agent{i}.pth"
            agent.save(filename_cur)
    
    def load_weights(self, filename_root):
        """ Load weights to update agent's actor and critic networks.
        Expected is a format like the one produced by self.save()

        Params
        ======
            filename_root (string): where to load data from. Root name, to which 'agentX' is appended for each agent.
        """
        for i, agent in enumerate(self.ddpg_agents):
            filename_cur = f"{filename_root}_agent{i}.pth"
            agent.load_weights(filename_cur)

