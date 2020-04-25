from ddpg import DDPGAgent
import torch
# import torch.nn as nn # Needed for grad clipping
import torch.nn.functional as F
import random
from utilities import soft_update, transpose_to_tensor
from buffer import ReplayBuffer

GRAD_CLIPPING = 1.0     # For gradient clipping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MADDPG_Agent(): 
    def __init__(self, state_size, action_size, num_agents=2, seed=1):
        super(MADDPG_Agent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.num_agents = num_agents             # Two agents for tennis environment
        self.maddpg_agent = [DDPGAgent(self.state_size, self.action_size, seed, index=i) for i in range(self.num_agents)]

        self.tstep = 0


    def get_actors(self):
        """Get actual actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """Get target actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, states_all_agents, noise=0.0):
        """Get actions from all agents in the MADDPG object.

        Params
        ======
            states_all_agents (array-like): the states for each agent
            noise: the noise to apply
        """

        actions = []
        for agent, state in zip(self.maddpg_agent, states_all_agents):
            actions.append(agent.act(state,noise).numpy())
        # actions = [agent.act(state, noise).numpy() for agent, state in zip(self.maddpg_agent, states_all_agents)]
        return actions

    def act_targets(self, states_all_agents, noise=0.0):
        """Get actions by target networks from all agents in the MADDPG object.
Params
        ======
            states_all_agents (array-like): the states for each agent
            noise: the noise to apply
        """

        actions = [agent.act_targets(state, noise) for agent, state in zip(self.maddpg_agent, states_all_agents)]
        return actions

    def step(self, state, action, reward, next_state, done):
        """ Save experience in replay memory, and learn new target weights

        Params
        =====
            state:      current state
            action:     taken action
            reward:     earned reward
            next_state: next state
            done:       Whether episode has finished
        """
        # Increase t_step
        self.tstep += 1

        # Step for each agent
        for agent in self.maddpg_agent:
            state_agent = state[agent.index,:]
            #action_agent = action[agent.index]
            reward_agent = reward[agent.index]
            next_state_agent = next_state[agent.index,:]
            done_agent = done[agent.index]

            #TODO: Maybe add states & actions for all agents? --> Adjust model, as well
            agent.step(state_agent, action, reward_agent, next_state_agent, done_agent)

