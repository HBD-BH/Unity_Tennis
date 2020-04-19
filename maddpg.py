from ddpg import DDPGAgent
import torch
# import torch.nn as nn # Needed for grad clipping
import torch.nn.functional as F
import random
from utilities import soft_update, transpose_to_tensor
from buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 512        # Minibatch size
GAMMA = 0.99            # Discount factor
TAU   = 1e-3            # For soft update of target parameters

GRAD_CLIPPING = 1.0     # For gradient clipping
UPDATE_EVERY = 2
NUM_UPDATES = 4

NOISE = 1               # Scaling factor for noise
NOISE_DECAY = 1e-6     # Subtracting decay for noise


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MADDPG_Agent(): 
    def __init__(self, state_size, action_size, seed=1, gamma=GAMMA, tau=TAU):
        super(MADDPG_Agent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.num_agents = 2             # Two agents for tennis environment
        self.maddpg_agent = [DDPGAgent(self.state_size, self.action_size, seed, index=i) for i in range(self.num_agents)]

        self.gamma = gamma
        self.tau = tau
        self.num_update = NUM_UPDATES
        self.noise = NOISE
        self.noise_decay = NOISE_DECAY
        self.t_step = 0

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

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

        actions = [agent.act(state, noise) for agent, state in zip(self.maddpg_agent, states_all_agents)]
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
        ======
            state:      current state
            action:     taken action
            reward:     earned reward
            next_state: next state
            done:       Whether episode has finished
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY 

        # If UPDATE_EVErY and enough samples are available in memory, get random subset and learn
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for i in range(self.num_update):
                experiences = [self.memory.sample() for _ in range(self.num_agents)]
                self.learn(experiences)


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        This will happen for each actor's critic and actor.

        According to the lessons:
            actor_target(state)             gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """

        actions = []
        next_actions = []

        for agent_number, agent in enumerate(self.maddpg_agent):
            states, _, _, next_states, _ = experiences[agent_number]
            state = states[agent_number::self.num_agents,:]
            next_state = next_states[agent_number::self.num_agents,:]

            actions.append(agent.actor_local(state))
            next_actions.append(agent.actor_target(next_state))

        for agent_number, agent in enumerate(self.maddpg_agent):
            agent.learn(experiences[agent_number],actions, next_actions)

        # Decay noise over time
        self.noise = max(self.noise - self.noise_decay, 0)

