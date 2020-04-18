from ddpg import DDPGAgent
import torch
# import torch.nn as nn # Needed for grad clipping
import torch.nn.functional as F
import random
from utilities import soft_update, transpose_to_tensor
from buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 256        # Minibatch size
GAMMA = 0.99            # Discount factor
TAU   = 3e-3            # For soft update of target parameters

GRAD_CLIPPING = 1.0     # For gradient clipping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MADDPG_Agent(): 
    def __init__(self, state_size, action_size, seed=1, gamma=GAMMA, tau=TAU):
        super(MADDPG_Agent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.maddpg_agent = [DDPGAgent(self.state_size, self.action_size, seed),
                             DDPGAgent(self.state_size, self.action_size, seed)]

        self.gamma = gamma
        self.tau = tau

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
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            for a_i in range(len(self.maddpg_agent)):
                experiences = self.memory.sample()
                self.learn(experiences, a_i)
            self.target_soft_update()


    def learn(self, experiences, agent_number):
        """Update value parameters using given batch of experience tuples.
        This will happen for critic and all actor.

        According to the lessons:
            actor_target(state)             gives   action
            critic_target (state, action)   gives   Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            agent_number (int): index of agent to update
        """
        
        states, actions, rewards, next_states, dones = experiences

        agent = self.maddpg_agent[agent_number]

        # ------------------- update critic ------------------- #
        next_actions = agent.actor_target(next_states[agent_number::len(self.maddpg_agent),:])
        target_critic_input = torch.cat((next_states[agent_number::len(self.maddpg_agent),:], next_actions), dim=1).to(device)

        # Get Q targets (for next states) from target model (on CPU)
        with torch.no_grad():
            Q_targets_next = agent.critic_target(target_critic_input)
        # Compute Q targets for current states 
        Q_targets = rewards[:,agent_number].view(-1,1) + self.gamma * Q_targets_next * (1 - dones[:,agent_number].view(-1,1))

        # Get expected Q values from local model
        actions = actions.squeeze().float()#torch.cat(actions, dim=1)

        critic_input = torch.cat((states[agent_number::len(self.maddpg_agent),:], actions[agent_number::len(self.maddpg_agent),:]), dim=1).to(device)
        Q_expected = agent.critic_local(critic_input)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the critic loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if GRAD_CLIPPING > 0: 
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), GRAD_CLIPPING) # As mentioned on project page
        agent.critic_optimizer.step()

        # ------------------- update actor ------------------- #
        # Create input to agent's actor, detach other agents to save computation time
        actions_expected = agent.actor_local(states[agent_number::len(self.maddpg_agent),:])

        # Create input to agent's critic to get policy
        critic_input = torch.cat((states[agent_number::len(self.maddpg_agent)], actions_expected), dim=1)

        # Compute actor loss based on expectation from actions_expected
        actor_loss = -agent.critic_local(critic_input).mean()

        # Minimize the actor loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def target_soft_update(self):
        """Soft update model parameters for actor and critic of all MADDPG agents.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)

