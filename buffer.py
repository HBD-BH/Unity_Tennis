import random
from collections import namedtuple, deque
import torch
import numpy as np
from utilities import transpose_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPSILON = 1e-5         # PER: Ensuring numeric stability
ALPHA = 0              # PER: 1 = full prioritization, 0 = no prioritization

# Replay buffer as in previous projects and exercises.
# Adjusted to lists of lists as in MADDPG lab
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=ALPHA):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): 1 = full prioritization, 0 = no prioritization
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.priorities = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.epsilon = EPSILON

    def add(self, state, action, reward, next_state, done, prio):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(prio)
    
    def sample(self,beta):
        """Randomly sample a batch of experiences from memory.
        
        Params
        ======
            beta (float 0..1): To what extent use importance weights 
                                (0 - no corrections, 1 - full correction)
        """

        # Sample random indices according to priorities
        priorities = np.array(self.priorities).reshape(-1)
        priorities = np.power(priorities + self.epsilon, self.alpha) 
        probs = priorities/np.sum(priorities)
        sampled_indices = np.random.choice(np.arange(len(probs)), size=self.batch_size, p=probs)

        experiences = [self.memory[i] for i in sampled_indices] # Sample prioritized indices
        probs = np.array([probs[i] for i in sampled_indices]).reshape(-1)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        weights = np.power(self.batch_size * probs, -beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, sampled_indices)

    def update(self, indices, prios):
        """ Update priority values (after training)

        Params
        ======
            indices (array-like): Indices of experiences to update
            priorities (array-like): priorities to update to
        """
        for i, prio in zip(indices, prios):
            self.priorities[i] = prio

        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

