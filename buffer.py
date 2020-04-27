import random
from collections import namedtuple, deque
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    """Fixed-size buffer to store prioritized experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int):  dimension of each action
            buffer_size (int):  maximum size of buffer
            batch_size (int):   size of each training batch
            seed (int):         random seed
            alpha (0..1):       Toggle prioritization: 0 - no up to 1 - full 
        """

        self.action_size = action_size          # Not actually used, but might be helpful
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.epsilon = 1e-5       # To avoid numeric instabilities 
          
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(priority)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory.

        Params
        ======
            beta: Toggle how importance weights are used: 0 - no corrections to 1 - full correction
        
        """
        priorities = np.array(self.priorities).reshape(-1)            # Make row vector
        priorities = np.power(priorities + self.epsilon, self.alpha)  # Alpha toggles prioritization
        probs = priorities/np.sum(priorities)  # Compute a pdf over the priorities
        sampled_indices = np.random.choice(np.arange(len(probs)), size=self.batch_size, p=probs)  # Choose random indices given probabilities probs
        experiences = [self.memory[i] for i in sampled_indices]     # Get subset of the experiences
        probs = np.array([probs[i] for i in sampled_indices]).reshape(-1)  # Get probabilities of sampled indices

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Return weights (after correction using beta)
        weights = np.power(len(experiences) * probs, -beta)     # Correction
        weights /= weights.max()                                # Get values with abs in [0, 1]
        weights = torch.from_numpy(weights).float().to(device)  # Transform for device

        return (states, actions, rewards, next_states, dones, weights, sampled_indices)

    def update(self, indices, new_prios):
        """Update the priority values after training given the samples drawn.

        Params:
        ======
            indices (array-like):       The sampled indices 
            priorities (array-like):    The measure according to which to update the priorities (e.g., error between expected and target Q value)
        """
        for i, prio in zip(indices, new_prios):
            self.priorities[i] = prio

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
