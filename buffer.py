#import numpy as np
import random
from collections import deque
import torch
from utilities import transpose_list

# Replay buffer as in previous projects and exercises.
# Adjusted to lists of lists as in MADDPG lab
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
    
    def add(self, experience)
        """Add a new experience to memory."""
        experiences = transpose_list(experience)
        for e in experiences:
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Transpose the list of lists
        return transpose_list(experiences)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
