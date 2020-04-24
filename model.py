import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Actor(nn.Module):
    "Actor Network" 

    def __init__(self, state_size, action_size, seed, hidden_layers=[256,256]):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            state_size: integer, size of the input (e.g., state space)
            action_size: integer, size of the output layer (e.g., action space)
            seed (int): Random seed
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.nonlin = F.selu
        
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
    def forward(self, state):
        ''' Forward pass through the network, returns the action '''
        
        x = state
        # Forward through each layer in `hidden_layers`, with SELU activation
        for linear in self.hidden_layers:
            x = F.selu(linear(x))
    
        x = self.output(x)

        # Return an action probability
        return F.tanh(x)

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Critic(nn.Module):
    "Critic Network" 

    def __init__(self, state_size, action_size, seed, hidden_layers=[256,256]):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            state_size: integer, size of the input (e.g., state space)
            action_size: integer, size of the output layer (e.g., action space)
            seed (int): Random seed
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Add the first layer, input to a hidden layer
        # For the critic, this is num_agents * (states + actions)
        self.hidden_layers = nn.ModuleList([nn.Linear((state_size + action_size) * 1, hidden_layers[0])])
        self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.nonlin = F.selu
        
        self.output = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, state):
        ''' Forward pass through the network, returns the estimated value '''
        
        x = state
        # Forward through each layer in `hidden_layers`, with SELU activation
        for linear in self.hidden_layers:
            x = F.selu(linear(x))
    
        x = self.output(x)

        # Return the estimated value itself
        return x

