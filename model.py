import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Network(nn.Module):
    def __init__(self, input_size, output_size, seed, hidden_layers=[512,512], actor=False):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input (e.g., state space)
            output_size: integer, size of the output layer (e.g., action space)
            seed (int): Random seed
            hidden_layers: list of integers, the sizes of the hidden layers
            actor (bool): True if the network is used for an actor, false otherwise
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Reduce network complexity for actors
        if actor: 
            hidden_layers = [512,512]

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.nonlin = F.selu
        
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.actor = actor
        
    def forward(self, state):
        ''' Forward pass through the network, returns the action '''
        
        x = state
        # Forward through each layer in `hidden_layers`, with SELU activation
        for linear in self.hidden_layers:
            x = F.selu(linear(x))
    
        x = self.output(x)
    
        if self.actor:
            # Actor: return an action probability
            return F.tanh(x)

        else: 
            # Critic: return an expected reward
            return x

            
