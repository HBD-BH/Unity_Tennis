import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# According to lesson
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

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
        #self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
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
            x = self.nonlin(linear(x))
    
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

        if len(hidden_layers)==1:
            hidden_layers = [hidden_layers, hidden_layers]

        # Add the first layer, input to a hidden layer
        # For the critic, this is num_agents * (states)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size *1, hidden_layers[0])])
        # self.hidden_layers.extend([nn.BatchNorm1d(hidden_layers[0])])
        
        # Add second hidden layer, with actions as additional inputs
        self.hidden_layers.extend([nn.Linear(hidden_layers[0] + 2*action_size, hidden_layers[1])])

        if len(hidden_layers)>2:
            layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.nonlin = F.selu
        
        self.output = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, state, action):
        ''' Forward pass through the network, returns the estimated value '''

        #print(f"Critic: state size {state.size()}")
        
        # State is input to first layer, convert everything to float
        x = self.nonlin(self.hidden_layers[0](state)).float()
        action = action.float()

        #print(f"x before cat: {x.size()}")
        x = torch.cat((x, action), dim=1)
        #print(f"x after cat: {x.size()}")
        x = self.nonlin(self.hidden_layers[1](x))


        # Forward through each layer in `hidden_layers`, with activation
        if len(self.hidden_layers)>2:
            for linear in self.hidden_layers[2:]:
                x = self.nonlin(linear(x))
    
        x = self.output(x)

        # Return the estimated value itself
        return x


