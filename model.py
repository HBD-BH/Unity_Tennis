import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Weight init according to lessons
def hidden_init(layer):
    """Helper function to initialize layers
    Returns [-1/sqrt(f) +1/sqrt(f)] where f is the fan_in of the layer.
    
    Params
    ======
        layer: Current layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Actor(nn.Module):
    "Actor Network" 

    def __init__(self, state_size, action_size, seed, hidden_layers=[512,256]):
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
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Add batch normalization for each layer
        self.norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_layers[i]) for i in range(len(hidden_layers))])

        # Nonlinear activiation function
        self.nonlin = F.selu
        
        # The actor outputs action_size values
        self.output = nn.Linear(hidden_layers[-1], action_size)

        # Initialize weigths 
        self.initialize_weights()

    def initialize_weights(self):
        """ Initialize model weights
        All hidden layers except the last one: 
        Initialize from uniform distribution [-1/sqrt(f) +1/sqrt(f)], where f is fan_in of the layer
        Final layer: 
        Initialize from [-3/10^3 +3/10^3]
        """
        for linear in self.hidden_layers[:-1]:
            linear.weight.data.uniform_(*hidden_init(linear))
        
        self.hidden_layers[-1].weight.data.uniform_(-3e-3,+3e-3)


        
    def forward(self, state):
        ''' Forward pass through the network, returns the action '''

        # Add singleton dimension for 1-D data
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = state
        # Forward through each layer in `hidden_layers`, with batch normalization and activation
        for i, linear in enumerate(self.hidden_layers):
            x = self.nonlin(self.norm_layers[i](linear(x)))

        x = self.output(x)

        # Return an action probability
        return F.tanh(x)

# Similar to Deep Q-Network lecture exercise and the PyTorch extracurricular Content
class Critic(nn.Module):
    "Critic Network" 

    def __init__(self, state_size, action_size, seed, hidden_layers=[512,256]):
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
        self.hidden_layers.extend([nn.Linear(hidden_layers[0] + 1*action_size, hidden_layers[1])])

        # In case we have additional hidden layers, add them
        if len(hidden_layers)>2:
            layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Add normalization for each layer
        self.norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_layers[i]) for i in range(len(hidden_layers))])
        
        self.drop = nn.Dropout(p=0.4)
        
        # Nonlinear activation function
        self.nonlin = F.selu
        
        # The critic only outputs a single (Q) value
        self.output = nn.Linear(hidden_layers[-1], 1)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """ Initialize model weights
        All hidden layers except the last one: 
        Initialize from uniform distribution [-1/sqrt(f) +1/sqrt(f)], where f is fan_in of the layer
        Final layer: 
        Initialize from [-3/10^3 +3/10^3]
        """
        for linear in self.hidden_layers[:-1]:
            linear.weight.data.uniform_(*hidden_init(linear))
        
        self.hidden_layers[-1].weight.data.uniform_(-3e-3,+3e-3)


        
    def forward(self, state, action):
        ''' Forward pass through the network, returns the estimated value '''

        # Add singleton dimension for 1-D input
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # State is input to first layer, convert everything to float
        x = self.nonlin(self.norm_layers[0](self.hidden_layers[0](state))).float()
        action = action.float()

        # Add action as input to second hidden layer
        x = torch.cat((x, action), dim=1)
        x = self.nonlin(self.norm_layers[1](self.hidden_layers[1](x)))

        # Add dropout layer to improve performance
        x = self.drop(x)

        # If there are additional hidden layers, 
        # Forward through each layer in `hidden_layers`, with normalization and activation
        if len(self.hidden_layers)>2:
            for i, linear in enumerate(self.hidden_layers[2:]):
                x = self.nonlin(self.norm_layers[i+2](linear(x)))
    
        x = self.output(x)

        # Return the estimated value itself
        return x
