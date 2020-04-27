# Multiple Agents Control

### Introduction 

This code solves Udacity's version of Unity's [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. The goal is to train an agent to control a racket in order to pass a ball without it falling to the ground. 

It is a solution to the third project of Udacity's [Deep Reinforcement Learning](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Nanodegree.

### Setting

A reward of `+0.1` is given for each time an agent manages to hit the ball over the net. Whenever the ball falls to ground or is hit outside of the bounds of the game, a reward of `-0.01` is given. The goal of the agent is thus to play the ball over the net as quickly as possible.  

The observation space size is 24 (3 times 8), corresponding to position and velocity of the ball as well as the own racket. Each agent can only make local observations.  

Possible actions are jumping as well as moving towards (or away from) the net. Each of the 2 actions is continuous with a value in [-1,1].

The agent is to be trained by playing against itself. The environment is considered solved with an average maximum reward of +0.5 over 100 consecutive episodes (i.e., the maximum score per episode across both agents has to be at least 0.5 for 100 consecutive episodes). 

### Install dependencies

For correct installation, please make sure to use Python 3.6. 

In order to run the Jupyter Notebook `Tennis.ipynb`, please see the installation instructions [here](https://jupyter.readthedocs.io/en/latest/install.html).   

To run the notebook, you also have to download the `Tennis` environment from Udacity's [project page](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet). If your system is not Linux, please adjust the respective line of code in the beginning of `Tennis.ipynb` to point to your environment.

### Running the code

To run the code, please open `Tennis.ipynb` as a Jupyter notebook and follow the instructions given therein.

### Documentation

You find a report with a more detailed description of the implementation in the [doc](doc) subfolder.
