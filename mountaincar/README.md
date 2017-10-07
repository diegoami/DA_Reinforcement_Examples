## Synopsis
This is a Deep Reinforcement Learning solution to some classic control problems. I've used it to solve [MountainCar-v0 problem](https://gym.openai.com/envs/MountainCar-v0), [CartPole-v0](https://gym.openai.com/envs/CartPole-v0) and [CartPole-v1] (https://gym.openai.com/envs/CartPole-v1) in OpenAI's [Gym](https://gym.openai.com/docs).
This code uses [Tensorflow](https://www.tensorflow.org/) to model a value function for a Reinforcement Learning agent.
I've run it with Tensorflow 1.0 on Python 3.5 under Windows 7.

Some of the hyperparameters used in the main.py script to solve MountainCar-v0 have been optained partly through exhaustive search, and partly via Bayesian optimization with [Scikit-Optimize](https://scikit-optimize.github.io/). The optimized hyperparameters and their values are:
* Size of 1st fully connected layer: 198
* Size of 2nd fully connected layer: 96
* Learning rate: 2.33E-4
* Period (in steps) for the update of the target network parameters as per the DQN algorithm: 999
* Discount factor: 0.99
* Whether to use Double DQN: False

## References
1. [Deep Learning tutorial](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf), David Silver, Google DeepMind.
2. [My code on Github](https://github.com/avalcarce/openai_playground)