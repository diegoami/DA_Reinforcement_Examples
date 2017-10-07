
# coding: utf-8

# The purpose of this notebook is to get you acquainted with: <br> 
#  - basic gym syntax, **env.reset()**, which resets the environment and **env.step(action)**, which applies the given action, <br>
#  - and basic terms **observation/state** and **reward**. <br>
#  
#  **observation (object)**: Observation gives information about the current state of the agent’s environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game. <br>The terms "observation" and "state" are often used interchangebly.<br><br>
# **reward (float)**: The reward is a feedback signal from the environment. It tells the agent if it’s doing a good job or not and agent’s primary objective to maximize the cumulative(total) reward.<br><br>
# **done (boolean)**: whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)<br><br>
# **info (dict)**: diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.<br>
# [OpenAI Gym Documentation](https://gym.openai.com/docs)

# In[54]:


import gym

## Create the environment
#env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0
env = gym.make('MountainCar-v0')
#env = gym.make('Taxi-v2')  # https://gym.openai.com/envs/Taxi-v1


# ![FrozenWorld](http://i.imgur.com/OJPULOI.png)

# In[8]:


from gym.spaces import Discrete 
from gym.spaces import Box
from gym import Space
## run in total episodes
for i_episode in range(1):
    ## restart and reset the game state.
    ## save the observation, observation == state
    state = env.reset()
    
    for t in range(1000):
        ## render the environment
        env.render()
        print(env.action_space)
        ## select a random action from available actions
        #action = env.action_space.sample()
        #if (t % 5 == 2):
        #    action = 0
        #else:
        #    action = 1
        if (t  < 50 ):
            action = 0
        else:
            action = 2
        print(action)
        
        ## apply the selected actions, and get next state
        next_state, reward, done, info = env.step(action)
        #print(state, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
        state = next_state


# In[22]:


get_ipython().magic('pinfo2 Space')


# In[64]:


import numpy as np
import itertools
def q_learning(env, num_episodes, alpha=0.85, discount_factor=0.99, Qarg = None):
    """
    Q learning algorithm, off-polics TD control. Finds optimal gready policies
    Args:
    env: Given environment to solve
    num_episodes: Number of episodes to learn
    alpha: learning rate
    discount factor: weight/importance given to future rewards
    epsilon: probability of taking random action. 
             We are using decaying epsilon, 
             i.e high randomness at beginning and low towards end
    Returns:
    Optimal Q
    """
     
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    # create a numpy array filled with zeros 
    # rows = number of observations & cols = possible actions
    Q = Qarg
    
    for i_episode in range(num_episodes):
            # reset the env
            state = env.reset()
            # itertools.count() has similar to 'while True:'
            for t in itertools.count():
                #Q_target = np.zero
                # generate a random num between 0 and 1 e.g. 0.35, 0.73 etc..
                # if the generated num is smaller than epsilon, we follow exploration policy 
                if np.random.random() <= epsilon:
                    # select a random action from set of all actions
                    action = env.action_space.sample()
                # if the generated num is greater than epsilon, we follow exploitation policy
                else:
                    # select an action with highest value for current state
                    ns = np.array(state)
                    position, velocity = ns
                    positionD = int(position * 10 + 12)
                    velocityD = int(velocity * 10 + 7)

                    action = np.argmax(Q[positionD, velocityD, :])
                
                # apply selected action, collect values for next_state and reward
                next_state, reward, done, _ = env.step(action)
                ns = np.array(next_state)
                position, velocity = ns
                positionD = int(position * 10 + 12)
                velocityD = int(velocity * 10 + 7)
                
                # Calculate the Q-learning target value
                
                Q_target = reward + discount_factor*np.max(Q[positionD,velocityD,:])
                # Calculate the difference/error between target and current Q
                Q_delta = Q_target - Q[positionD,velocityD,action]
                # Update the Q table, alpha is the learning rate
                Q[positionD,velocityD,action] = Q[positionD,velocityD,action] + (alpha * Q_delta)
                
                # break if done, i.e. if end of this episode
                if done:
                    break
                # make the next_state into current state as we go for next iteration
                state = next_state
            # gradualy decay the epsilon
            if epsilon > 0.1:
                epsilon -= 1.0/(num_episodes/10)
    
    return Q    # return optimal Q


# In[65]:


def test_algorithm(env, Q):
    """
    Test script for Q function
    Args:
    env: Given environment to test Q function
    Q: Q function to verified
    Returns:
    Total rewards for one episode
    """
    
    state = env.reset()
    total_reward = 0
    
    while True:
        ns = np.array(state)
        position, velocity = ns
        positionD = int(position * 10 + 12)
        velocityD = int(velocity * 10 + 7)
        
        # selection the action with highest values i.e. best action
        action = np.argmax(Q[positionD, velocityD, :])
        # apply selected action
        next_state, reward, done, _ = env.step(action)
        # render environment
        env.render()
        # calculate total reward
        total_reward += reward
        
        if done:
            print(total_reward)
            break
            
        state = next_state
    
    return total_reward 


# **Quick Question-** <br>
# What does the state value above, signify? e.g. in State: 2, what does 2 represent?

# In[57]:


import gym

## Create the environment
#env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0
env = gym.make('MountainCar-v0')
#env = gym.make('Taxi-v2')  # https://gym.openai.com/envs/Taxi-v1
Q = q_learning(env, 20000)


# In[58]:


test_algorithm(env, Q)


# In[81]:



Q = q_learning(env, 200000, Qarg=Q)


# In[62]:


Q[1,:,:]


# In[83]:


env = gym.make('MountainCar-v0')
test_algorithm(env, Q)


# In[69]:


Q


# In[80]:


with open('Q.out','w') as f:
    np.savetxt(f,Q,fmt='%.5e')


# In[84]:


velocityD = int(velocity * 10 + 7)


# In[117]:


import numpy as np
import itertools
import random
def q_learning(env, num_episodes, alpha=0.85, discount_factor=0.99, Qarg = None, explore=True):
    """
    Q learning algorithm, off-polics TD control. Finds optimal gready policies
    Args:
    env: Given environment to solve
    num_episodes: Number of episodes to learn
    alpha: learning rate
    discount factor: weight/importance given to future rewards
    epsilon: probability of taking random action. 
             We are using decaying epsilon, 
             i.e high randomness at beginning and low towards end
    Returns:
    Optimal Q
    """
     
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    # create a numpy array filled with zeros 
    # rows = number of observations & cols = possible actions
    Q = Qarg
    
    for i_episode in range(num_episodes):
            # reset the env
            state = env.reset()
            # itertools.count() has similar to 'while True:'
            for t in itertools.count():
                #Q_target = np.zero
                # generate a random num between 0 and 1 e.g. 0.35, 0.73 etc..
                # if the generated num is smaller than epsilon, we follow exploration policy 
                
                if explore or np.random.random() <= epsilon:
                    # select a random action from set of all actions
                    #action = env.action_space.sample()
                    action =0 if random.random() > 0.5 else 2
                # if the generated num is greater than epsilon, we follow exploitation policy
                else:
                    # select an action with highest value for current state
                    ns = np.array(state)
                    position, velocity = ns
                    positionD = int(position * 10 + 12)
                    velocityD = int(velocity * 10 + 7)

                    action = np.argmax(Q[positionD, velocityD, :])
                
                # apply selected action, collect values for next_state and reward
                next_state, reward, done, _ = env.step(action)
                ns = np.array(next_state)
                position, velocity = ns
                positionD = int(position * 10 + 12)
                velocityD = int(velocity * 10 + 7)
                
                # Calculate the Q-learning target value
                print(position, velocity)
                Q_target = reward + discount_factor*np.max(Q[positionD,velocityD,:])
                # Calculate the difference/error between target and current Q
                Q_delta = Q_target - Q[positionD,velocityD,action]
                # Update the Q table, alpha is the learning rate
                Q[positionD,velocityD,action] = Q[positionD,velocityD,action] + (alpha * Q_delta)
                
                # break if done, i.e. if end of this episode
                if done:
                    break
                # make the next_state into current state as we go for next iteration
                state = next_state
            # gradualy decay the epsilon
            if epsilon > 0.1:
                epsilon -= 1.0/(num_episodes/10)
    
    return Q    # return optimal Q


# In[124]:


Q[1:]


# In[122]:


env = gym.make('MountainCar-v0')
test_algorithm(env, Q)


# In[125]:



Q = q_learning(env, 10000, Q)


# In[126]:


Q


# In[ ]:




