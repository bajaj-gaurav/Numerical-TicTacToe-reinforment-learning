#!/usr/bin/env python
# coding: utf-8

# ## Tic-Tac-Toe Agent
# 
# In this notebook, we will learn to build an RL agent (using Q-learning) that learns to play Numerical Tic-Tac-Toe with odd numbers. The environment is playing randomly with the agent, i.e. its strategy is to put an even number randomly in an empty cell. The following is the layout of the notebook:
#         - Defining epsilon-greedy strategy
#         - Tracking state-action pairs for convergence
#         - Define hyperparameters for the Q-learning algorithm
#         - Generating episode and applying Q-update equation
#         - Checking convergence in Q-values

# #### Importing libraries
# Code to import Tic-Tac-Toe class from the environment file

# In[42]:


from TCGame_Env import TicTacToe
import collections
import numpy as np
import random
import pickle
import time
import copy 
from matplotlib import pyplot as plt


# In[43]:


# Function to convert state array into a string to store it as keys in the dictionary
# states in Q-dictionary will be of form: x-4-5-3-8-x-x-x-x
#   x | 4 | 5
#   ----------
#   3 | 8 | x
#   ----------
#   x | x | x

def Q_state(state):

    return ('-'.join(str(e) for e in state)).replace('nan','x')


# In[44]:


# Function which will return valid (all possible actions) actions corresponding to a state
# Important to avoid errors during deployment.
def valid_actions(state):
    valid_Actions = []
    valid_Actions = [i for i in env.action_space(state)[0]]
    return valid_Actions


# In[45]:


# Function which will add new Q-values to the Q-dictionary. 
def add_to_dict(state, valid_acts):
    state1 = Q_state(state)
    if state1 not in Q_dict.keys():
        Q_dict[state1] = {}
        for action in valid_acts:
            Q_dict[state1][action]=0


# #### Epsilon-greedy strategy

# ## Many values were considered to find out the optimum max_epsilon, min_epsilon and x in the following function. 
# #### The aim was to find the values which could give high epsilon initially for more exploration

# In[46]:


# Defining epsilon-greedy policy.
# After many values of x, we have come to the conclusion to select x as -0.0000008
#Please look at the graph at last of the notebook
max_epsilon = 1.0
min_epsilon = 0.001
x = -0.0000008
def epsilon_greedy(state, time):
    #epsilon = - 1/ (1 + np.exp((-time+7500000)/1700000)) + 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(x * time)
    z = np.random.random()
    q_state = Q_state(state)
    
    if z > epsilon:
        action = max(Q_dict[q_state],key=Q_dict[q_state].get)   #Exploitation: this gets the action corresponding to max q-value of current state
    else:
        possible_actions = valid_actions(state)
        action = random.choice(possible_actions)   #Exploration: randomly choosing an action  
    return action


# #### Tracking the state-action pairs for checking convergence

# In[47]:


# Initialise Q_dictionary as 'Q_dict' and States_tracked as 'States_track' (for convergence)
Q_dict = collections.defaultdict(dict)
States_track = collections.defaultdict(dict)

print(len(Q_dict))
print(len(States_track))


# In[48]:


# Initialise states to be tracked for convergence
def initialise_tracking_states():
    sample_q_values = [('1-6-5-8-x-3-4-7-2', (4, 9)), ('9-x-x-1-4-6-x-x-x',(1, 7)), ('x-x-x-8-x-2-1-7-x',(2, 3)),('x-x-x-8-x-x-1-x-x',(7, 7))]    #select any 4 Q-values
    for q_values in sample_q_values:
        state = q_values[0]
        action = q_values[1]
        States_track[state][action] = []


# In[49]:


# Function to save the Q-dictionary as a pickle file

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[50]:


# Function to save values to the tracking states
def save_tracking_states():
    for state in States_track.keys():
        for action in States_track[state].keys():
            if state in Q_dict and action in Q_dict[state]:
                States_track[state][action].append(Q_dict[state][action])


# In[51]:


initialise_tracking_states()


# #### Define hyperparameters

# In[52]:


EPISODES = 10000000
#EPISODES = 10
LR = 0.02                 #learning rate
GAMMA = 0.9
threshold = 2000          # every these many episodes, the 4 Q-values will be stored/appended (convergence graphs)
#threshold = 1
policy_threshold = 3000    #every these many episodes, the Q-dict will be updated


# ### Q-update loop

# In[53]:


# Main code

# Tracking the total time
start_time = time.time()

# Running for EPISODES number of times
for episode in range(EPISODES):
    env = TicTacToe()
    
    # initial state which is initialized in the environment file
    initial_state = copy.deepcopy(env.state)
    curr_state = copy.deepcopy(env.state)
    
    # add initial state to Q_dict
    add_to_dict(curr_state,valid_actions(curr_state))
    
    # initial reward
    reward = None
    
    # finding all possible actions for the current state
    # possible_actions = valid_actions(curr_state)
    
    # randomly choosing an action for the agent
    # curr_action = random.choice(list(possible_actions))
    
    # calculating the next state after the first agent and enviroment move
    # next_state = env.initial_step(curr_state, curr_action)
    
    # add the next state to the dictionary
    # add_to_dict(next_state,valid_actions(next_state))
    
    #curr_state = copy.deepcopy(next_state)
    
    total_reward = 0
    terminated = False
    
    # loop on termination condition of Win(Wins), Loss(Wins) or Tie(Tie)
    while not terminated:
        
        # calculating the next action on the basis of the epsilon policy
        curr_action = epsilon_greedy(curr_state, episode)
        
        # finding the next_state and reward
        next_state, reward, terminated = env.step(copy.deepcopy(curr_state), curr_action)
        # print(next_state)
        
        # add the next state to the dictionary
        add_to_dict(next_state,valid_actions(next_state))
        
        # converting the list into "-" separated string and replacing nan by x
        next_q_state = Q_state(next_state)
        curr_q_state = Q_state(curr_state)
        
        # UPDATE RULE
        if not terminated:
            max_next = max(Q_dict[next_q_state],key=Q_dict[next_q_state].get)
            dis = GAMMA*(Q_dict[next_q_state][max_next])
        else:
            # if the next state is a terminal state, then the Q-values from that state are 0. (No action is possible from that state)
            dis = 0
        
        Q_dict[curr_q_state][curr_action] += LR * ((reward + dis) - Q_dict[curr_q_state][curr_action])
        
        curr_state = copy.deepcopy(next_state)
        total_reward += reward

    # print(total_reward)
    # print(terminated)
        
    #TRACKING Q-VALUES
    if (episode == threshold-1):        #at the 1999th episode
        initialise_tracking_states()
        
    #TRACKING Q-VALUES
    # save the tracked state action pair Q values in intervals   
    if ((episode+1) % threshold) == 0:
        #print('saving tracking')
        save_tracking_states()
        save_obj(States_track,'States_tracked')        
    
    # Commenting out the below code inorder to print the final value
    #SAVING POLICY
    #if ((episode+1)% policy_threshold ) == 0:  #every 3000th episodes, the Q-dict will be saved
    #   save_obj(Q_dict,'Policy')   
    
elapsed_time = time.time() - start_time
print('Total time taken ',elapsed_time)
save_obj(States_track,'States_tracked')   
save_obj(Q_dict,'Policy')


# In[ ]:





# #### Check the Q-dictionary

# In[ ]:


Q_dict


# In[55]:


len(Q_dict)


# ### try checking for one of the states

# # We choose a random state and lets observe a q-value

# In[56]:


Q_dict['9-x-x-1-4-6-x-x-x']


# # We see above that our RL has been trained really well. As per the results, we see that the best possible next action is (6,5) which will result in agent winning

# ### Checking 1 more random state

# In[57]:


Q_dict['x-x-x-8-x-2-1-7-x']


# In[58]:


if (1,7) in Q_dict['9-x-x-1-4-6-x-x-x']:
    print(Q_dict['9-x-x-1-4-6-x-x-x'][(1, 7)])


# ### checking for a particular position

# In[59]:


pos = (0,1)
for key in Q_dict.keys():
    if pos in Q_dict[key]:
        print(key)
        print(Q_dict[key][pos])
    


# #### Checking the states tracked for Q-values convergence

# In[ ]:





# # Testing States tracked

# ### Printing all the tracked states

# In[60]:


States_track


# In[61]:


with open('States_tracked.pkl', 'rb') as handle:
    States_track = pickle.load(handle)


# In[62]:


States_track['x-x-x-8-x-x-1-x-x'][(7, 7)]


# In[63]:


print(len(States_track))


# In[64]:


# looking at the length
for key, value in States_track.items():
    for k,v in value.items():
        print(len(v))


# ## Plotting the graphs for the States_track

# ## Plotting the combined graph for last 1000 points for all the States tracked

# In[65]:


#print(type(xaxis))
xaxis = np.asarray(range(0,1000))
plt.figure(0, figsize=(16,7))
plt.subplot(241)
plt.title('state=(1-6-5-8-x-3-4-7-2) action=(4, 9)')
plt.plot(xaxis,np.asarray(States_track['1-6-5-8-x-3-4-7-2'][(4, 9)][-1000:]))
plt.subplot(242)
plt.title('state=(9-x-x-1-4-6-x-x-x) action=(1, 7)')
plt.plot(xaxis,np.asarray(States_track['9-x-x-1-4-6-x-x-x'][(1, 7)][-1000:]))
plt.subplot(243)
plt.title('state=(x-x-x-8-x-2-1-7-x) action=(2, 3)')
plt.plot(xaxis,np.asarray(States_track['x-x-x-8-x-2-1-7-x'][(2, 3)][-1000:]))
plt.subplot(244)
plt.title('state=(x-x-x-8-x-x-1-x-x) action=(7, 7)')
plt.plot(xaxis,np.asarray(States_track['x-x-x-8-x-x-1-x-x'][(7, 7)][-1000:]))


plt.show()


# ## Plotting the individual graphs

# In[66]:


xaxis = np.arange(len(States_track['9-x-x-1-4-6-x-x-x'][(1, 7)]))
plt.figure(0, figsize=(16,7))

plt.plot(xaxis,np.array(States_track['9-x-x-1-4-6-x-x-x'][(1, 7)]))
plt.show()


# In[67]:


xaxis = np.arange(len(States_track['1-6-5-8-x-3-4-7-2'][(4, 9)]))
plt.figure(0, figsize=(16,7))

plt.plot(xaxis,np.array(States_track['1-6-5-8-x-3-4-7-2'][(4, 9)]))
plt.show()


# In[68]:


xaxis = np.arange(len(States_track['x-x-x-8-x-2-1-7-x'][(2, 3)]))
plt.figure(0, figsize=(16,7))

plt.plot(xaxis,np.array(States_track['x-x-x-8-x-2-1-7-x'][(2, 3)]))
plt.show()


# In[69]:


xaxis = np.arange(len(States_track['x-x-x-8-x-x-1-x-x'][(7, 7)]))
plt.figure(0, figsize=(16,7))

plt.plot(xaxis,np.array(States_track['x-x-x-8-x-x-1-x-x'][(7, 7)]))
plt.show()


# ### From the plots, we can see that the agent has learnt well and we can observe that the values are converging. Thus, we can include that RL training is successful.

# ### Epsilon - decay check

# In[70]:


max_epsilon = 1.0
min_epsilon = 0.001
time = np.arange(0,5000000)
epsilon = []
for i in range(0,5000000):
    epsilon.append(min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.0000008*i))


# In[71]:


plt.plot(time, epsilon)
plt.show()


# In[ ]:





# # Conclusion

# ## We see that the RL has been trained successfully and the results are as expected. Also, from the sample values, we could see that the values are converging

# In[ ]:





# In[ ]:




