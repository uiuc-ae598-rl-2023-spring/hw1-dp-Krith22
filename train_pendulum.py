#!/usr/bin/env python
# coding: utf-8

# In[61]:


import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
env = discrete_pendulum.Pendulum()


# SARSA

# In[51]:


def SARSA(states,actions,gamma,alpha,epsilon):
    # epsilon = 0.1
    total_episodes = 1000
    max_steps = 1000
    # alpha = 0.05
    # gamma = 0.95

    Q = np.zeros((states,actions))
    Q_T = np.zeros(total_episodes)
    # Function to choose the next action with episolon greedy
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = random.randrange(0, actions, 1)
        else:
            action = np.argmax(Q[state, :])
        return action 
    # Starting the SARSA learning
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        action1 = choose_action(state1)
        QT = 0
        while t < max_steps:
            state2, reward, done = env.step(action1)
    
            #Choosing the next action
            action2 = choose_action(state2)
            
            #Learning the Q-value
            Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2] - Q[state1, action1])
            QT += gamma**t * reward
            state1 = state2
            action1 = action2
            
            #Updating the respective vaLues
            t += 1 
            #If at the end of learning process
            if done:
                break 
        Q_T[episode] = QT   
    return Q_T, Q


# Q-Learning

# In[79]:


def QLearning(states,actions,gamma,alpha,epsilon):
    # epsilon = 0.1
    total_episodes = 1000
    max_steps = 1000
    # alpha = 0.05
    # gamma = 0.95

    Q = np.zeros((states,actions))
    Q_T = np.zeros(total_episodes)
    # Function to choose the next action with episolon greedy
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = random.randrange(0, actions, 1)
        else:
            action = np.argmax(Q[state, :])
        return action
    
    
    # Starting the SARSA learning
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        QT = 0
    
        while t < max_steps:
            action1 = choose_action(state1)  
            # Getting the next state
            state2, reward, done = env.step(action1)
            action1p = np.argmax(Q[state2, :])
            # action1p = np.argmax(Q[state2, :])
            #Learning the Q-value
            Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action1p] - Q[state1, action1])
            QT += gamma**t * reward
            state1 = state2
            
            #Updating the respective vaLues
            t += 1
            
            #If at the end of learning process
            if done:
                break 
        Q_T[episode] = QT
    return Q_T, Q


# TD(0) and Optimal-Policy

# In[78]:


def optimalpolicy(Q):
    pi = np.zeros((len(Q[:,0])))
    for i in range(len(Q[:,0])):
        pi[i] = np.argmax(Q[i,:])
    # print(pi)
    return pi
def TD_0(Q_algorithm, states):    
    pi = optimalpolicy(Q_algorithm)
    Q = np.zeros((states))
    total_episodes = 1000
    max_steps = 1000
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        while t < max_steps:
            action1 = pi[state1]
            # Getting the next state
            state2, reward, done = env.step(action1)
            Q[state1] = Q[state1] + alpha * (reward + gamma * Q[state2] - Q[state1])
            state1 = state2
            t += 1
            if done:
                break   
    return Q


# Trajectory 

# In[83]:


def trajectory(pi):
    env = discrete_pendulum.Pendulum()

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    
    done = False
    while not done:
        a = pi[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


# In[54]:


epsilon = 0.8
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
Q_T_sarsa, Q_sarsa = SARSA(states,actions,gamma,alpha,epsilon)


# SARSA plot - The return versus the number of episodes.

# In[55]:


plt.plot(Q_T_sarsa)
plt.ylabel('Return')
plt.xlabel('Episodes')


# Learning Curve SARSA for different values of Epsilon

# In[58]:


e  = np.linspace(0.2,0.8,3)
x = np.linspace(1,1001,1000)
for i in range(len(e)):
    Q_T, Q = SARSA(states,actions,gamma,alpha,e[i])
    plt.plot(Q_T, label = i)

plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-2,15)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Learning Curve SARSA for different values of Alpha

# In[60]:


alpha  = np.linspace(0.02,0.05,3)
x = np.linspace(1,1001,1000)
for i in range(len(alpha)):
    Q_T, Q = SARSA(states,actions,gamma,alpha[i],epsilon)
    plt.plot(Q_T, label = alpha[i])
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-2,15)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Q-Learning plot - The return versus the number of episodes.

# In[62]:


epsilon = 0.8
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
Q_T_QL, Q_QL = QLearning(states,actions,gamma,alpha,epsilon)
plt.plot(Q_T_QL)
plt.ylabel('Return')
plt.xlabel('Episodes')


# Learning Curve Q-Learning for different values of Epsilon

# In[86]:


e  = np.linspace(0.2,0.8,3)
x = np.linspace(1,1001,1000)
for i in range(len(e)):
    Q_T, Q = QLearning(states,actions,gamma,alpha,e[i])
    plt.plot(Q_T, label = i)

plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-2,15)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Learning Curve Q-Learning for different values of Alpha

# In[87]:


alpha  = np.linspace(0.02,0.05,3)
x = np.linspace(1,1001,1000)
for i in range(len(alpha)):
    Q_T, Q = QLearning(states,actions,gamma,alpha[i],epsilon)
    plt.plot(Q_T, label = alpha[i])
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-2,15)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Plot of Policy - SARSA

# In[70]:


pi = optimalpolicy(Q_sarsa)
plt.plot(pi,label = 'Policy - SARSA')


# In[84]:


trajectory(pi)


# Plot of Policy - QLearning

# In[71]:


pi = optimalpolicy(Q_QL)
plt.plot(pi)


# In[85]:


trajectory(pi)


# Plot of State-Value Function : SARSA

# In[80]:


states = env.num_states
V_SARSA = TD_0(Q_sarsa,states)
plt.plot(V_SARSA)


# Plot of State-Value Function - QLearning

# In[75]:


states = env.num_states
V_QL = TD_0(Q_QL,states)
plt.plot(V_QL)

