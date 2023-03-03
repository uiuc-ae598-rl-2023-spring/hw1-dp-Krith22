#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

class GridWorld():
    """
    The world is a 5 x 5 grid based on Example 3.5 from Sutton 2019. There are 25 states. We index these states as follows:

        0   1   2   3   4
        5   6   7   8   9
        10  11  12  13  14
        15  16  17  18  19
        20  21  22  23  24

    For example, state "1" is cell "A" in Sutton 2019, state "3" is cell "B", and so forth.

    There are 4 actions. We index these actions as follows:

                1 (up)
        2 (left)        0 (right)
                3 (down)

    If you specify hard_version=True, then the action will be selected uniformly at random 10% of the time.
    """

    def __init__(self, hard_version=False):
        self.hard_version = hard_version
        self.num_states = 25
        self.num_actions = 4
        self.last_action = None
        self.max_num_steps = 100
        self.reset()

    def p(self, s1, s, a):
        if self.hard_version:
            return 0.1 * 0.25 * sum([self._p_easy(s1, s, i) for i in range(4)]) + 0.9 * self._p_easy(s1, s, a)
        else:
            return self._p_easy(s1, s, a)

    def _p_easy(self, s1, s, a):
        if s1 not in range(25):
            raise Exception(f'invalid next state: {s1}')
        if s not in range(25):
            raise Exception(f'invalid state: {s}')
        if a not in range(4):
            raise Exception(f'invalid action: {a}')
        # in A
        if s == 1:
            return 1 if s1 == 21 else 0
        # in B
        if s == 3:
            return 1 if s1 == 13 else 0
        # right
        if a == 0:
            if s in [4, 9, 14, 19, 24]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s + 1 else 0
        # up
        if a == 1:
            if s in [0, 1, 2, 3, 4]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s - 5 else 0
        # left
        if a == 2:
            if s in [0, 5, 10, 15, 20]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s - 1 else 0
        # down
        if a == 3:
            if s in [20, 21, 22, 23, 24]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s + 5 else 0

    def r(self, s, a):
        if self.hard_version:
            return 0.1 * 0.25 * sum([self._r_easy(s, i) for i in range(4)]) + 0.9 * self._r_easy(s, a)
        else:
            return self._r_easy(s, a)

    def _r_easy(self, s, a):
        if s not in range(25):
            raise Exception(f'invalid state: {s}')
        if a not in range(4):
            raise Exception(f'invalid action: {a}')
        # in A
        if s == 1:
            return 10
        # in B
        if s == 3:
            return 5
        # right
        if a == 0:
            if s in [4, 9, 14, 19, 24]:
                return -1
            else:
                return 0
        # up
        if a == 1:
            if s in [0, 1, 2, 3, 4]:
                return -1
            else:
                return 0
        # left
        if a == 2:
            if s in [0, 5, 10, 15, 20]:
                return -1
            else:
                return 0
        # down
        if a == 3:
            if s in [20, 21, 22, 23, 24]:
                return -1
            else:
                return 0

    def step(self, a):
        # Store the action (only used for rendering)
        self.last_action = a

        # If this is the hard version of GridWorld, then change the action to
        # one chosen uniformly at random 10% of the time
        if self.hard_version:
            if random.random() < 0.1:
                a = random.randrange(self.num_actions)

        # Compute the next state and reward
        if self.s == 1:
            # We are in the first teleporting state
            self.s = 21
            r = 10
        elif self.s == 3:
            # We are in the second teleporting state
            self.s = 13
            r = 5
        else:
            # We are in neither teleporting state

            # Convert the state to i, j coordinates
            i = self.s // 5
            j = self.s % 5

            # Apply action to i, j coordinates
            if a == 0:      # right
                j += 1
            elif a == 1:    # up
                i -= 1
            elif a == 2:    # left
                j -= 1
            elif a == 3:    # down
                i += 1
            else:
                raise Exception(f'invalid action: {a}')

            # Would the action move us out of bounds?
            if i < 0 or i >= 5 or j < 0 or j >= 5:
                # Yes - state remains the same, reward is negative
                r = -1
            else:
                # No - state changes (convert i, j coordinates back to number), reward is zero
                self.s = i * 5 + j
                r = 0

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        done = (self.num_steps >= self.max_num_steps)

        return (self.s, r, done)

    def reset(self):
        # Choose initial state uniformly at random
        self.s = random.randrange(self.num_states)
        self.num_steps = 0
        self.last_action = None
        return self.s

    def render(self):
        k = 0
        output = ''
        for i in range(5):
            for j in range(5):
                if k == self.s:
                    output += 'X'
                elif k == 1 or k == 3:
                    output += 'o'
                else:
                    output += '.'
                k += 1
            output += '\n'
        if self.last_action is not None:
            print(['right', 'up', 'left', 'down'][self.last_action])
            print()
        print(output)


# In[2]:


import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from gridworld import GridWorld
env = gridworld.GridWorld(hard_version=False)


# POLICY ITERATION

# In[13]:


def Policyiteration(actions,states,gamma,delta):
    actions = (0, 1,2,3)  
    state = np.linspace(1,25,25)  
    states = np.zeros(25)
    for i in range(len(state)):
        states[i] = int(state[i])
    gamma = 0.95  
    delta = 0.0000001 
    max_policy_iter = 1000  # Maximum number of policy iterations
    max_value_iter = 1000  # Maximum number of value iterations
    pi = [int(0) for s in states]
    V = [0 for s in states]

    V_T = np.zeros(max_policy_iter)
    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        optimal_policy_found = True

    
        for j in range(max_value_iter):
            max_diff = 0  # Initialize max difference
            G = 0
            for s in range(24):
                V_old = V[s]
                V[s] = 0
                A = pi[s]

                
                val = 0  
                r = env.r(s,A)
                for s_next in range(25):
                    val = val + env.p(s_next,s,A) * (r + gamma * V[s_next])  
                G = G + val
                V[s] = val 
                # Update maximum difference
                max_diff = max(max_diff, abs(V[s] - V_old))

                
                
            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break
        V_T[i] = G/25

        # Policy iteration
        # With updated state values, improve policy if needed
        for s in range(25):

            val_max = V[s]
            
            for a in actions:
                val = 0  
                r = env.r(s,a)
                for s_next in range(25):
                    val = val + env.p(s_next,s,a) * (r + gamma * V[s_next])

                # Update policy if (i) action improves value and (ii) action different from current policy
                if val > val_max and pi[s] != a:
                    pi[s] = a
                    val_max = val
                    # V[s] = val_max
                    optimal_policy_found = False

        # If policy did not change, algorithm terminates
        if optimal_policy_found:
            break

    return V


# In[16]:


actions = range(env.num_actions)
states = range(env.num_states)
gamma = 0.95  
delta = 0.0000001 
# Policyiteration(actions,states,gamma,delta)  # Uncomment this to see the values.


# VALUE ITERATION

# In[10]:


def Valueiteration(actions,states,gamma,delta):
    # actions = (0, 1,2,3)  # actions (0=left, 1=right)
    # states = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)  # states (tiles)
    # gamma = 0.95  # discount factor
    # delta = 1e-6  # Error tolerance
    # pi = [0 for s in states] policy
    max_iter = 100  # Maximum number of value iterations
    pi = [int(0) for s in states]
    V = [0 for s in states]
    V
    V_T = np.zeros(max_iter)
    for i in range(max_iter):
        g = 0
        
        max_diff = 0  # Initialize max difference
        # Initialize values
        for s in range(25):
            V_old = V[s]
            V[s] = 0
            K = np.zeros(len(actions))
            # max_val = 0
            for a in actions:

                # Compute state value
                val = 0
                r = env.r(s,a)  # Get direct reward
                for s_next in states:
                    val +=  env.p(s_next,s,a) * (r + gamma * V[s_next])  # Add discounted downstream values
                K[a] = val
                
            V[s] = np.max(K)
            pi[s] = K.argmax(axis=0)
            g = g+ V[s]
            # Update value with highest value
            
            # Update maximum difference
            max_diff = max(max_diff, abs(V[s] - V_old))
        V_T[i] =g/25
        # Update value functions
        

        # If diff smaller than threshold delta for all states, algorithm terminates
        if max_diff < delta:
            break
    return V


# In[15]:


actions = range(env.num_actions)
states = range(env.num_states)
gamma = 0.95  
delta = 0.0000001 
# Valueiteration(actions,states,gamma,delta)  # Uncomment this to see the values.


# SARSA 

# In[21]:


def SARSA(states,actions,gamma,alpha,epsilon):
    # epsilon = 0.1
    total_episodes = 1000
    max_steps = 1000
    # alpha = 0.05
    # gamma = 0.95

    Q = np.zeros((states,actions))
    
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
        
        while t < max_steps:
            state2, reward, done = env.step(action1)
    
            #Choosing the next action
            action2 = choose_action(state2)
            
            #Learning the Q-value
            Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2] - Q[state1, action1])
            
            state1 = state2
            action1 = action2
            
            #Updating the respective vaLues
            t += 1 
            #If at the end of learning process
            if done:
                break    
        return Q


# In[27]:


epsilon = 0.1
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
# Q = SARSA(states,actions,gamma,alpha,epsilon)


# TD(0)

# In[42]:


def optimalpolicy(state):
    pi = np.zeros((len(Q[:,0])))
    for i in range(len(Q[:,0])):
        pi[i] = np.argmax(Q[i,:])
        i = state
    print(pi)
    return pi
def TD_0(states):    
    Q = np.zeros((states))
    total_episodes = 1000
    max_steps = 1000
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        while t < max_steps:
            action1 = optimalpolicy(state1)[state1]
            # Getting the next state
            state2, reward, done = env.step(action1)
            Q[state1] = Q[state1] + alpha * (reward + gamma * Q[state2] - Q[state1])
            state1 = state2
            t += 1
            if done:
                break   
    return Q 


# In[44]:


states = env.num_states
# TD_0(states)


# Q-LEARNING

# In[51]:


def QLearning(states,actions,gamma,alpha,epsilon):
    # epsilon = 0.1
    total_episodes = 1000
    max_steps = 1000
    # alpha = 0.05
    # gamma = 0.95

    Q = np.zeros((states,actions))

    # Function to choose the next action with episolon greedy
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = random.randrange(0, actions, 1)
        else:
            action = np.argmax(Q[state, :])
        return action
    reward=0
    
    # Starting the SARSA learning
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        Q_T = 0
    
        while t < max_steps:
            action1 = choose_action(state1)  
            # Getting the next state
            state2, reward, done = env.step(action1)
            action1p = np.argmax(Q[state2, :])
            # action1p = np.argmax(Q[state2, :])
            #Learning the Q-value
            Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action1p] - Q[state1, action1])
            Q_T += gamma**t * reward
            state1 = state2
            
            #Updating the respective vaLues
            t += 1
            
            #If at the end of learning process
            if done:
                break 
    return Q_T


# In[52]:


epsilon = 0.1
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
# QLearning(states,actions,gamma,alpha,epsilon)


# TD(0)

# In[53]:


def optimalpolicy(state):
    pi = np.zeros((len(Q[:,0])))
    for i in range(len(Q[:,0])):
        pi[i] = np.argmax(Q[i,:])
        i = state
    print(pi)
    return pi
def TD_0(states):    
    Q = np.zeros((states))
    total_episodes = 1000
    max_steps = 1000
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()
        while t < max_steps:
            action1 = optimalpolicy(state1)[state1]
            # Getting the next state
            state2, reward, done = env.step(action1)
            Q[state1] = Q[state1] + alpha * (reward + gamma * Q[state2] - Q[state1])
            state1 = state2
            t += 1
            if done:
                break   
    return Q 


# In[54]:


states = env.num_states
# TD_0(states)

