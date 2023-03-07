#!/usr/bin/env python
# coding: utf-8

# In[226]:


import random
import matplotlib.pyplot as plt
import gridworld
from gridworld import GridWorld
env = gridworld.GridWorld(hard_version=False)
def Policyiteration(env,action,state,gamma,delta):
    actions = np.arange(0,action,1)  
    states = np.arange(0,state,1) 
    gamma = 0.95  
    delta = 0.000001 
    max_policy_iter = 100  # Maximum number of policy iterations
    max_value_iter = 100  # Maximum number of value iterations
    
    pi = np.zeros(state)
    for i in range(state):
        pi[i] = random.randrange(4)
    V = np.zeros(state)

    V_T = np.zeros(max_policy_iter)
    t = 0
    
    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        
        
        for j in range(max_value_iter):
            max_diff = 0  # Initialize max difference
            G = 0
            for s in range(25):
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
            
                
        V_T[i] = G/25

        # Policy iteration
        # With updated state values, improve policy if needed
        for s in range(25):

            val_max = V[s]
            
            for a in range(4):
                val = 0  
                r = env.r(s,a)
                for s_next in range(25):
                    val = val + env.p(s_next,s,a) * (r + gamma * V[s_next])

                # Update policy if (i) action improves value and (ii) action different from current policy
                if val > val_max and pi[s] != a:
                    pi[s] = a
                    val_max = val
                
        # If policy did not change, algorithm terminates
        

    return V_T, V , pi , G, t

def Valueiteration(env,action,state,gamma,delta):
    actions = np.arange(0,action,1)  
    states = np.arange(0,state,1) 
    max_iter = 100  # Maximum number of value iterations
    pi = np.zeros(state)
    for i in range(state):
        pi[i] = random.randrange(4)
    V = [0 for s in states]
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
    return V_T, V , pi 




def SARSA(env,states,actions,gamma,alpha,epsilon):
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

def QLearning(env,states,actions,gamma,alpha,epsilon):
    # epsilon = 0.1
    total_episodes = 100
    max_steps = 100
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

def trajectory(pi):
    env = gridworld.GridWorld(hard_version=False)

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


# In[227]:


epsilon = 0.8
alpha = 0.05
gamma = 0.95
delta = 1e-6
states = env.num_states
actions = env.num_actions


# RESULTS FOR POLICY ITERATION

# In[228]:


Vmean_Policy, V_Policy, pi_policy ,g , t= Policyiteration(env,actions,states,gamma,delta)
plt.plot(Vmean_Policy)


# In[229]:


trajectory(pi_policy)


# 2. POLICY PLOT 

# In[230]:


plt.plot(pi_policy)


# RESULT FOR VALUE ITERATION

# In[231]:


Vmean_Policy, V_Policy, pi_policy = Valueiteration(env,actions,states,gamma,delta)
plt.plot(Vmean_Policy)


# In[232]:


trajectory(pi_policy)


# In[233]:


plt.plot(pi_policy)


# SARSA RESULTS

# In[57]:


epsilon = 0.8
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
Q_T_sarsa, Q_sarsa = SARSA(env,states,actions,gamma,alpha,epsilon)


# In[46]:


plt.plot(Q_T_sarsa)
plt.ylabel('Return')
plt.xlabel('Episodes')


# In[105]:


e  = np.linspace(0.2,0.8,3)
x = np.linspace(1,1001,1000)
for i in range(len(e)):
    Q_T, Q = SARSA(env,states,actions,gamma,alpha,e[i])
    plt.plot(Q_T, label = e[i])

plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-50,50)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# In[108]:


alpha  = np.linspace(0.02,0.05,3)
x = np.linspace(1,1001,1000)
for i in range(len(alpha)):
    Q_T, Q = SARSA(env,states,actions,gamma,alpha[i],epsilon)
    plt.plot(Q_T, label = alpha[i])
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-30,50)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# In[109]:


pi = optimalpolicy(Q_sarsa)
plt.plot(pi,label = 'Policy - SARSA')


# In[139]:


trajectory(pi)


# In[119]:


states = env.num_states
alpha = 0.05
gamma = 0.95
V_SARSA = TD_0(Q_sarsa, states)
plt.plot(V_SARSA)


# RESULTS FOR Q-LEARNING

# In[120]:


epsilon = 0.8
alpha = 0.05
gamma = 0.95
states = env.num_states
actions = env.num_actions
Q_T_QL, Q_QL = QLearning(env,states,actions,gamma,alpha,epsilon)


# In[121]:


plt.plot(Q_T_QL)
plt.ylabel('Return')
plt.xlabel('Episodes')


# In[122]:


e  = np.linspace(0.2,0.8,3)
x = np.linspace(1,1001,1000)
for i in range(len(e)):
    Q_T, Q = QLearning(env,states,actions,gamma,alpha,e[i])
    plt.plot(Q_T, label = e[i])

plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-50,50)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# In[123]:


alpha  = np.linspace(0.02,0.05,3)
x = np.linspace(1,1001,1000)
for i in range(len(alpha)):
    Q_T, Q = QLearning(env,states,actions,gamma,alpha[i],epsilon)
    plt.plot(Q_T, label = alpha[i])
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.ylim(-30,50)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# In[124]:


pi = optimalpolicy(Q_QL)
plt.plot(pi,label = 'Policy - SARSA')


# In[140]:


trajectory(pi)


# In[126]:


states = env.num_states
alpha = 0.05
gamma = 0.95
V_QL = TD_0(Q_QL, states)
plt.plot(V_QL)

