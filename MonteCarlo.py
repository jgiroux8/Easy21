import numpy as np
from Easy21 import TERMINAL


class MC_Control():
    def __init__(self,env):
        self.env = env
        self.policy = np.zeros(self.env.state_combinations)
        self.Q = np.zeros(self.env.state_combinations)
        self.counts = np.zeros(self.env.state_combinations)
        self.returns = np.zeros(self.env.state_combinations)
        self.state_counts = np.zeros(self.env.state_combinations[:2])
        
        
    def get_epsilon(self,N):
        N_0 = 100
        return N/(N_0 + N)

    def epsilon_greedy_policy(self,Q,N,state):
        dealer, player = state
        epsilon = self.get_epsilon(np.sum(N[dealer-1, player-1]))
        if np.random.rand() < (1 - epsilon):
            action = np.argmax(Q[dealer-1, player-1, :])
        else:
            action = np.random.choice([0,1])
            
        return action
        
        

    def run_MC(self,episodes):
        
        for i in range(episodes):
            exp = []
            state = self.env.reset()
            while True:
                action = self.epsilon_greedy_policy(self.Q,self.state_counts,state)
                state_,reward = self.env.step(state,action)
                exp.append((state,action,reward))
                state = state_
                
                if state[1] == TERMINAL:
                    break
                
            for j in range(len(exp)):
                state,action,reward = exp[j]
                # Find the first occurance of the (state, action) pair in the episode
                for k in range(len(exp)):
                    s,a,r = exp[k]
                    if (s == state) and (a == action):
                        idx = k
                        break
                
                G = sum([x[2] for i,x in enumerate(exp[idx:])])

                state = (state[0] - 1,state[1] - 1)
                self.returns[state][action] += G
                self.counts[state][action] += 1.0
                self.state_counts[state] += 1.0
                self.Q[state][action] =+ (self.returns[state][action] - self.Q[state][action])/self.counts[state][action]
        
        
        return np.array(self.Q)
    
