import numpy as np
from Easy21 import TERMINAL

class Sarsa():
     def __init__(self,env,lamb,gamma,Q_star=None):
         self.env = env
         self.policy = np.zeros(self.env.state_combinations)
         self.Q = np.zeros(self.env.state_combinations)
         self.counts = np.zeros(self.env.state_combinations)
         self.returns = np.zeros(self.env.state_combinations)
         self.state_counts = np.zeros(self.env.state_combinations[:2])
         self.elig_trace = np.zeros(self.env.state_combinations)
         self.gamma = gamma
         self.lamb = lamb
         self.Q_star = Q_star
         
     def get_epsilon(self,N):
         N_0 = 100
         return N/(N_0 + N)
     
     def reset_elig(self,):
         self.elig_trace = np.zeros(self.env.state_combinations)
         
     def compute_MSE(self,Q):
         return np.mean((self.Q_star - Q)**2)
     
     def epsilon_greedy_policy(self,Q,N,state):
         dealer, player = state
         epsilon = self.get_epsilon(np.sum(N[dealer-1, player-1]))
         if np.random.rand() < (1 - epsilon):
             action = np.argmax(Q[dealer-1, player-1, :])
         else:
             action = np.random.choice([0,1])
             
         return action
         
         

         
         
     
     def run_sarsa(self,episodes):
         
         MSE_list = []
         
         for i in range(episodes):
         
             if i % 1000 == 0:
                if self.Q_star is not None:
                    MSE_list.append([i,self.compute_MSE(self.Q)])
                    
             self.reset_elig()
             state = self.env.reset()
             action = self.epsilon_greedy_policy(self.Q,self.state_counts,state)

             while state[1] != TERMINAL:

                 state_prime,reward = self.env.step(state,action)

                 if state_prime[1] == TERMINAL:
                     state = (state[0] -1, state[1] - 1)
                     delta = reward - self.Q[state][action]
                     self.elig_trace[state][action] =+ 1.0
                     self.state_counts[state] += 1.0
                     self.counts[state][action] += 1.0

                     idx = np.where(self.elig_trace != 0)
                     self.Q[idx] += delta*self.elig_trace[idx]/self.counts[idx]
                     self.elig_trace[idx] *= self.gamma*self.lamb
                     break
                     
                 action_prime = self.epsilon_greedy_policy(self.Q,self.state_counts,state_prime)
                 state_prime = (state_prime[0] - 1,state_prime[1] - 1)
                 state = (state[0] - 1,state[1] - 1)
                 delta = reward + self.gamma*self.Q[state_prime][action_prime] - self.Q[state][action]
                 
                 
                 self.elig_trace[state][action] =+ 1.0
                 self.state_counts[state] += 1.0
                 self.counts[state][action] += 1.0
                 
                 idx = np.where(self.elig_trace != 0)
                 self.Q[idx] += delta*self.elig_trace[idx]/self.counts[idx]
                 self.elig_trace[idx] *= self.gamma*self.lamb
                 
                 state = state_prime
                 action = action_prime

                 
                 
         
         return np.array(self.Q),np.array(MSE_list)
         
