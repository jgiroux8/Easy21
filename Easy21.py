import numpy as np

TERMINAL = -1


class Easy21():
    def __init__(self,):
        self.colors = ['Black','Red']
        self.dealer_state = np.random.randint(1,11)
        self.agent_state = np.random.randint(1,11)
        self.actions = [0,1] # Hit, Stick
        self.state_combinations = (10,21,2)
        # Dealer can be in range (0-10, or terminal)
    
    def sample_deck(self,):
        card = np.random.randint(low=1,high=11)
        # 0 is black -> p=0.66, red is 1, p = 0.33
        color = np.random.choice([1,-1], p=[0.67,0.33])
        # 1 is black, -1 is red
        
        return card,color
    
    def check_terminal(self,state):
        if state > 21 or state < 1:
            return True
        else:
            return False
        
    def reset(self,):
        self.__init__()
        return (self.dealer_state,self.agent_state)
    
    def step(self,state,action):
        
        # Agent
        action = ('hit','stick')[action]
        if action == 'hit':
            # update state
            card,color = self.sample_deck()
            self.agent_state += color*card
            
            if self.check_terminal(self.agent_state):
                next_state, reward = TERMINAL,-1
                
            else:
                next_state,reward = self.agent_state,0
            
        # If we stick, update dealer state
        elif action == 'stick':
            next_state = TERMINAL
            while 0 <= self.dealer_state <= 17:
                card,color = self.sample_deck()
                self.dealer_state += color*card
                                 
            if self.check_terminal(self.dealer_state):
                reward = 1
                self.dealer_state = TERMINAL
                
            else:
                reward = int(bool(self.dealer_state < self.agent_state)) - int(bool(self.dealer_state > self.agent_state))
                self.dealer_state = TERMINAL
        
        return (self.dealer_state,next_state),reward
