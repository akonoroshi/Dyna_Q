import numpy as np
import pandas as pd
import math
from operator import add

class Dyna_Q:
    def __init__(self, actions, term_state, gamma=1, alpha=0.7, num_sim=5, num_iter=30, sample_prop=1, method="UCB", c=0.0001, temp=1.5):
        self.actions = actions # List of actions
        self.term_state = term_state # Terminal state
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount rate (close to 0: more immediate reward, 1: more long-term reward)
        self.q_table = pd.DataFrame(columns=self.actions) # states x actions
        self.tc = {} # count of (state, action, state_), used to calculate probability below
        self.t = {} # transition function: probability of seeing state_ given state and action
        self.r = {} # reward function: expected reward of a certain action at a given state
        self.k = num_sim # number of simulations during planning
        self.l = num_iter # maximum number of iterations in one simulation
        self.batch = pd.DataFrame(columns=['s', 'a', 'r', 's_']) # buffer to store actions and their results
        self.sample_prop = sample_prop # proportion of batch used to update Q
        self.method = method
        self.temp = temp

    # If the state is new, add it
    def add_state(self, state):
        if state not in self.q_table.index:
            # Append the new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    np.random.rand(len(self.actions)) * 0.01,
                    index=self.q_table.columns,
                    name=state,
                )
            )

            # Add the new state to T and R
            self.tc[state] = {act : {state_ : 0.00001 for state_ in self.q_table.index} for act in self.actions}
            self.t[state] = {act : {state_ : 1 / len(self.q_table.index) for state_ in self.q_table.index} for act in self.actions}
            self.r[state] = {act : 0 for act in self.actions}
            for state_ in self.q_table.index:
                if state_ == state:
                    break
                for act in self.actions:
                    self.tc[state_][act][state] = 0.00001
                    self.t[state_][act][state] = self.tc[state_][act][state] / sum(self.tc[state_][act].values())
                    self.r[state_][act] = 0


    # Do exploitation and exploration using UCB1 algorithm
    def choose_action(self, observation):
        # Precondition: observation (state) is a dicrete value
        self.add_state(observation)
        max_value = float('-inf')
        max_action = self.actions[0]

        if self.method == "UCB":
            # UCB1
            num_state = sum(sum(self.tc[observation][act].values()) for act in self.actions)
            if num_state < 1: # Prevent log from being negative
                num_state = 1
            for act in self.actions:
                if self.q_table.ix[observation, act] + self.c * math.sqrt(2 * math.log(num_state) / sum(self.tc[observation][act].values())) > max_value:
                    max_action = act
                    max_value = self.q_table.ix[observation, act]

        elif self.method == "softmax":
            pi = np.exp(np.array(self.q_table.ix[observation]) / self.temp) / sum(np.exp(np.array(self.q_table.ix[observation]) / self.temp))
            max_action = np.random.choice(self.actions, p=pi)

        return max_action

    # Happens every single time after choosing action and observing a new state
    def store_result(self, s, a, r, s_):
        # Update model
        self.add_state(s_)

        # Store data to buffer
        df = pd.DataFrame([[s, a, r, s_]], columns=['s', 'a', 'r', 's_'])
        self.batch = self.batch.append(df)
        
    # Happens after the real world enters a terminal state
    # IMPORTANT: CALL AFTER update_Models if you call it together
    def update_Q(self):
        samples = self.batch.sample(frac=self.sample_prop)
        for index, sample in samples.iterrows():
            q_target = sample['r'] + self.gamma * self.q_table.ix[sample['s_'], :].max()
            self.q_table.ix[sample['s'], sample['a']] += self.alpha * (q_target - self.q_table.ix[sample['s'], sample['a']])
        
        self.batch = self.batch.iloc[0:0]

    # Happens after the real world enters a terminal state
    # IMPORTANT: CALL BEFORE update_Q
    def update_Models(self):
        samples = self.batch.sample(frac=self.sample_prop)
        for index, sample in samples.iterrows():
            self.tc[sample['s']][sample['a']].update({sample['s_'] : self.tc[sample['s']][sample['a']][sample['s_']] + 1})
            # self.tc[s][a][s_] += 1
            self.t[sample['s']][sample['a']].update({sample['s_'] : self.tc[sample['s']][sample['a']][sample['s_']] / sum(self.tc[sample['s']][sample['a']].values())})
            #self.t[s][a][s_] = self.tc[s][a][s_] / sum(self.tc[s][a].values())
            self.r[sample['s']].update({sample['a'] : (1 - self.alpha) * self.r[sample['s']][sample['a']] + self.alpha * sample['r']})
            # self.r[s][a] = (1 - self.alpha) * self.r[s][a] + self.alpha * r

    # Happens after the real world enters a terminal state and you update Q and models
    def planning(self):
        for i in range(self.k):
            # Choose an initial state randomly
            while True:
                state = np.random.choice(self.q_table.index)
                if state != self.term_state:
                    break
            
            # Start a simulation
            for j in range(self.l):
                # Choose an action randomly
                action = np.random.choice(self.actions)

                # Get a new state and reward according to current functions
                prob = np.array(list(self.t[state][action].values()))
                prob /= prob.sum() # Normalize
                state_ = np.random.choice(self.q_table.index, p=prob)
                reward = self.r[state][action]
                self.store_result(state, action, reward, state_)
                state = state_
                if state == self.term_state:
                    break
        
            self.update_Q()
