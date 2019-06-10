"""Environment file for Reinforcement Learning Project"""
# Import routines

import numpy as np
import math
import random
import time

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(1,2),(1,3),(1,4),(1,5),(2,1),(2,3),(2,4),(2,5),
                            (3,1),(3,2),(3,4),(3,5),(4,1),(4,2),(4,3),(4,5),
                            (5,1),(5,2),(5,3),(5,4),(0,0)]
        self.state_space = []
        for i in range(m):
            for j in range(t):
                for k in range(d):
                    self.state_space.append([i,j,k])

        #self.state_init = self.state_space[np.random.choice(len(self.state_space))]
        self.state_init = [np.random.choice(range(m)),0,0]
        # Start the first round
        self.day_of_month = 0
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = list(np.eye(m)[np.array(state[0])])+list(np.eye(t)[np.array(state[1])])+list(np.eye(d)[np.array(state[2])])
        return state_encod


    # Use this function if you are using architecture-2
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""


    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]


        actions.append((0,0))

        return possible_actions_index,actions



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        if action == (0,0):
            reward = -5
        else:
            rev = R*(Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]])
            cost = C*(Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]] + Time_matrix[state[0]][action[0]-1][state[1]][state[2]])
            reward = rev - cost
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = list(state)
        terminal=bool(0)

        if action == (0,0):

            next_state[1] = next_state[1] + 1
            if(next_state[1]>23):
                next_state[1] = 0
                next_state[2] = next_state[2]+1
                self.day_of_month +=1
                if (next_state[2] == 7):
                    next_state[2] = 0
            time=1

        else:
            next_state[0] = action[1]-1
            time = Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]] + Time_matrix[state[0]][action[0]-1][state[1]][state[2]]
            """condition for out of bound error"""
            if (next_state[1] + time) <= 23:
                next_state[1] = int(next_state[1] + time)
            else:
                next_state[1] = int(next_state[1] + time - 24)
                next_state[2] = next_state[2] + 1
                self.day_of_month +=1
                if next_state[2] == 7:
                    next_state[2] = 0
        """termination codition"""
        if self.day_of_month == 30:
            terminal = bool(1)

        return next_state,terminal


    def reset(self):
        return self.action_space, self.state_space, self.state_init
