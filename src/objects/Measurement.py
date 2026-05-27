'''
File: .py
Description:
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 27 May 2026
'''

import numpy as np

from src.objects.State import State

class Measurement:

    def __init__(self, initial_state, initial_time, jac_const, period, mass_ratio):
        self.init_state_vector = initial_state # of class State
        self.state_vector = initial_state # becomes a vector holding States for each time step
        
        self.init_time = initial_time # start of propagation
        self.time = initial_time # becomes a vector that holds all time steps

        self.jacobi_constant = jac_const # constant that identifies the orbit
        self.period = period # constant for time length of orbit
        self.mu = mass_ratio # constant that defines with bodies are used in CR3BP

        pass