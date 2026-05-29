'''
File: main.py
Description: The main function for the project: "CR3BP_Trajectory Propogation".
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 26 May 2026
'''

import numpy as np

class State:

    # holds position and velocity of an object
    x = 0
    y = 0
    z = 0

    dxdt = 0
    dydt = 0
    dzdt = 0

    full_state = np.array([x, y, z, dxdt, dydt, dzdt])

    def __init__(self, x, y, z, dxdt, dydt, dzdt):
        self.x = x
        self.y = y
        self.z = z
        
        self.dxdt = dxdt
        self.dydt = dydt
        self.dzdt = dzdt

        self.full_state = np.array([x, y, z, dxdt, dydt, dzdt])

        pass