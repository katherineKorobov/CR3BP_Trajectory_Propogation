'''
File: main.py
Description: The main function for the project: "CR3BP_Trajectory Propogation".
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 26 May 2026
'''

class State:

    # holds position and velocity of an object
    x = 0
    y = 0
    z = 0

    dxdt = 0
    dydt = 0
    dzdt = 0

    def __init__(self, x, y, z, dxdt, dydt, dzdt):
        self.x = x
        self.y = y
        self.z = z
        
        self.dxdt = dxdt
        self.dydt = dydt
        self.dzdt = dzdt

        pass