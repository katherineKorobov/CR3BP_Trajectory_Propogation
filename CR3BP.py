"""
Katherine Korobov
Cislunar Initial Orbit Determination Using Adbissible Regions Theory CU SPUR Project
Mentors: Queenique Dinh, Marcus Holzinger
Date: May 28, 2024
This code is a Python script that simulates the motion of a spacecraft in the 
Circular Restricted Three-Body Problem (CR3BP) using the - method for numerical integration.
"""

# Import useful libraries
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as spi 

# Create functions
def setConstants():
    mu = 
    return mu

def readCSV(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1) # Read CSV file and skip header, generates new row for each line
    return data

def initialConditions(filename):
    data = readCSV(filename)
    x = data[0, 0]  # Initial x position
    y = data[0, 1]  # Initial y position
    z = data[0, 2]  # Initial z position
    x_prime = data[0, 3]  # Initial x velocity
    y_prime = data[0, 4]  # Initial y velocity
    z_prime = data[0, 5]  # Initial z velocity
    return [x, y, z, x_prime, y_prime, z_prime]


integrationMethod

plot

def main():
    mu = setConstants()
    state = initialConditions()

    #time of integration
    t_0 = 0  # Initial time
    t_f = 1000  # Final time

    t_span = (t_0, t_f)  # Time span for integration

    



define integrationMethod(state):
    define the integration method to be used, e.g., Runge-Kutta 4th order


#use integration method and EOM to propagate a new state vector 


#EOM for CR3BP
x_2prime = 2 * y_prime + x - ((1 - mu) * (x + mu)) / (r1**3) - (mu * (x - 1 + mu)) / r2**3
y_2prime = -2 * x_prime + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
z_2prime = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

#for loop to propagate state vector for a set number of iterations

#plot results


orbits:
L2 Souther Halo
L1 Norhern Halo
