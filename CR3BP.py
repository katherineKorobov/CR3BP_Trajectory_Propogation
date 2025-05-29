#Katherine Korobov
#Cislunar Initial Orbit Determination Using Adbissible Regions Theory CU SPUR Project
#Mentor: Queenique Dinh, Marcus Holzinger
#Date: May 28, 2024
#This code is a Python script that simulates the motion of a spacecraft in the Circular Restricted Three-Body Problem (CR3BP) using the -- method for numerical integration.

#imports libraries needed, runge kutta integration?
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

define setConstants():
    mu = 
    r1 = 
    r2 =

#set initial conditions

Define initial statevector

define integration time?

integrate 

plot



define integrationMethod(state):
    define the integration method to be used, e.g., Runge-Kutta 4th order


#set initial conditions and initial condition vector/array
state = [x, y, z, x_prime, y_prime, z_prime]  # initial state vector
# Constants for the CR3BP
setConstants()

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
