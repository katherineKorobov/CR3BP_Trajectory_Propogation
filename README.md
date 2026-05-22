# CR3BP_Trajectory_Propogation
Introduction
------------
This code is a collection of Python scripts that simulate the motion of a spacecraft in the Circular Restricted Three-Body Problem (CR3BP) using odeint, a 4th Order Runge-Kutta method for numerical integration. Particularly, it focuses on the Earth-Moon-Spacecraft system, however, it could also work for other mass ratios.

Contents
--------
There are two main directories: src and data. The src directory holds plotting, utility, and modeling code used in the main.py script. data contains the initial conditions for various orbit families in the cisular domain. Some sets of data hold more than one orbit, and all obrits can be plotted. For verification of orbits, the Jacobi Constant is utilized. The constant is calculated and then plotted. It is a conserved quantity in the CR3BP space, thus, it should be constant throughout the orbit.

TODO: verify the calculated jacobi constant with the provied one.

Dependencies
-------------
(1) Python version 2.7 or greater.

(2) NumPy version 1.12 or greater.

(3) sci py

(4) matplotlib

(5) CSV data with a particular format. All data has been pulled from the [NASA JPL Database](https://ssd.jpl.nasa.gov/tools/periodic_orbits.html). 

To Run
----------
When executing the python script, run in the terminal: python .\src\main.py .\data\<initial-contidions-file>.csv

Information
----------
Author: Katherine Korobov
Affiliation: CU SPUR Cislunar Initial Orbit Determination Using Adbissible Regions Theory
Mentors: Queenique Dinh, Marcus Holzinger, Dan Scheeres
Date: May 28, 2024
Date Restructed: May 21, 2026
