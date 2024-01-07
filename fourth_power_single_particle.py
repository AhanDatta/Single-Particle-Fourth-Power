import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt
import pandas as pd

"""
Here we suppose a single particle in a one dimensional potential given by x^4
For simplicity, we take mass and the unit-fixing constant for the potential to one
H = p^2 + x^4. From this, we find the resulting eom
dx/dt = 2p, and dp/dt = -4x^3
From this, we get a system of equations which we will approximate
"""

#Settings constants for the sim, using phase space (x, p) where both are scalar-like
END_TIME = 10.0
INITIAL_POSITION = 1.0
INITIAL_MOMENTUM = 0.0
INITIAL_STATE = [INITIAL_POSITION, INITIAL_MOMENTUM]

#Takes a point in phase space and return the time derivate as a vector (np array of size 2)
def eoms(t: float, phase_point):
    curr_position = phase_point[0]
    curr_momentum = phase_point[1]
    return np.array([2 * curr_momentum, -4 * (curr_position**3)])

#Runs solution and stores solution arrays
solution = integrate.solve_ivp(eoms, (0, END_TIME), INITIAL_STATE, max_step = 0.01, vectorized=True)
time_data = solution.t
position_data = solution.y[0]
momentum_data = solution.y[1]

#Plots the position and momentum 
fig, axd = plt.subplots(2)
axd[0].set_ylabel("Position")
axd[1].set_ylabel("Momentum")
axd[1].set_xlabel("Time")

axd[0].plot(time_data, position_data) 
axd[1].plot(time_data, momentum_data)
plt.show()

#Writes out data to csv
df = pd.DataFrame({'Time': time_data, 
                   'Position': position_data, 
                   'Momentum': momentum_data})
df.to_csv('fourth_power_single_particle_data.csv')