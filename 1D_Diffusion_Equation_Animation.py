# One dimensional simulation of the diffusion equation
# Assuming a fully insulated rod, with ends at constant temp.
# By Yash Desai
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Length of 'rod'
L = 0.1
# Number of partitions of rod (finite elements)
n = 15
# Constant temperature of the left end
T1 = 60
# Constant temperature of the right end
T2 = 20
# Size of partitions 
dx = L/n
# Thermal diffusivity (gamma) of copper is used
gamma = 0.000114
# 60 seconds of run time
t_total = 60
# Time partitions of 1/10 of a second
dt = 0.1

# Make an array with positions of middle of each partition
x = np.linspace(dx/2, L - dx/2,n)

# Initialize rod to initial temperature distribution.
# T(0,x) = 2x
T = np.array([2*i for i in range(n)])
dTdt = np.empty(n)

# In steps of dt, create an arrray of points in time
t = np.arange(0,t_total,dt)

for j in range(1,len(t)):
    # Can include plt.clf() for updating plot
    # plt.clf()
    for i in range(1,n-1):

        # Applying discrete definition of second derivative
        dTdt[i] = gamma*((T[i+1]-(2*T[i])+T[i-1])/dx**2)
        
    # Taking into account boundary condition. ie. T[0-1] DNE
    dTdt[0] = gamma*((T[1]-(2*T[0])+T1)/dx**2)
    dTdt[n-1] = gamma*((T2-(2*T[n-1])+T[n-2])/dx**2)

    # Update temperature data for rod
    T = T + dTdt*dt
    plt.plot(x,T)
    plt.axis([0,L,0,100])
    plt.xlabel('Distance (meters)')
    plt.ylabel('Temperature (Celsius)')
    plt.plot()
    plt.pause(0.01)
