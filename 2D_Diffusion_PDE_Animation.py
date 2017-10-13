# Two dimenstional simulation of the diffusion equation
# Assuming a 'box' with four walls of equal temperature
# By Yash Desai
# Animation: https://www.youtube.com/watch?v=3FfiIwahWYs
import numpy as np
from math import pi,sin
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.ion() allows continuous plotting
plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')

# Dimensions of 'box'(distance from wall to wall)
Lx = 0.1
Ly = 0.1

# Number of partitions (finite elements)
n = 15
# Temperature of walls
T1 = 0
T2 = 0
T3 = 0
T4 = 0
# Size of partitions
dx = Lx/n
dy = Ly/n
# Thermal diffusivity (gamma) of copper is used
gamma = 0.000114
# 60 seconds of run time
t_total = 60
# Time partitions of 1/10 of a seoncd
dt = 0.1

# Two arrays with the mid points of each partition
x = np.linspace(dx/2, Lx - dx/2, n)
y = np.linspace(dy/2, Ly - dy/2, n)
# Declare a second array, that is the 2D representation of the
# ones above.
X,Y = np.meshgrid(x,y)

# Initialize the box to a certain temperature distribution
# In this case the distribution is sinusoidal such that the
# maximum temperature is in the middle and the minimum at the
# walls
Tx = np.array([60*sin(pi*i/Lx) for i in x])
Ty = np.array([60*sin(pi*i/Ly) for i in y])
# Declare a second array, that is the 2D representation of the
# ones above. Note the difference between the variables Tx,Ty
# and TX, TY. This is done so that they can be manipulated
# separately. 
TX, TY = np.meshgrid(Tx,Ty)


dTxdt = np.empty(n)
dTydt = np.empty(n)

# In steps of dt, create an array of points in time up to the
# final time.
t = np.arange(0,t_total,dt)

for j in range(1,len(t)):

    ax.clear()

    for i in range(1,n-1):

        # Appling discrete definition of second derivative to
        # both the x and y differentials.
        dTxdt[i] = gamma*((Tx[i+1]-(2*Tx[i])+Tx[i-1])/dx**2)
        dTydt[i] = gamma*((Ty[i+1]-(2*Ty[i])+Ty[i-1])/dx**2)

    # Taking into account boundary conditions. Since for eg.
    # Tx[0-1] DNE. so the fixed temperature of the walls has
    # to be used.
    dTxdt[0] = gamma*((Tx[1]-(2*Tx[0])+T1)/dx**2)
    dTydt[0] = gamma*((Ty[1]-(2*Ty[0])+T3)/dx**2)
    dTxdt[n-1] = gamma*((T2-(2*Tx[n-1])+Tx[n-2])/dx**2)
    dTydt[n-1] = gamma*((T4-(2*Ty[n-1])+Ty[n-2])/dx**2)

    # Update the temperature data for the box
    Tx = Tx + dTxdt*dt
    Ty = Ty + dTydt*dt
    TX, TY = np.meshgrid(Tx,Ty)

    # Z represents the temperature at a point (x,y). Since there
    # are two dimensions, the temperature of the point will
    # equal the sum of the temperature found from each 1D
    # scenario.
    Z = (TX + TY)

    surf = ax.plot_surface(X,Y,Z, cmap=cm.rainbow,
                           linewidth=0, antialiased = False)

    ax.set_zlim(0,120)
    ax.set_xlim(0,0.1)
    ax.set_ylim(0,0.1)
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Distance (meters)')
    ax.set_zlabel('Temperature (C)')
    plt.show()
    plt.pause(0.01)

