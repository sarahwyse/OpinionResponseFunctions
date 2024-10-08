#ODE model with M=1

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

#model
def M1(t, z):    
    A, B = z
    return [B*(A+C) - A*B,
            A*B - B*(A+C)]

#run simulation
C = 0.05
y0 = [0, 1-C]
sol = int.solve_ivp(M1, [0, 150], y0, method='LSODA')

#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,2))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T)
plt.xlabel('Time')
plt.ylabel('Uncommitted proportion')
plt.legend([r'$y_A$', '$y_B$'])

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()
