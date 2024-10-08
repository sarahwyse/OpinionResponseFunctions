#ODE model with M=2

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

#model
def M2(t, z):
    AA, AB, BA, BB = z
    return [-AA*(BB+0.5*AB+0.5*BA) + AB*(AA+0.5*AB+0.5*BA+C),
            -AB*(AA+0.5*AB+0.5*BA+C) + BB*(AA+0.5*AB+0.5*BA+C) - AB*(BB+0.5*AB+0.5*BA) + BA*(AA+0.5*AB+0.5*BA+C),
            -BA*(BB+0.5*AB+0.5*BA) + AA*(BB+0.5*AB+0.5*BA) - BA*(AA+0.5*AB+0.5*BA+C) + AB*(BB+0.5*AB+0.5*BA),
            -BB*(AA+0.5*AB+0.5*BA+C) + BA*(BB+0.5*AB+0.5*BA)]

#run simulation
C = 0.05
y0 = [0, 0, 0, 1-C]    
sol = int.solve_ivp(M2, [0, 150], y0, method='LSODA')

#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,4))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T)
plt.xlabel('Time')
plt.ylabel('Uncommitted proportion')
plt.legend([r'$y_{AA}$', '$y_{AB}$', '$y_{BA}$', '$y_{BB}$'])

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()
