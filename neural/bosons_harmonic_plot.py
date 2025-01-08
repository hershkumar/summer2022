import numpy as np 
from scipy.special import gamma
import matplotlib.pyplot as plt


# read in the points from harmonic_bosons_delta_true.csv file
data = np.genfromtxt('harmonic_bosons_delta_true.csv', delimiter=',')
Es = data[:,0]
gs = data[:,1]

# read data from 2_bosons_delta.csv
data = np.genfromtxt('2_bosons_delta_5000.csv', delimiter=',')
g_points = data[:,0]
E_points = data[:,1]
uncert_points = data[:,2]

data2 = np.genfromtxt('2_bosons_delta_80000.csv', delimiter=',')
g_points2 = data2[:,0]
E_points2 = data2[:,1]
uncert_points2 = data2[:,2]


plt.plot(gs, Es, '-',color='black', label='Analytic Expression')
plt.errorbar(g_points, E_points, yerr=uncert_points, fmt='o', markersize='5', color='red', label=r'$\beta \approx 5000$')
# plt.errorbar(g_points2, E_points2, yerr=uncert_points2, fmt='o', markersize='5', color='blue', label=r'\beta \approx 80000')
plt.xlabel(r'$g$')
plt.ylabel(r'$E_0$')
plt.xlim(-.6,.6)
plt.title('$N=2$ Bosons in Harmonic Trap with $\delta$ Potential')
plt.grid(True)
plt.legend()
# plt.savefig("2_bosons_delta_vs_exact.svg", format="svg")
plt.show()
