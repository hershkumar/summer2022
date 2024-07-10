import numpy as np 
from matplotlib import pyplot as plt
import gvar as gv

# read in the data
data = np.loadtxt('5+5.csv', delimiter=',')


num_params = data[:,0]
gs = data[:,1]
energies = data[:,2]
uncerts = data[:,3]

g_neg_1 = gs == -1
g_neg_05 = gs == -0.5
g0 = gs == 0
g05 = gs == .5
g1 = gs == 1
g15 = gs == 1.5
g2 = gs == 2

plt.errorbar(num_params[g0], energies[g0], yerr=uncerts[g0], label="$g=0$", fmt='-o')
plt.errorbar(num_params[g05], energies[g05], yerr=uncerts[g05], label="$g=0.5$", fmt='-v')
plt.errorbar(num_params[g1], energies[g1], yerr=uncerts[g1], label="$g=1.0$", fmt='-s')
plt.errorbar(num_params[g15], energies[g15], yerr=uncerts[g15], label="$g=1.5$", fmt='-*')
plt.errorbar(num_params[g2], energies[g2], yerr=uncerts[g2], label="$g=2.0$", fmt='-D')


plt.xlabel("Number of Parameters")
plt.ylabel("Energy")
plt.title("Energy vs. Number of Parameters for Various Couplings")
plt.xscale('log')
# plt.legend()
plt.grid()
# plt.show()

gneg1var = gv.gvar(energies[g_neg_1], uncerts[g_neg_1])[-1]
gneg05var = gv.gvar(energies[g_neg_05], uncerts[g_neg_05])[-1]
g0var = gv.gvar(energies[g0], uncerts[g0])[-1]
g05var = gv.gvar(energies[g05], uncerts[g05])[-1]
g1var = gv.gvar(energies[g1], uncerts[g1])[-1]
g15var = gv.gvar(energies[g15], uncerts[g15])[-1]
g2var = gv.gvar(energies[g2], uncerts[g2])[-1]

print("g=",-1, ", E_0=",gneg1var)
print("g=",-.5, ", E_0=",gneg05var)
print("g=",0, ", E_0=",g0var)
print("g=",.5, ", E_0=",g05var)
print("g=",1, ", E_0=",g1var)
print("g=",1.5, ", E_0=",g15var)
print("g=",2, ", E_0=",g2var)