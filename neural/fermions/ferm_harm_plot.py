from matplotlib import pyplot as plt
import numpy as np
import gvar as gv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# read in the ferm_harm_nodelta.csv file as a list of rows
with open('ferm_harm_nodelta_lowN.csv', 'r') as f:
    rows = f.readlines()

def parse_row(row):
    # split the row by commas
    row = row.split(',')
    N_up = int(row[0])
    N_down = int(row[1])
    E = float(row[2])
    uncert = float(row[3])
    return N_up, N_down, E, uncert

def true(N_up, N_down):
    return (N_up**2 + N_down**2)/2

Ns = []
Es = []
uncerts = []
for row in rows:
    N_up, N_down, E, uncert = parse_row(row)
    N = N_up + N_down
    Ns.append(N)
    Es.append(E)
    uncerts.append(uncert)

plt.title(r"$\sum_i^N \left(- \frac{1}{2m} \frac{\partial^2}{\partial x_i^2} + \frac{1}{2}m \omega^2x_i^2\right)$, $N_\uparrow = N_\downarrow$")
plt.xlabel("$N$")
plt.ylabel(r"Percent Difference in $\langle E_0 \rangle$")
# plot the true energy
Ns = np.array(Ns)
true_energies = true(Ns/2, Ns/2)
# compute the percent difference between the true energy and the energy from the simulation
Es = np.array(Es)
uncerts = np.array(uncerts)
test = gv.gvar(Es, uncerts)
percent_diff = ((test - true_energies)/true_energies)*100
split_e = [gv.mean(i) for i in percent_diff]
split_u = [gv.sdev(i) for i in percent_diff]
plt.errorbar(Ns, split_e, yerr=split_u, label="", fmt='o', color="red")
plt.grid(True)
plt.ylim(-.7,.7)
plt.savefig("ferm_harmonic_nodelta.svg", format="svg")
plt.show()
