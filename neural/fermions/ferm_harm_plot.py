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
plt.ylabel("Energy")
# plot the energy as a function of N_up
plt.errorbar(Ns, Es, yerr=uncerts, fmt='o', label="Energy", color="red")
# plot the true energy
Ns = np.array(Ns)
true_energies = true(Ns/2, Ns/2)
plt.plot(Ns, true_energies, label='Analytic Energy', color='black')
plt.legend()
plt.grid(True)

plt.show()
