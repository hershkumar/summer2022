from matplotlib import pyplot as plt
import numpy as np
import gvar as gv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# read in the ferm_harm_nodelta.csv file as a list of rows
with open('fermi_delta_Nup_1down.csv', 'r') as f:
    rows = f.readlines()

def parse_row(row):
    # split the row by commas
    row = row.split(',')
    N_up = int(row[0])
    E = float(row[1])
    uncert = float(row[2])
    return N_up, E, uncert


def compute_true_energy(N_up):
    N_down = 1
    ret = (N_up**2 + N_down**2)/2 
    if N_up == 1: 
        ret += 0.0854344122657581
    elif N_up == 2:
        ret += 0.12291311754684836
    elif N_up == 3:
        ret += 0.15085178875638838
    elif N_up == 4:
        ret += 0.1753833049403748
    elif N_up == 5:
        ret += 0.1965076660988075
    elif N_up == 6:
        ret += 0.21626916524701872
    elif N_up == 7:
        ret += 0.23330494037478702
    elif N_up == 8:
        ret += 0.2503407155025553
    elif N_up == 9:
        ret += 0.2656729131175468
    return ret


Ns = []
Es = []
uncerts = []
for row in rows:
    N_up, E, uncert = parse_row(row)
    N = N_up
    Ns.append(N)
    Es.append(E)
    uncerts.append(uncert)

plt.title(r"$\sum_i^N \left(- \frac{1}{2m} \frac{\partial^2}{\partial x_i^2} + \frac{1}{2}m \omega^2x_i^2\right) + g\sum_{i<j} \delta(x_i - x_j)$, $N_\downarrow = 1, g = .2$")
plt.xlabel(r"$N_\uparrow$")
plt.ylabel("Percent Difference")
true_energies = np.array([compute_true_energy(i) for i in Ns])

Es = np.array(Es)
uncerts = np.array(uncerts)
test = gv.gvar(Es, uncerts)
percent_diff = ((test - true_energies)/true_energies)*100
split_e = [gv.mean(i) for i in percent_diff]
split_u = [gv.sdev(i) for i in percent_diff]
plt.errorbar(Ns, split_e, yerr=split_u, label="", fmt='o', color="red")
plt.grid(True)
plt.ylim(-5,5)
plt.show()
