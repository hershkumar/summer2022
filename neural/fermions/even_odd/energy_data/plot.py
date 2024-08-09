from matplotlib import pyplot as plt
import numpy as np
import gvar as gv 

# load the data from "4+4_precise_energies.csv"
four_four_data = np.loadtxt("4+4_precise_energies.csv", delimiter=",")
four_five_data = np.loadtxt("4+5_precise_energies.csv", delimiter=",")
five_five_data = np.loadtxt("5+5_precise_energies.csv", delimiter=",")


# x axis has 4 points, 0, -.5, -1, -1.5
x = np.array([0, -.5, -1, -1.5])
# y axis will have the energy values with the uncertainties
# for the four four data, the energies are in the third column, and the uncertainties are in the fourth column
y_four_four = four_four_data[:, 2]
# for the four five data, the energies are in the second column, and the uncertainties are in the third column
y_four_five = four_five_data[:, 1]
# for the five five data, the energies are in the third column, and the uncertainties are in the fourth column
y_five_five = five_five_data[:, 2]

# now plot the data using error bars
plt.errorbar(x, y_four_four, yerr=four_four_data[:, 3], label="4+4", fmt='-o', capsize=4)
plt.errorbar(x, y_four_five, yerr=four_five_data[:, 2], label="4+5", fmt='-o', capsize=4)
plt.errorbar(x, y_five_five, yerr=five_five_data[:, 3], label="5+5", fmt='-o', capsize=4)
plt.legend()
plt.grid()
plt.xlabel(r"$g$")
plt.ylabel(r"$E$")
plt.title("Energy vs. $g$, Attractive Fermions")
plt.show()

# make each data point a gvar object
y_four_four_gvar = gv.gvar(y_four_four, four_four_data[:, 3])
y_four_five_gvar = gv.gvar(y_four_five, four_five_data[:, 2])
y_five_five_gvar = gv.gvar(y_five_five, five_five_data[:, 3])

# print the gvar objects
print(y_four_four_gvar)
print(y_four_five_gvar)
print(y_five_five_gvar)