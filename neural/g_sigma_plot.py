import numpy as np
from matplotlib import pyplot as plt


# read in the data
filename = "g_sigma_solvable.csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1)

N = 2
def analytic(g):
    return (N * omega)/2 - m * g**2  * (N*(N**2 - 1))/(24)

# plot the analytic solution from -1 to 1
g = np.linspace(-1, 1, 100)
omega = 1
m = 1
plt.plot(g, analytic(g), label='Analytic', color="black")
# now plot the data using errorbars
plt.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='o', label='Data', color="red")
plt.grid()
plt.title(r"$V(x_1,x_2) = \frac{1}{2}(x_1^2 + x_2^2) + g\delta(x_1 - x_2) + \sigma|x_1-x_2|$")
plt.xlabel(r"$g = -2\sigma$")
plt.ylabel(r"$\langle E_0\rangle$")
plt.savefig("g_sigma_exact.svg", format="svg")
plt.show()