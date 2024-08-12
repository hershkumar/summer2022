import numpy as np 
from matplotlib import pyplot as plt 
import gvar as gv 


# import the data from precision_vals.csv
data = np.genfromtxt('precision_vals.csv', delimiter=',', skip_header=0)

# x axis is the second column
x = data[:,1]
# y axis is the third column
y = data[:,2]
# error is the fourth column
err = data[:,3]

# plot using errorbars
plt.errorbar(x, y, yerr=err, fmt='o', capsize=2)
plt.xlabel(r'$g$')
plt.ylabel('Ground State Energy')
plt.title('5+5 Ground State Energies')
plt.show()
