import numpy as np 
from matplotlib import pyplot as plt

# read in the data
data = np.loadtxt('1+1.csv')





plt.xlabel("Number of Parameters")
plt.ylabel("Energy")

plt.legend()
plt.grid()
plt.show()
