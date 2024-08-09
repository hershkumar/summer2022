import numpy as np 
from matplotlib import pyplot as plt 
import time
import sys

# take in a filename as an argument
filename = str(sys.argv[1])
# read in the csv
data = np.genfromtxt(filename, delimiter=',')
energies = data[:,0]
uncerts = data[:,1]

def read(filename):
    data = np.genfromtxt(filename, delimiter=',')
    energies = data[:,0]
    uncerts = data[:,1]
    return energies, uncerts

# set up a blank pyplot window
plt.figure()
plt.show(block=False)

while(True):
    es, us = read(filename)
    x = [i for i in range(len(es))]
    # Clear the previous plot
    plt.clf()

    # Plot the energies and uncertainties
    plt.errorbar(x,es, yerr=us, label='Energies')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Update the plot
    plt.pause(0.01)
    time.sleep(5)