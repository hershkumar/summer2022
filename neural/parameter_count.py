from matplotlib import pyplot as plt 
import numpy as np

parameter_count = []
final_energy = []
true_energy = 0.3098

energies = [(energy - true_energy)/true_energy for energy in final_energy]
params_logged = np.log10(parameter_count)

plt.scatter(params_logged, energies)
plt.show()