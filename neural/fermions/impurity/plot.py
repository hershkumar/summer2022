import numpy as np 
from matplotlib import pyplot as plt 

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.dpi': 300})

def dE3(N, g1D):
    ap = 1
    a1D = -2/g1D

    yt = -np.sqrt(2.0/N)*np.pi*ap/a1D
    return ((N*yt)/(np.pi**2))*(1 - yt/4 + (yt/(2*np.pi) + (2*np.pi)/yt)*np.arctan(yt/(2*np.pi)))
   

#g = 0.2
#N up 1 down
Ns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,])
results = np.array([1.0763857, 2.6173434, 5.1460529, 8.668864, 13.19061494, 18.70996737, 25.23027,
                   32.74728, 41.2597, 50.777434505])
reserrs = [0.000847, 0.001007, 0.00187, 0.00388, 0.00511, 0.00613, 0.00541,
          0.0066, 0.0076, 0.00885,] 

#g = 2.0
#N up 1 down
Ns_st = np.array([1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, ])
results_st = np.array([1.488919431, 3.3311, 6.10185196, 9.844496, 14.5487,
                       20.226, 26.912664, 34.56552, 43.2136, 52.8536])
reserrs_st = [0.00219, 0.00444, 0.00695, 0.010737, 0.01368,
              0.0129, 0.015269, 0.02034, 0.01816, 0.0238]


def nonint(N):
    return (N**2 + 1)/2
nonints = np.array([nonint(n) for n in Ns])

Ns_cont = np.arange(0, 11, .001)
mcg = np.array([dE3(n, .2) for n in Ns_cont])
mcg_st = np.array([dE3(n, 2.0) for n in Ns_cont])

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(12, 15)

axs[0].set_xticks(np.arange(0, max(Ns)+1, 1.0))
axs[1].set_xticks(np.arange(0, max(Ns)+1, 1.0))

axs[0].errorbar(Ns, results - nonints, yerr=reserrs, fmt='o', color="blue", capsize=4)

axs[1].errorbar(Ns, results_st - nonints, yerr=reserrs_st, fmt='o', color="blue", capsize=4)
axs[0].plot(Ns_cont, mcg, color="red")
axs[1].plot(Ns_cont, mcg_st, color="red")


axs[0].set_title(r"Impurity: $g = 0.2$")
axs[1].set_title(r"Impurity: $g = 2.0$")
axs[0].set_xlim(0,10.5)
axs[1].set_xlim(0,10.5)
axs[0].set_ylim(0,.30)
axs[1].set_ylim(0,2.5)
axs[0].set_ylabel(r"$\Delta E = E_0 - E_{nonint}$")
axs[1].set_ylabel(r"$\Delta E = E_0 - E_{nonint}$")
axs[1].set_xlabel(r"$N_\uparrow$")
plt.tight_layout()
#plt.show()
# annotate in the top left of each plot 
axs[0].annotate("(a)",(11,1010), xycoords='figure points')
axs[1].annotate("(b)",(11,500), xycoords='figure points')

plt.savefig("impurity.pdf", bbox_inches='tight')
