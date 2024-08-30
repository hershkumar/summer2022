# N = N_up + N_down fermions in a harmonic trap, with delta function interaction

import csv
import os
import sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["JAX_ENABLE_X64"]="True"

import numpy as np
import time
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax
from jax import grad, hessian, jit, vmap
from jax.nn import celu
import gvar as gv
from functools import partial
from IPython.display import clear_output
import jax.example_libraries.optimizers as jax_opt
from tqdm import tqdm, trange
from math import factorial
import pickle

# set the default device to the cpu
jax.default_device(jax.devices("cpu")[0])
jax.config.update('jax_platform_name', 'cpu')

#use pickle to save the parameters to a file 
def save_params(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
# use pickle to dump the energies and uncertainties to a file
def save_energies(hs, us, filename):
    with open(filename, 'wb') as f:
        pickle.dump((hs, us), f)

# use pickle to load the parameters from a file
def load_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)    
# use pickle to load the energies and uncertainties from a file
def load_energies(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# using this data:
# 1, 0.0854344122657581
# 2, 0.12291311754684836
# 3, 0.15085178875638838
# 4, 0.1753833049403748
# 5, 0.1965076660988075
# 6, 0.21626916524701872
# 7, 0.23330494037478702
# 8, 0.2503407155025553
# 9, 0.2656729131175468

def compute_true_energy():
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


##### Constants
N_up = 5
N_down = 5
N = N_up + N_down

FACT_UP = 2 #increase this when N goes up
FACT_DOWN = 2 # increase this when N goes up
SYM_DEN = 5
GPU_INDEX = 1
# division factor in the ansatz
DIV = 2
INITIAL_SAMPLE = jnp.array(np.random.uniform(-2, 2, N))
phi_structure = [150,150]

m = 1
hbar = 1
omega = 1
harmonic_omega = 1
g = 0

# PARAM_PREFIX = "5+5/large_g_150_params_g_"
# directory = "/Users/hersh/Desktop/5+5 data/"
d = "./"

#######

# this just gets the shapes of the weights and biases for a neural network with the given structure
def gen_weight_shapes(input_size, hidden_sizes, output_size):
    weights = []
    biases = []

    if hidden_sizes != [0]:
        sizes = [input_size] + hidden_sizes + [output_size]
    else:
        sizes = [input_size, output_size]
    for i in range(len(sizes) - 1):
        w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
        b = np.random.randn(1, sizes[i+1]) 
        weights.append(w)
        biases.append(b) 
    return weights, biases

# get the shapes
weight_shapes, bias_shapes = gen_weight_shapes(N, phi_structure, 1)

# generates a set of weights and biases for a neural network with the given structure
# returns a flattened array of the parameters

def gen_params(input_size, hidden_sizes, output_size):
    weights = []
    biases = []

    if hidden_sizes != [0]:
        sizes = [input_size] + hidden_sizes + [output_size]
    else:
        sizes = [input_size, output_size]
    for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
            b = np.random.randn(1, sizes[i+1]) 
            weights.append(w)
            biases.append(b)
    return flatten_params(weights, biases) 

# calls the neural network with the given parameters and input
@jit
def nn(x, params):
    weights, biases = unflatten_params(params) 
    a = x
    for i in range(len(weights) - 1):
        z = jnp.dot(a, weights[i]) + biases[i]
        a = celu(z)
    a = jnp.dot(a, weights[-1]) + biases[-1]
    return a[0][0] 

# takes the weights and biases of a network and returns a flattened array of the parameters
@jit
def flatten_params(weights, biases):
    params = jnp.array([])
    for i in range(len(weights)):
        params = jnp.concatenate((params, weights[i].flatten()))
        params = jnp.concatenate((params, biases[i].flatten()))
    return jnp.array(params)

# takes a flattened array of parameters and returns the weights and biases of the network
@jit
def unflatten_params(params):
    weights = []
    biases = []
    start = 0
    for i in range(len(weight_shapes)):
        end = start + weight_shapes[i].size 
        weights.append(jnp.reshape(jnp.array(params[start:end]), weight_shapes[i].shape))
        start = end
        end = start + bias_shapes[i].size
        biases.append(jnp.reshape(jnp.array(params[start:end]), bias_shapes[i].shape))
        start = end
    return weights, biases



network = gen_params(N, phi_structure, 1)
# the length of the flattened parameters of a single particle neural network
phi_params_length = len(network)

# function that takes the coords, and moves coords[index] to the front of the list
@partial(jit, static_argnums=(1,))
def shift_coords(coords, index):
    return jnp.concatenate([jnp.array([coords[index]]), jnp.array(coords[:index]), jnp.array(coords[index + 1:])])

@partial(jit, static_argnums=(1,))
def inputs_up(coords, j):
    reordered = shift_coords(coords, j)
    sym_piece1 = reordered[1:N_up]
    sym_piece2 = reordered[N_up:]

    new1 = []
    new2 = []
    for i in range(1, N_up):
        new1.append(sum((jnp.array(sym_piece1)/SYM_DEN)**i))
    for i in range(1, N_down+1):
        new2.append(sum((jnp.array(sym_piece2)/SYM_DEN)**i))
    
    return jnp.array([reordered[0]] + new1 + new2)

@partial(jit, static_argnums=(1,))
def inputs_down(coords, j):
    reordered = shift_coords(coords, j + N_up)
    
    sym_piece1 = reordered[1:N_up+1]
    sym_piece2 = reordered[N_up + 1:]
    
    new1 = []
    new2 = []
    for i in range(1, N_up+1):
        new1.append(sum((jnp.array(sym_piece1)/SYM_DEN)**i))
    for i in range(1, N_down):
        new2.append(sum((jnp.array(sym_piece2)/SYM_DEN)**i))
        
    return jnp.array([reordered[0]] + new1 + new2)

@jit
def Phi_up(coords, params):
    # construct the matrix of outputs of the neural networks
    # take only the up spin coordinates
    mat = jnp.zeros((N_up, N_up))
    for i in range(N_up):
        ith_params = params[i * phi_params_length : (i + 1) * phi_params_length]
        for j in range(N_up): 
            mat = mat.at[i,j].set(nn(inputs_up(coords, j), ith_params))
    return jnp.linalg.det(mat) * FACT_UP 

@jit
def Phi_down(coords, params):
    # construct the matrix of outputs of the neural networks
    # take only the up spin coordinates
    mat = jnp.zeros((N_down, N_down))
    for i in range(N_down):
        temp = i + N_up
        ith_params = params[temp * phi_params_length : (temp + 1) * phi_params_length]
        for j in range(N_down): 
            mat = mat.at[i,j].set(nn(inputs_down(coords, j), ith_params))
    return jnp.linalg.det(mat)* FACT_DOWN

@jit
def psi(coords, params):
    return Phi_up(coords, params) * Phi_down(coords, params) * jnp.exp(-omega * jnp.sum((coords/DIV)**2))


@jit
def mcstep_E(xis, limit, positions, params):
    
    params = jax.device_put(params, device=jax.devices("cpu")[0])
    
    newpositions = jnp.array(positions) + xis
    
    # prob = psi(newpositions, params)**2./psi(positions, params)**2.
    prob = (psi(newpositions, params)/psi(positions, params))**2.
    
    def truefunc(p):
        return [newpositions, True]

    def falsefunc(p):
        return [positions, False]
    
    return jax.lax.cond(prob >= limit, truefunc, falsefunc, prob)

def sample(params, Nsweeps, Ntherm, keep, stepsize, positions_initial=INITIAL_SAMPLE, progress=False):
    sq = []
    sq_prime = []
    counter = 0
    num_total = Nsweeps * keep + Ntherm + 1 
    params = jax.device_put(params, device=jax.devices("cpu")[0])

    randoms = np.random.uniform(-stepsize, stepsize, size = (num_total, N))
    limits = np.random.uniform(0, 1, size = num_total)

    positions_prev = positions_initial
    
    if progress:
        for i in tqdm(range(0, num_total), position = 0, leave = True, desc = "MC"):
            
            new, moved = mcstep_E(randoms[i], limits[i], positions_prev, params)
        
            if moved == True:
                counter += 1
                
            if i%keep == 0 and i >= Ntherm:
                #sq = np.vstack((sq, np.array(new)))
                sq.append(new)
                
            positions_prev = new
                
    else: 
        for i in range(num_total):
            new, moved = mcstep_E(randoms[i], limits[i], positions_prev, params)
        
            if moved == True:
                counter += 1
                
            if i%keep == 0 and i >= Ntherm:
                #sq = np.vstack((sq, np.array(new)))
                sq.append(new)
                
            positions_prev = new
    # generate the primed samples by going through every sample and making sample[N_up] = sample[0]
    sq_prime = sq.copy()
    for i in range(len(sq)):
        a = np.array(sq[i])
        a[N_up] = a[0]
        sq_prime[i] = jnp.array(a) 

    return jnp.array(sq), jnp.array(sq_prime), counter/num_total


psi_hessian = jax.jacfwd(jit(grad(psi, 0)), 0) # type: ignore

@jit
def ddpsi(coords, params):
    #return jnp.diagonal(A_hessian(transform(coords), params))
    return jnp.diag(psi_hessian(coords, params))


# derivative of the wavefunction with respect to the parameters
dnn_dtheta = jit(grad(psi, 1)) 
vdnn_dtheta = jit(vmap(dnn_dtheta, in_axes=(0, None), out_axes=0))


@jit
def Es_nodelta(coords, params):
    return - (1/2) * (1/ psi(coords, params)) * jnp.sum(ddpsi(coords, params)) + (1/2) * jnp.sum(coords**2) 

vEs_nodelta = jit(vmap(Es_nodelta, in_axes=(0,None), out_axes=0))

@jit
def Es_delta(coords, coords_prime, params, alpha, g):
    return N_up * N_down * g * (psi(coords_prime, params)**2)/(psi(coords, params)**2) * (1/(np.sqrt(np.pi)*alpha))*np.e**(-(coords[N_up]/alpha)**2)

vEs_delta = jit(vmap(Es_delta, in_axes=(0,0, None, None, None), out_axes=0))

@jit
def gradient_comp(coords, coords_prime, params, es_nodelta, energy_calc, es_delta):
    return 2/(psi(coords,params)) * dnn_dtheta(coords, params) * (es_nodelta - energy_calc) + 2/(psi(coords_prime, params)) * dnn_dtheta(coords_prime, params) * es_delta

vgradient_comp = jit(vmap(gradient_comp, in_axes=(0,0,None,0, None, 0), out_axes=0))

def accumulator_sample(params, Nsweeps, Ntherm, keep, stepsize, g, positions_initial=INITIAL_SAMPLE, progress=True):
    num_total = Nsweeps * keep + Ntherm + 1
    params = jax.device_put(params, device=jax.devices("cpu")[0])

    randoms = np.random.uniform(-stepsize, stepsize, size=(num_total, N))
    limits = np.random.uniform(0, 1, size=num_total)

    accept_counter = 0
    es = 0
    grads = 0
    mean = 0
    m2 = 0
    alpha = 1

    positions_prev = positions_initial

    for i in range(num_total):
        new, moved = mcstep_E(randoms[i], limits[i], positions_prev, params)
        
        if i >= Ntherm and i % keep == 0:
            accept_counter += 1
            new_prime = np.copy(new)
            new_prime[N_up] = new_prime[0]

            temp_nodeltas = Es_nodelta(new, params)
            temp_deltas = Es_delta(new, new_prime, params, alpha, g)
            temp_sum = temp_nodeltas + temp_deltas

            es += temp_sum
            curr_e_avg = es / accept_counter
            grads += gradient_comp(new, new_prime, params, temp_nodeltas, curr_e_avg, temp_deltas)

            temp = temp_sum - mean
            mean += temp / accept_counter
            m2 += temp * (temp_sum - mean)

        positions_prev = new

    stddev = np.sqrt(m2 / (accept_counter - 1)) / jnp.sqrt(accept_counter)
    return es, grads, stddev

def accumulator_gradient(params, g, num_samples=10**3, thermal=200, skip=50, variation_size=1.0):
    # sample
    es, grads, uncert = accumulator_sample(params, num_samples, thermal, skip, variation_size, g)
    energy_calc = es/num_samples
    gradient_calc = grads/num_samples
    return gradient_calc, energy_calc, uncert

def gradient(params, g, num_samples=10**3, thermal=200, skip=50, variation_size=1.0):
    # first sample
    params = jax.device_put(params, device=jax.devices("cpu")[0])
    
    samples, samples_prime, _ = sample(params, num_samples, thermal, skip, variation_size)
    ys = jnp.array(samples_prime[:, N_up]) 
    alpha = np.sqrt(np.max(abs(np.array(ys)))**2/(-np.log(np.sqrt(np.pi)*(10**-10))))

    e_nodeltas = vEs_nodelta(samples, params)
    e_deltas = vEs_delta(samples, samples_prime, params, alpha, g)

    e_term = e_nodeltas + e_deltas
    energy_calc = jnp.mean(e_term)

    # compute the uncertainty in the energy
    uncert = jnp.std(e_term)/jnp.sqrt(num_samples) 
    # gradient computation
    grads = vgradient_comp(samples, samples_prime, params, e_nodeltas, energy_calc, e_deltas)
    gradient_calc = jnp.mean(grads, axis=0) 
    return gradient_calc, energy_calc, uncert


def step(params_arg, step_num, N, thermal, skip, variation_size, g):
        gr = gradient(params_arg, g, N, thermal, skip, variation_size)
        # print(gr)
        # hs.append(gr[1])
        # us.append(gr[2])
        opt_state = opt_init(params_arg)
        new = opt_update(step_num, gr[0], opt_state)
        return get_params(new), gr[1], gr[2]

def train(params, iterations, N, thermal, skip, variation_size, g):
    hs = []
    us = [] 
    ns = np.arange(iterations) 

    pbar = trange(iterations, desc="", leave=True)

    old_params = params.copy()
    for step_num in pbar:   
        new_params, energy, uncert = step(old_params, step_num, N, thermal, skip, variation_size, g)
        hs.append(energy)
        us.append(uncert)
        old_params = new_params.copy()
        # save the energies and uncertainties to a file
        save_energies(hs, us, "energies.pkl")
        pbar.set_description("Energy = " + str(energy), refresh=True)
        if np.isnan(energy):
            print("NaN encountered, stopping...")
            break
    clear_output(wait=True)
    return hs, us, ns, old_params



def find_step_size(params, start):
    lr = .1
    target = 0.5
    tolerance = .05
    max_it = 1000
    step = start
    best_step = start
    best_acc = 0
    it_num = 0
    # get the samples 
    _, _, acc = sample(params, 1000, 100, 10, step)
    # while the acceptance rate is not within +/- .5 of the target
    while (acc < target - tolerance or acc > target + tolerance) and it_num < max_it:
        it_num += 1
        # if the acceptance rate is too low, increase the step size
        if acc < target - tolerance:
            step -= lr
        # if the acceptance rate is too high, decrease the step size
        elif acc > target + tolerance:
            step += lr
        # if we cross the target, decrease the learning rate and go back
        if (acc < target and best_acc > target) or (acc > target and best_acc < target):
            lr /= 2
            step = best_step
        # keep track of the best step size
        if abs(acc - target) < abs(best_acc - target):
            best_acc = acc
            best_step = step
        
        # get the samples for the next step size
        _, _, acc = sample(params, 1000, 100, 10, step)
    return best_step


# make N sets of parameters
params = gen_params(N, phi_structure, 1)
for i in range(N - 1):
    params = jnp.concatenate((params, gen_params(N, phi_structure, 1)))
num_parameters = len(params)
# print(num_parameters)

def bin_samples(energies, bin_size):
    # first, bin the samples
    binned = np.array_split(energies, bin_size)
    # now, calculate the average of each bin
    binned_averages = [np.mean(b) for b in binned]
    # now, calculate the uncertainty of each bin
    bin_uncerts = np.std(binned_averages) / np.sqrt(bin_size)
    return bin_uncerts


def precise_val(params, g):
    num_final_samples = 500000
    samples, samples_prime, _ = sample(params, num_final_samples, 1000, 10, find_step_size(params,1))

    ys = jnp.array(samples_prime[:, N_up]) 
    alpha = np.sqrt(np.max(abs(np.array(ys)))**2/(-np.log(np.sqrt(np.pi)*(10**-10))))
    e_term = vEs_nodelta(samples, params) + vEs_delta(samples, samples_prime, params, alpha, g)
    mean_energy = 1 / num_final_samples * jnp.sum(e_term)

    energies = e_term

    # bins = np.linspace(1, 100, 100, dtype=int)
    bins = np.array(
        [
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            150,
            200,
            250,
            300,
            360,
            450,
            500,
            550,
            600,
            660,
            750,
            900,
            990,
            1100,
        ]
    )
    # now plot the average energy as a function of the number of bins
    us = []
    for b_size in bins:
        us.append(bin_samples(energies, b_size))
    # plt.scatter(bins, us)
    # plt.title("Bin size vs. Uncertainty")
    # plt.xlabel("Bin size")
    # plt.ylabel("Uncertainty")
    
    # plt.show()

    # final = gv.gvar(mean_energy, max(us))
    # print(final)
    return mean_energy, max(us)

def precise_val_acc(params, g):
    measurements = []
    # find the stepsize first
    step_size = find_step_size(params, .3)
    for i in trange(51):
        temp = accumulator_gradient(params, g, 1000000, 20000, 10, step_size)
        measurements.append(gv.gvar(temp[1], temp[2]))
    ret = np.mean(measurements)
    return gv.mean(ret), gv.sdev(ret)

print("Loading parameters...")
# now the actual important part
# load in the parameters from the filename passed as an argument
params = load_params(sys.argv[1])
g = float(sys.argv[2])


print("Computing precise value...")
# get the precise value
final = precise_val_acc(params, g)
print("Done!")
# print the final value
print(final)

rows = []
rows.append([num_parameters, g, final[0], final[1]])

# write the rows to the csv file
with open(sys.argv[3], "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
