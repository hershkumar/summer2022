import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax
from jax import grad, hessian, jit, vmap
from jax.nn import celu
import gvar as gv
from functools import partial
from IPython.display import clear_output
import jax.example_libraries.optimizers as jax_opt
from tqdm import trange
import cProfile
import sys


num_particles = 2
N = num_particles
structure = [50,100,300,100,50]
num_nodes = np.sum(structure)
m = 1
hbar = 1
omega = 1
harmonic_omega = 1
# read in g as an arg
g = float(sys.argv[1])
sigma = 0
C = 4
INITIAL_SAMPLE = jnp.array(np.random.uniform(-2, 2, N))

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        if hidden_sizes != [0]:
            sizes = [input_size] + hidden_sizes + [output_size]
        else:
            sizes = [input_size, output_size]

        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
            b = np.random.randn(1, sizes[i+1]) 
            self.weights.append(w)
            self.biases.append(b)

    @partial(jit, static_argnums=(0,))
    def transform(self, coords):
        ret = jnp.zeros(num_particles)
        for i in range(num_particles):
            ret = ret.at[i].set(jnp.sum(jnp.power(coords/C, i + 1)))
        return ret 

    @partial(jit, static_argnums=(0,))
    def __call__(self, x, params):
        x = self.transform(x)
        self.weights, self.biases = self.unflatten_params(params)
        a = x
        for i in range(len(self.weights) - 1):
            z = jnp.dot(a, self.weights[i]) + self.biases[i]
            a = celu(z)
        a = jnp.dot(a, self.weights[-1]) + self.biases[-1]
        return a[0][0]
    
    @partial(jit, static_argnums=(0,))
    def flatten_params(self):
        params = jnp.array([])
        for i in range(len(self.weights)):
            params = jnp.concatenate((params, self.weights[i].flatten()))
            params = jnp.concatenate((params, self.biases[i].flatten()))
        return jnp.array(params)
    
    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, params):
        weights = []
        biases = []
        start = 0
        for i in range(len(self.weights)):
            end = start + self.weights[i].size
            weights.append(jnp.reshape(jnp.array(params[start:end]), self.weights[i].shape))
            start = end
            end = start + self.biases[i].size
            biases.append(jnp.reshape(jnp.array(params[start:end]), self.biases[i].shape))
            start = end
        return weights, biases
    

# initialize the network
nn = NeuralNetwork(num_particles, structure, 1)

# symmetrization transformation
# I1 = x_1/C + x_2/C + ... + x_N/C
# I2 = (x_1/C)^2 + (x_2/C)^2 + ... + (x_N/C)^2
# ...
# IN = (x_1/C)^N + (x_2/C)^N + ... + (x_N/C)^N

@jit
def A(coords, params):
    return nn(coords, params) + omega * jnp.sum(coords**2)

@jit
def psi(coords, params):
    return jnp.exp(-A(coords, params)) 

# sample_body function except it also returns whether or not the move was accepted
# @jit
# def sample_body_accept(coords_t, params, key, variation_size):
#     gen_rand = jax.random.uniform(key, minval=-variation_size, maxval=variation_size)
#     new_key, subkey = jax.random.split(key)
    
#     coords_prime = coords_t + gen_rand
#     r = jax.random.uniform(subkey, minval=0, maxval=1)
#     condition = r <= psi(coords_prime, params)**2/psi(coords_t, params)**2
#     return (jax.lax.cond(condition, lambda x, _: x, lambda _, y : y, coords_prime, coords_t), new_key, condition)


# the sample function without any thermalization steps or skipping steps
def accept_ratio(params, num_samples=10**3, variation_size=5.0, key=jax.random.PRNGKey(np.random.randint(0,100))):
    coords_t = np.random.uniform(-variation_size, variation_size)
    num_accepted = 0
    for _ in range(num_samples):
        coords_t, key, accepted = sample_body_accept(coords_t, params, key, variation_size)
        if accepted:
            num_accepted += 1

    return num_accepted / num_samples


#### New sampling function
# def sample(params, num_samples=10**3, thermalization_steps=200, skip_count=50, variation_size=1.0):
#     outputs = []
#     num_accepted = 0
#     num_total = num_samples * skip_count + thermalization_steps + 1
#     rand_coords = np.random.uniform(-variation_size, variation_size, size=(num_total, num_particles))
#     rand_accepts = np.random.uniform(0, 1, size=num_total)

#     coords_t = jnp.zeros(num_particles)
#     for step in range(num_total):
#         coords_t, accepted = sample_body(params, coords_t, rand_coords[step], rand_accepts[step])
#         if accepted:
#             num_accepted += 1
#         if ((step > thermalization_steps) and (step % skip_count == 0)):
#             outputs.append(coords_t)
#     # create a second output array, where the second coordinate is equal to the first coordinate
#     outputs_prime = outputs.copy()
#     for i in range(len(outputs)):
#         a = np.array(outputs[i])
#         a[1] = a[0]
#         outputs_prime[i] = jnp.array(a)
#     return jnp.array(outputs), jnp.array(outputs_prime), num_accepted/num_total


@jit
def mcstep_E(xis, limit, positions, params):
    
    params = jax.device_put(params, device=jax.devices("cpu")[0])
    
    newpositions = jnp.array(positions) + xis
    
    prob = psi(newpositions, params)**2./psi(positions, params)**2.
    
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
        a[1] = a[0]
        sq_prime[i] = jnp.array(a) 

    return jnp.array(sq), jnp.array(sq_prime), counter/num_total



@jit
def sample_body(params, coords_t, rand_coords, rand_accepts):
    coords_prime = coords_t + rand_coords
    return jax.lax.cond(rand_accepts < psi(coords_prime, params)**2/psi(coords_t, params)**2, lambda x,_: (x,True) , lambda _,y: (y,False), coords_prime, coords_t)

# first derivative of the neural network with respect to the coordinates
# in Andy's notation this is dA/dx
dA_dx = jit(grad(A, 0)) # type: ignore

# second derivative of the neural network with respect to the coordinates
# in Andy's notation this is d^2A/dx^2
A_hessian = jax.jacfwd(dA_dx, 0) # type: ignore

@jit
def d2A_dx2(coords, params):
    #return jnp.diagonal(A_hessian(transform(coords), params))
    return jnp.diag(A_hessian(coords, params))

@jit
def Hpsi(coords, coords_prime, params, alpha):
    return Hpsi_without_delta(coords, params) + delta_potential(coords,coords_prime, params, alpha)

@jit
def sigma_term(coords):
    N = num_particles 
    sigma_term = 0
    for i in range(N):
        for j in range(i,N):
            sigma_term += sigma* jnp.abs(coords[i] - coords[j])  

@jit
def Hpsi_without_delta(coords, params):
   # sigma term
    N = num_particles 
    sigma_term = 0
    for i in range(N):
        for j in range(i,N):
            sigma_term += sigma* jnp.abs(coords[i] - coords[j]) 
    # return jnp.sum((m*.5*omega**2*coords**2)) - hbar**2 / (2*m) * jnp.sum(ddpsi(coords, params) ) * 1/psi(coords, params) + sigma_term 
    return 1/(2*m) * (jnp.sum(d2A_dx2(coords, params)) - jnp.sum(dA_dx(coords, params)**2)) + m*.5*harmonic_omega**2* jnp.sum(coords**2) + sigma_term
    # return 1/(2*m) * (jnp.sum(d2A_dx2(coords, params)) - jnp.sum(dA_dx(coords, params)**2))
    # return 1/(2*m) * (jnp.sum(d2A_dx2(coords, params)) - jnp.sum(dA_dx(coords, params)**2)) + m*.5*omega**2* jnp.sum(coords**2)

@jit
def second_term(coords, params):
    return dnn_dtheta(coords, params) * Hpsi_without_delta(coords, params)

vsecond_term = jit(vmap(second_term, in_axes=(0, None), out_axes=0))

@jit
def third_term(coords,coords_prime, params, y_max):
    return dnn_dtheta(coords_prime, params) * delta_potential(coords, coords_prime, params, y_max)

vthird_term = jit(vmap(third_term, in_axes=(0,0, None, None), out_axes=0))

@jit
def delta_potential(coords, coords_prime, params, alpha):
    N = num_particles    
    # compute e^(-2 NN(params_prime))
    # ratio = jnp.exp(-2 * A(coords_prime, params) + 2 * A(coords, params))
    ratio = (psi(coords_prime, params)**2)/(psi(coords, params)**2)
    delta_dist = (1/(jnp.sqrt(jnp.pi) * alpha)) * jnp.exp(-(coords[1]**2)/(alpha**2))
    return g * N*(N-1)/2 * ratio * delta_dist

vdelta_potential = jit(vmap(delta_potential, in_axes=(0,0, None, None), out_axes=0))
venergy = jit(vmap(Hpsi, in_axes=(0,0, None, None), out_axes=0))
vHpsi_without_delta = jit(vmap(Hpsi_without_delta, in_axes=(0, None), out_axes=0))


# derivative of the neural network with respect to every parameter
# in Andy's notation this is dA/dtheta
dnn_dtheta = jit(grad(A, 1)) 
vdnn_dtheta = vmap(dnn_dtheta, in_axes=(0, None), out_axes=0)

vboth = vmap(jnp.multiply, in_axes=(0, 0), out_axes=0)

def gradient(params, num_samples=10**3, thermal=200, skip=50, variation_size=1.0, verbose=False):
    # get the samples
    samples, samples_prime, _  = sample(params, num_samples, thermal, skip, variation_size)

    y_max = jnp.max(jnp.abs(jnp.array(samples[:,1])))
    alpha = y_max/(jnp.sqrt(-jnp.log(jnp.sqrt(jnp.pi) * 10**(-10))))

    psiHpsi = venergy(samples, samples_prime, params, alpha) 
    # Hpsi_terms_without_delta = vHpsi_without_delta(samples, params)
    # delta_term = vdelta_potential(samples,samples_prime, params, samples)

    # delta function additions
    dA_dtheta = vdnn_dtheta(samples, params)
    # dA_dtheta_repeated = vdnn_dtheta(samples_prime, params)

    dA_dtheta_avg = 1/num_samples * jnp.sum(dA_dtheta, 0)

    second_term = 1/num_samples * jnp.sum(vsecond_term(samples, params), 0)
    third_term = 1/num_samples * jnp.sum(vthird_term(samples, samples_prime, params, alpha), 0)
    # third_term =1/num_samples * jnp.sum(vboth(dA_dtheta_repeated,delta_term), 0)
    uncert = jnp.std(psiHpsi)/jnp.sqrt(num_samples)

    energy = 1/num_samples * jnp.sum(psiHpsi)

   
    if verbose:
        print(energy)

    gradient_calc = 2 * energy * dA_dtheta_avg - 2 * second_term - 2*third_term
    return gradient_calc, energy, uncert

def ugradient(params, num_samples=10**3, thermal=200, skip=50, variation_size=1.0, verbose=False):

    samples, samples_prime, _ = sample(params, num_samples, thermal, skip, variation_size)
    y_max = jnp.max(jnp.abs(jnp.array(samples[:,1])))
    alpha = y_max/(jnp.sqrt(-jnp.log(jnp.sqrt(jnp.pi) * 10**(-10))))
    Es = []
    dA_dthetas = []
    seconds = []
    thirds = []

    for i in range(len(samples)):
        coord = samples[i]
        coord_prime = samples_prime[i]

        Es.append(Hpsi(coord, coord_prime, params, alpha))
        dA_dthetas.append(dnn_dtheta(coord, params)) 
        seconds.append(second_term(coord, params))
        thirds.append(third_term(coord, coord_prime, params, alpha))


    Es = jnp.array(Es)
    dA_dthetas = jnp.array(dA_dthetas)
    seconds = jnp.array(seconds)
    thirds = jnp.array(thirds)

    energy = 1/num_samples * jnp.sum(Es)
    avg_dA_dtheta = 1/num_samples * jnp.sum(dA_dthetas, 0)
    second = 1/num_samples * jnp.sum(seconds, 0)
    third =  1/num_samples * jnp.sum(thirds, 0)

    uncert = jnp.std(Es)/jnp.sqrt(num_samples)
    
    gradient_calc = 2 * energy * avg_dA_dtheta - 2 * second - 2 * third
    return gradient_calc, energy, uncert


# define a function that takes in samples, bins them, and returns the average of each bin
def bin_samples(energies, bin_size):
    # first, bin the samples
    binned = np.array_split(energies, bin_size)
    # now, calculate the average of each bin
    binned_averages = [np.mean(b) for b in binned]
    # now, calculate the uncertainty of each bin
    bin_uncerts = np.std(binned_averages)/np.sqrt(bin_size)
    return bin_uncerts


# define a function that gets all samples, and then bins them with different bin sizes
def autocorrelation(params):
    samples = sample(params, num_samples=10**3, thermalization_steps=200, skip_count=40, variation_size=1)[0]
    energies = [Hpsi(s, params) for s in samples]
    
    bins = np.linspace(1, 100, 100, dtype=int)
    # now plot the average energy as a function of the number of bins
    us = []
    for b_size in bins:
        us.append(bin_samples(energies, b_size))
    plt.scatter(bins, us)
    plt.title("Bin size vs. Uncertainty")
    plt.xlabel("Bin size")
    plt.ylabel("Uncertainty")
    plt.show()

def step(params_arg, step_num, N, thermal, skip, variation_size):
        gr = gradient(params_arg, N, thermal, skip, variation_size)
        # print(gr)
        # hs.append(gr[1])
        # us.append(gr[2])
        opt_state = opt_init(params_arg)
        new = opt_update(step_num, gr[0], opt_state)
        return get_params(new), gr[1], gr[2]

def train(params, iterations, N, thermal, skip, variation_size):
    hs = []
    us = [] 
    ns = np.arange(iterations) 

    pbar = trange(iterations, desc="", leave=True)

    old_params = params.copy()
    for step_num in pbar:   
        new_params, energy, uncert = step(old_params, step_num, N, thermal, skip, variation_size)
        hs.append(energy)
        us.append(uncert)
        old_params = new_params.copy()
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


# start_params = nn.flatten_params()
start_params = nn.flatten_params()

opt_init, opt_update, get_params = jax_opt.adam(10**(-2))
resultsa = train(start_params, 100, 500, 1000, 10, find_step_size(start_params, 1))

opt_init, opt_update, get_params = jax_opt.adam(10**(-3))
resultsb = train(resultsa[3], 100, 10**3, 500, 10, find_step_size(resultsa[3], 1))

opt_init, opt_update, get_params = jax_opt.adam(10**(-4))
resultsc = train(resultsb[3], 300,2000, 1000, 10, find_step_size(resultsb[3], 1)) 


def astra_energy():
    return (N * omega)/2 - m * g**2  * (N*(N**2 - 1))/(24)



params = resultsc[3]
num_final_samples = 20000
params = jax.device_put(params, device=jax.devices("cpu")[0])
samples, samples_prime, _ = sample(params, num_final_samples, 100, 10, find_step_size(params, 1))

y_max = jnp.max(jnp.abs(jnp.array(samples[:,1])))
alpha = y_max/(jnp.sqrt(-jnp.log(jnp.sqrt(jnp.pi) * 10**(-10))))
es = []
da_dthetas = []
seconds = []
thirds = []

for i in range(len(samples)):
    coord = samples[i]
    coord_prime = samples_prime[i]

    es.append(Hpsi(coord, coord_prime, params, alpha))
    da_dthetas.append(dnn_dtheta(coord, params)) 
    seconds.append(second_term(coord, params))
    thirds.append(third_term(coord, coord_prime, params, alpha))

es = jnp.array(es)
da_dthetas = jnp.array(da_dthetas)
seconds = jnp.array(seconds)
thirds = jnp.array(thirds)

energy = 1/num_final_samples * jnp.sum(es)

true_energy = astra_energy()

def bin_samples(energies, bin_size):
    # first, bin the samples
    binned = np.array_split(energies, bin_size)
    # now, calculate the average of each bin
    binned_averages = [np.mean(b) for b in binned]
    # now, calculate the uncertainty of each bin
    bin_uncerts = np.std(binned_averages) / np.sqrt(bin_size)
    return bin_uncerts


energies = es

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

final = gv.gvar(energy, max(us))
print(final)


filename = "g_sigma_exact/g_" + str(g) + "_sigma_0.txt"
open(filename, "w").close()

# output the results to a file
with open(filename, "w") as f:
    f.write(str(energy) + "\n")
    f.write(str(max(us)) + "\n")
