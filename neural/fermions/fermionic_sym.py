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



num_particles = 3
structure = [50]
num_nodes = np.sum(structure)
m = 1
hbar = 1
omega = 1
harmonic_omega = 1
g = 0
sigma = 0

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
       # if running into NaNs, try to increase this
        C = 2
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

nns = []
for i in range(num_particles):
    nns.append(NeuralNetwork(1, structure, 1))

# For N particles, why not have N neural networks, and then make the wave function a slater determinant of the outputs of the neural networks? 

def A(coords, params):
    return nn(coords, params) + omega * jnp.sum(coords**2)

def psi(coords, params):
    return jnp.exp(-A(coords, params)) 


# create a test coordinate
coords = np.random.randn(num_particles)

# now write a function that takes in a coordinate, and makes n single swaps of the elements
def swap(coords, n):
    coords = coords.copy()
    for i in range(n):
        a = np.random.randint(num_particles)
        b = a
        while b == a:
            b = np.random.randint(num_particles)
        coords[a], coords[b] = coords[b], coords[a]
    return coords

print(coords)
print(swap(coords, 2))

