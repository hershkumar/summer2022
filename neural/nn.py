import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax
from jax import grad, hessian, jit, vmap
from jax.nn import celu, relu
import time
from functools import partial
from IPython.display import clear_output
import optax
from tqdm import trange


num_particles = 1
m = 1
hbar = 1
omega = 2


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1])
            b = jnp.zeros((1, sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    @partial(jit, static_argnums=(0,))
    def __call__(self, x, params):
        self.weights, self.biases = self.unflatten_params(params)
        a = x
        for i in range(len(self.weights)):
            z = jnp.dot(a, self.weights[i]) + self.biases[i]
            a = celu(z)
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
nn = NeuralNetwork(1, [50], 1)

params = nn.flatten_params()
print(nn(np.array([3.4]), params))
# make a numpy array of length 151
x = np.linspace(-5, 5, len(params))

print(nn(np.array([3.4]), x))
