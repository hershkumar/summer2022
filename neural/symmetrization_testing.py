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
import pickle


C = 5.0
def P(coords):
    ret = jnp.zeros(3)
    for i in range(3):
        # we use the symmetrization constant C to prevent the sum from getting too large
        ret = ret.at[i].set(jnp.sum(jnp.power(coords/C, i + 1)))
    return ret


def split(input_list):
    if len(input_list) % 3 != 0:
        raise ValueError("The length of the list must be divisible by 3")
    
    return [input_list[i:i+3] for i in range(0, len(input_list), 3)]


def symmetrize(coords):
    # first split the input into 3-vectors
    x1,x2,x3 = split(coords)
    x1_prime = (x1) + (x2) + (x3)
    x2_prime = P(x1) * P(x2) + P(x2) * P(x3) + P(x3) * P(x1)
    x3_prime = P(x1) * ((x2)*(x3)) + P(x2) * ((x3) * (x1)) + P(x3) * ((x1) * (x2))
    return jnp.concatenate([x1_prime, x2_prime, x3_prime])

coords = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
print(symmetrize(coords))
