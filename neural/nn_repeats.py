# %%
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax
from jax import grad, hessian, jit, vmap
from jax.nn import celu
import time
from functools import partial
from IPython.display import clear_output
import optax
from tqdm import trange


num_particles = 2
num_nodes = 50
m = 1
hbar = 1
omega = 1


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
            w = np.random.randn(sizes[i], sizes[i + 1])
            b = np.random.randn(1, sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    @partial(jit, static_argnums=(0,))
    def __call__(self, x, params):
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
            weights.append(
                jnp.reshape(jnp.array(params[start:end]), self.weights[i].shape)
            )
            start = end
            end = start + self.biases[i].size
            biases.append(
                jnp.reshape(jnp.array(params[start:end]), self.biases[i].shape)
            )
            start = end
        return weights, biases


# initialize the network
nn = NeuralNetwork(num_particles, [num_nodes], 1)


# symmetrization transformation
# I1 = x_1/C + x_2/C + ... + x_N/C
# I2 = (x_1/C)^2 + (x_2/C)^2 + ... + (x_N/C)^2
# ...
# IN = (x_1/C)^N + (x_2/C)^N + ... + (x_N/C)^N


@jit
def transform(coords):
    C = 2
    ret = jnp.zeros(num_particles)
    for i in range(num_particles):
        ret = ret.at[i].set(jnp.sum(jnp.power(coords / C, i + 1)))
    return ret


@jit
def psi(coords, params):
    return jnp.exp(-nn(transform(coords), params)) * jnp.exp(jnp.sum(-(coords**2)))


@jit
def sample_body(coords_t, params, key, variation_size):
    gen_rand = jax.random.uniform(
        key, shape=(num_particles,), minval=-variation_size, maxval=variation_size
    )
    new_key, subkey = jax.random.split(key)
    coords_prime = coords_t + gen_rand
    temp_rand = jax.random.uniform(subkey, minval=0, maxval=1)
    return (
        jax.lax.cond(
            temp_rand < psi(coords_prime, params) ** 2 / psi(coords_t, params) ** 2,
            lambda x, _: x,
            lambda _, y: y,
            coords_prime,
            coords_t,
        ),
        new_key,
    )


def sample(
    params,
    num_samples=10**3,
    thermalization_steps=200,
    skip_count=50,
    variation_size=1.0,
    key=jax.random.PRNGKey(np.random.randint(0, 10000)),
    skipping=True,
):
    outputs = []
    all_data = []

    coords_t = np.random.uniform(-variation_size, variation_size, num_particles)
    for step in range(num_samples * skip_count + thermalization_steps + 1):
        coords_t, key = sample_body(coords_t, params, key, variation_size)
        if (
            (step > thermalization_steps)
            & (step % skip_count == 0)
            & (skipping == True)
        ):
            outputs.append(coords_t)
        elif skipping == False:
            all_data.append(coords_t)
    if skipping == False:
        return all_data
    return jnp.array(outputs)


# sample_body function except it also returns whether or not the move was accepted
@jit
def sample_body_accept(coords_t, params, key, variation_size):
    gen_rand = jax.random.uniform(key, minval=-variation_size, maxval=variation_size)
    new_key, subkey = jax.random.split(key)

    coords_prime = coords_t + gen_rand
    condition = (
        jax.random.uniform(subkey, minval=0, maxval=1)
        < psi(coords_prime, params) ** 2 / psi(coords_t, params) ** 2
    )
    return (
        jax.lax.cond(condition, lambda x, _: x, lambda _, y: y, coords_prime, coords_t),
        new_key,
        condition,
    )


# the sample function without any thermalization steps or skipping steps
def accept_ratio(
    params,
    num_samples=10**3,
    variation_size=5.0,
    key=jax.random.PRNGKey(np.random.randint(0, 100)),
):
    coords_t = np.random.uniform(-variation_size, variation_size)
    num_accepted = 0
    for _ in range(num_samples):
        coords_t, key, accepted = sample_body_accept(
            coords_t, params, key, variation_size
        )
        if accepted:
            num_accepted += 1

    return num_accepted / num_samples


hessian = hessian(psi)
ddpsi_single = jit(grad(jit(grad(psi, 0, allow_int=True)), 0, allow_int=True))


@jit
def ddpsi(coords, params):
    if num_particles != 1:
        return jnp.diagonal(hessian(coords, params))
    else:
        return ddpsi_single(coords, params)


@jit
def Hpsi(coords, params):
    return jnp.sum((m * 0.5 * omega**2 * coords**2)) - hbar**2 / (
        2 * m
    ) * jnp.sum(ddpsi(coords, params)) * 1 / psi(coords, params)


venergy = jit(vmap(Hpsi, in_axes=(0, None), out_axes=0))


@jit
def logpsi(coords, params):
    return jnp.log(psi(coords, params))


# define the derivative with respect to every parameter of the log of psi:
dlogpsi_dtheta_stored = jit(grad(logpsi, 1))

vlog_term = jit(vmap(dlogpsi_dtheta_stored, in_axes=(0, None), out_axes=0))

vboth = vmap(jnp.multiply, in_axes=(0, 0), out_axes=0)


def gradient(samples, params, thermal=200, skip=50, variation_size=1.0, verbose=False):
    # get the number of samples
    num_samples = len(samples)

    psiHpsi = venergy(samples, params)
    logs = vlog_term(samples, params)

    uncert = jnp.std(psiHpsi) / jnp.sqrt(num_samples)

    energy = 1 / num_samples * jnp.sum(psiHpsi)
    if verbose:
        print(energy)
    log_term = 1 / num_samples * jnp.sum(logs, 0)

    both = 1 / num_samples * jnp.sum(vboth(psiHpsi, logs), 0)

    gradient_calc = 2 * (both - energy * log_term)
    return gradient_calc, energy, uncert


# define a function that takes in samples, bins them, and returns the average of each bin
def bin_samples(energies, bin_size):
    # first, bin the samples
    binned = np.array_split(energies, bin_size)
    # now, calculate the average of each bin
    binned_averages = [np.mean(b) for b in binned]
    # now, calculate the uncertainty of each bin
    bin_uncerts = np.std(binned_averages) / np.sqrt(bin_size)
    return bin_uncerts


# define a function that gets all samples, and then bins them with different bin sizes
def autocorrelation(params):
    samples = sample(
        params,
        num_samples=10**3,
        thermalization_steps=200,
        skip_count=40,
        variation_size=1,
        key=jax.random.PRNGKey(np.random.randint(0, 100)),
        skipping=False,
    )
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


def train(
    params, iterations, N, thermal, skip, variation_size, optimizer, num_repeats=100
):
    hs = []
    us = []
    opt_state = optimizer.init(params)

    def step(params, opt_state, N, thermal, skip, variation_size, num_repeats):
        # get the samples
        samples = sample(params, N, thermal, skip, variation_size)
        for _ in range(num_repeats):
            gr = gradient(samples, params, thermal, skip, variation_size)
            # print(gr)
            hs.append(gr[1])
            us.append(gr[2])
            updates, opt_state = optimizer.update(gr[0], opt_state, params)
            params = optax.apply_updates(params, updates)
        gr = gradient(samples, params, thermal, skip, variation_size)
        return params, opt_state, gr[1]

    pbar = trange(iterations, desc="", leave=True)
    for step_num in pbar:
        params, opt_state, energy = step(
            params, opt_state, N, thermal, skip, variation_size, num_repeats
        )
        pbar.set_description("Energy = " + str(energy), refresh=True)
        if np.isnan(energy):
            print("NaN encountered, stopping...")
            break
    clear_output(wait=True)
    return hs, us, params


# %%
variation = 1
print(
    "Accept/Reject ratio",
    accept_ratio(nn.flatten_params(), num_samples=10**3, variation_size=variation),
)

# %%
total_hists = []
total_uncerts = []
start_params = nn.flatten_params()

hists, uncerts, params = train(
    start_params, 100, 10**3, 200, 40, variation, optax.adam(0.01)
)
total_hists = total_hists + list(hists)
total_uncerts = total_uncerts + list(uncerts)

# %%
true_energy = 0.5 * hbar * omega * num_particles
plt.plot(
    np.arange(0, len(total_hists)),
    total_hists,
    label="Adam, "
    + str(round(total_hists[-1], 3))
    + " +/- "
    + str(round(total_uncerts[-1], 2)),
)
# plot the uncertainties
a_hists = np.array(total_hists)
a_uncerts = np.array(total_uncerts)
plt.fill_between(
    np.arange(0, len(total_hists)), a_hists - a_uncerts, a_hists + a_uncerts, alpha=0.4
)
# get the network structure
structure = nn.hidden_sizes
plt.annotate(
    " Network Structure: " + str(structure), xy=(0.1, 0.95), xycoords="axes fraction"
)
plt.plot(
    np.arange(0, len(total_hists)),
    [true_energy for x in np.arange(0, len(total_hists))],
    label=r"True Energy, " + str(round(true_energy, 3)),
)
pdiff = (total_hists[-1] - true_energy) / true_energy * 100
plt.annotate(
    " Final Percent Diff = " + str(round(pdiff, 3)) + "%",
    xy=(0.1, 0.9),
    xycoords="axes fraction",
)
plt.legend()
plt.title("Average Energy vs. Iteration, " + str(num_particles) + " particles")
plt.show()
