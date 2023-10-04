# this script creates a neural network object
using Zygote, Flux, Statistics, Plots

num_particles = 2
harmonic_omega = 1.0
omega = 1
g = 0
sigma = -g/2

# make the model 
model = f64(Chain(
	Dense(num_particles, 5, celu),
	Dense(5, 1)
))


function transform(coord)
    C = 2
    return [sum(coord ./ C).^(i+1) for i in eachindex(coord)] 
end


function nn(input)
	return model(transform(input))[1]
end

# define the A function
function A(coords)
	return - nn(coords) + omega * sum(coords.^2)
end

function psi(coords)
	return exp(-A(coords))
end

function psi_sq(coords)
	return psi(coords).^2
end

# derivative of A wrt the coordinates
function dA_dx(coords)
	return gradient(A, coords)[1]
end

# take the gradient of A wrt the parameters of the network, which are not passed in as arguments, but are global
function dA_dparams(coords)
	return gradient(Flux.params(model)) do
		A(coords)
	end
end

