using Zygote, Flux, Statistics, Plots, Distributions, LinearAlgebra, ProgressBars, IJulia
using Flux.Optimise: update!

num_particles = 2
harmonic_omega = 1.0
omega = 1
g = 0
sigma = -g/2
C = 2

# make the model
model = f64(Chain(
	Dense(num_particles => 50, bias=true, celu; init=Flux.ones32),
    Dense(50 => 1, bias=true; init=Flux.ones32)
))


transform(coord) = [sum(coord ./ C).^(i+1) for i in eachindex(coord)] 
A(coords) = model(transform(coords))[1] + omega * sum(coords.^2)
psi(coords) = exp(-A(coords))
psi_sq(coords) = psi(coords).^2

# derivative of A wrt the coordinates
dA_dx(coords) = gradient(A,coords)[1]

# take the gradient of A wrt the parameters of the network, which are not passed in as arguments, but are global
#TODO: theres something wrong with this gradient, it gives the wrong answer for the biases
dA_dtheta(coords) = gradient(() -> A(coords), Flux.params(model))



# second derivative of A wrt the coordinates
d2A_dx2(coords) = diaghessian(A, coords)[1]

function sample(distribution,  num_particles, num_samples, num_thermalization, num_skip, variation)
    # start by computing the total number of steps that we will take
    total_steps = num_samples * num_skip + num_thermalization
    # number of accepted steps
    num_accepted = 0
    # create a vector of outputs of length num_samples
	samples = Array{Float64}(undef, 0, num_particles)

    # pick a starting point
    x = zeros(num_particles) 
    for step in 1:total_steps 
        # propose a new point
        rand_shift = rand(Uniform(-variation, variation), num_particles)

		x_prime = x + rand_shift
        # generate a random number between 0 and 1
		r = rand(Uniform(0,1))
        if r < distribution(x_prime)/distribution(x)
            # accept the new point
            x = x_prime
            num_accepted += 1
        end

        # now check whether it is time to record a sample
        if step > num_thermalization && mod(step, num_skip) == 0
            # record the sample
            samples = [samples; x']
        end
    end

    # for every sample in samples, replace the second value with a copy of the first value
    samples_prime = copy(samples)
    for i in 1:num_samples
        samples_prime[i, 2] = samples[i, 1]
    end

    return samples, samples_prime, (num_accepted/total_steps) 
end

# define the energy computation
function Hpsi(coords, coords_prime, alpha)
    return Hpwd(coords)  .+ delta_pot(coords, coords_prime, alpha) 
end

function sigma_term(coords)
    sigma_term = 0
    for i in 1:num_particles
        for j in i:num_particles
            sigma_term += sigma * abs(coords[i]-coords[j])
        end
    end
    return sigma_term
end

function Hpwd(coords)
    return 1/2 .* (sum(d2A_dx2(coords)) .- sum(dA_dx(coords).^2)) + 1/2 .* harmonic_omega^2 .* sum(coords.^2) .+ sigma_term(coords)
end

function delta_pot(coords, coords_prime, alpha)
    ratio = psi_sq(coords_prime)/psi_sq(coords)
    delta_dist = (1/(sqrt(pi) * alpha)) * exp(-(coords[1]^2)/(alpha^2))
    return g * num_particles*(num_particles-1)/2 * ratio * delta_dist
end


function second_term(coords)
    return dA_dtheta(coords) .* Hpwd(coords)
end

function third_term(coords, coords_prime, alpha)
    return dA_dtheta(coords_prime) .* delta_pot(coords, coords_prime, alpha)
end

function gradsum(grads)
    sum = grads[1]
    for g in grads[2:end]
        sum = sum .+ g
    end
    return sum
end

function grad(num_samples=10^3, thermal=1000, skip=200, variation=1.0)
    # get the samples
    samples, samples_prime, _ = sample(psi_sq, num_particles, num_samples, thermal, skip, variation)

    # get the maximum of the x coordinate in the samples
    y_max = maximum(abs.(samples[:, 2]))
    alpha = y_max/(sqrt(-log(sqrt(pi) * 10^(-10))))

    Es = []
    dadthetas = []
    seconds = []
    thirds = []
    # now iterate through the samples
    for i in 1:num_samples
        # get the current sample
        coords = samples[i, :]
        coords_prime = samples_prime[i, :]
        
        # append Hpsi to Es
        push!(Es, Hpsi(coords, coords_prime, alpha))
        push!(dadthetas, dA_dtheta(coords))
        push!(seconds, second_term(coords))
        push!(thirds, third_term(coords, coords_prime, alpha))
    end
    # compute the average of Es
    E_avg = 1/num_samples * sum(Es)
    dadtheta_avg = 1/num_samples .* gradsum(dadthetas)
    seconds_avg = 1/num_samples .* gradsum(seconds)
    thirds_avg = 1/num_samples .* gradsum(thirds)
    # @show E_avg
    # @show dadtheta_avg[Flux.params(model)[1]]
    # @show dadtheta_avg[Flux.params(model)[2]]
    # @show dadtheta_avg[Flux.params(model)[3]]
    # @show dadtheta_avg[Flux.params(model)[4]]
    # @show seconds_avg[Flux.params(model)[4]]
    uncert = std(Es)/sqrt(num_samples)

    g = 2 * E_avg .* dadtheta_avg .- 2 .* seconds_avg  .- 2 .* thirds_avg
    return g, E_avg, uncert
end

function train(initial_params, iterations, num_samples, learning_rate, variation)
    opt = ADAM(learning_rate)  # Ensure to set the learning rate
    # set the model parameters 
    Flux.loadparams!(model, initial_params)
    # arrays to store the energies and uncertainties
    energies = []
    uncerts = []

    # now train
    iter =  ProgressBars.tqdm(1:iterations) 
    for step in iter
        # compute the gradient 
        g, energy, uncert =  grad(num_samples, 1000, 200, variation)
        # append the energy and uncert
        push!(energies, energy)
        push!(uncerts, uncert)
        set_description(iter , string(energy))
        # update the parameters
        for p in Flux.params(model)
            Flux.update!(opt, p, g[p])  # Update parameters in place
        end
        IJulia.clear_output(true)
    end
    return energies, uncerts
end

samples, samples_prime, rate = sample(psi_sq, num_particles, 10^4, 1000, 100, 1)
println(rate)