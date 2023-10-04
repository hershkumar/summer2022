# defines the metropolis sampling function 

using Distributions
function sample(distribution, params,  num_particles, num_samples, num_thermalization, num_skip, variation)
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
		x_prime = x + rand(Uniform(-variation, variation), num_particles) 
        # generate a random number between 0 and 1
		r = rand(Uniform(0,1))
        if r < distribution(x_prime, params)/distribution(x, params)
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
    samples_prime = samples
    for i in 1:num_samples
        samples_prime[i, 2] = samples[i, 1]
    end

    return samples, samples_prime, (num_accepted/total_steps) 
end

function distribution(x, params)
    return exp(x[1]^2 - x[1]^4)
end

# sample the distribution

@time samples, samples_prime, acceptance_rate = sample(distribution, 1, 2, 10^4, 200, 10, 1)
# println(samples)

# plot the samples
using PyPlot
pygui(true)

figure(figsize=(5,5))
# get a list of all the first values
x = Float64[]
for i in 1:size(samples, 1)
    push!(x, samples[i, 1])
end

#histogram the x values
hist(x, bins=20, density=true)
# plot the distribution
xs = -5:0.1:5
ys = exp.(xs.^2 - xs.^4)
plot(xs, ys, color="red")

title("Acceptance rate = $acceptance_rate")
plt.show()
