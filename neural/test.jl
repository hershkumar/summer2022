using Random

function metropolis(f, proposal, x0, n_samples)
    # f: the target distribution
    # proposal: a function that proposes a new sample given the current sample
    # x0: the initial sample
    # n_samples: the number of samples to generate
    
    # initialize the chain
    x = x0
    
    # initialize the acceptance counter
    n_accept = 0
    
    # initialize the samples array
    samples = zeros(n_samples)
    
    # loop over the number of samples
    for i in 1:n_samples
        # propose a new sample
        x_new = proposal(x)
        
        # calculate the acceptance probability
        alpha = min(1, f(x_new) / f(x))
        
        # generate a uniform random number
        u = rand()
        
        # decide whether to accept the new sample
        if u < alpha
            x = x_new
            n_accept += 1
        end
        
        # add the current sample to the samples array
        samples[i] = x
    end
    
    # return the samples and the acceptance rate
    return samples, n_accept / n_samples
end

function proposal(x)
    # the proposal distribution is a normal distribution with mean x and standard deviation 1
    return x + randn()
end

function f(x)
    # the target distribution is a normal distribution with mean 0 and standard deviation 1
    return exp(-x^2 / 2) / sqrt(2 * pi)
end

# now sample from the target distribution
# record how long it takes to generate 10000 samples
@time samples, acceptance_rate = metropolis(f, proposal, 0, 10000)
println(length(samples))
println(samples[1:10])