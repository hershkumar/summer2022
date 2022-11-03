#quick NN output tester
from jax.nn import celu
import numpy as np
# NN takes 1 input, spits out 1 output

# one hidden layer:
num_nodes = 5

# first set of weights are the input to each of the hidden layer nodes
ih = np.array([[-0.2597665],[-0.29645568],[0.09428109],[0.8580086],[-0.46084416]])


# second set is the hidden layer to the output
ho = np.array([[0.3290954 ],[-0.4969729],[0.10998356],[-0.68060553],[-1.2391438]])
ho = np.transpose(ho)


def act(x):
    return celu(x)


# compute the output
def output(x):
    inp = np.array([x])
    temp = ih @ inp
    # pass through activation
    temp = act(temp)
    print(temp)
    # now pass to output layer
    temp = ho @ temp
    temp = act(temp)
    print(temp)

output(3.0)