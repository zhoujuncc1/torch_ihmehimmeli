import numpy as np
from lambert2 import lambertw


kNoSpike = 1000
decay_rate = 0.18176949150701854
decay_params = {'rate': decay_rate, 'rate_inverse': 1/decay_rate}
layer_size = [784, 512, 10]
weights = [np.zeros([512, 784]), np.zeros([10, 512])]
fire_threshold = 1.0


# Minimum argument for the main branch of the Lambert W function.
kMinLambertArg = -1.0 / np.e
# Maximum argument for which gsl_sf_lambert_W0 produces a valid result.
kMaxLambertArg = 1.7976131e+308


def GetSortedIndices(activations):
    return np.argsort(activations)

def ExponentiateSortedValidSpikes(activations, sorted_indices, decay_rate):
    exp_activations = np.zeros_like(activations, dtype=np.float64)
    exp_activations.fill(kNoSpike)
    i = 0
    while i < len(sorted_indices) and activations[sorted_indices[i]] < kNoSpike:
        exp_activations[sorted_indices[i]] = np.exp(
            decay_rate * activations[sorted_indices[i]])
        i += 1
    return exp_activations


def ActivateNeuronAlpha(weight, activation, exp_activation, sorted_indices, threshold):
    #causal_set, a, b, w, decay_params
    A = 0.0
    B = 0.0
    W = 0.0
    spike_time = kNoSpike
    causal_set = np.zeros_like(activation)
    # input spike one by one
    for spike_idx in sorted_indices:
        if spike_time <= activation[spike_idx]:
            return spike_time, A, B, W, causal_set # no need to integrate more
            # Otherwise, integrate this spike
            # Reset spike time, in case an inhibitory input cancels a potential spike.
        spike_time = kNoSpike
        causal_set[spike_idx] = 1

        w_exp_z = weight[spike_idx] * exp_activation[spike_idx]
        A += w_exp_z
        B += w_exp_z * activation[spike_idx]

# The value of the first derivative of the activation function in the
# intersection point with the fire threshold is given by *A multiplied by a
# never-negative value. Thus, if *A is negative the intersection will be in
# a decreasing-potential area, and thus not a spike.
        if A < 0:
            continue
        b_over_a = B/A
        lambert_arg = -decay_params['rate'] * threshold / A * np.exp(decay_params['rate'] * b_over_a)
        if lambert_arg >= kMinLambertArg and lambert_arg <= kMaxLambertArg:
            val = lambertw(lambert_arg)
            W = val
            spike_time = b_over_a - W * decay_params['rate_inverse']

            # For inhibitory weights, this might be a false alarm.
            # This is not the same as spike_time < inputs[spike_ind]: it is also true for NaNs.
            if not (spike_time >= activation[spike_idx]):
                spike_time = kNoSpike
        # END_IF
    # END_FOR
    # If we get here, either there is no spike, in which case
    # all presynaptic neurons are to blame, or there is eventually
    # a spike caused by all presynaptic inputs.
    if (spike_time == kNoSpike):
        causal_set[:] = True
    return spike_time, A, B, W, causal_set

def driver_function():
    activation= np.zeros([784]) # input with batch
    for layer in range(len(layer_size)-1):
        sorted_indices = np.argsort(activation)# sort indices from small to large
        exp_activation = ExponentiateSortedValidSpikes(activation, sorted_indices, decay_rate)
        activation_next = np.zeros(layer_size[layer+1])
        A = np.zeros_like(activation_next)
        B = np.zeros_like(activation_next)
        W = np.zeros_like(activation_next)
        causal_set = np.zeros([layer_size[layer+1], layer_size[layer]])
        for n in range(layer_size[layer+1]):
            act,a,b,w,c = ActivateNeuronAlpha(weights[layer][n], activation, exp_activation, sorted_indices, fire_threshold)
            activation_next[n]=act 
            A[n]=a
            B[n]=b
            W[n]=w
            causal_set[n]=c
        activation=activation_next