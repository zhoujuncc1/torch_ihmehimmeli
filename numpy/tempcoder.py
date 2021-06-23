import numpy as np
from lambert2 import lambertw


kNoSpike = 1000
decay_rate = 0.18176949150701854
decay_params = {'rate': decay_rate, 'rate_inverse': 1/decay_rate}
layer_size = [784, 512, 10]
weights = [np.zeros([512, 784]), np.zeros([10, 512])]
fire_threshold = 1.0
penalty_no_spike = 48.3748
clip_derivative = 539.7

# Minimum argument for the main branch of the Lambert W function.
kMinLambertArg = -1.0 / np.e
# Maximum argument for which gsl_sf_lambert_W0 produces a valid result.
kMaxLambertArg = 1.7976131e+308


def GetSortedIndices(activations):
    return np.argsort(activations)

def ExponentiateSortedValidSpikes(activations, decay_rate):
    return np.where(activations==kNoSpike, kNoSpike, np.exp(decay_rate * activations))



# def ActivateNeuronAlpha(weight, activation, exp_activation, sorted_indices, threshold):
#     #causal_set, a, b, w, decay_params
#     A = 0.0
#     B = 0.0
#     W = 0.0
#     spike_time = kNoSpike
#     causal_set = np.zeros_like(activation)
#     # input spike one by one
#     for spike_idx in sorted_indices:
#         if spike_time <= activation[spike_idx]:
#             return spike_time, A, B, W, causal_set # no need to integrate more
#             # Otherwise, integrate this spike
#             # Reset spike time, in case an inhibitory input cancels a potential spike.
#         spike_time = kNoSpike
#         causal_set[spike_idx] = 1

#         w_exp_z = weight[spike_idx] * exp_activation[spike_idx]
#         A += w_exp_z
#         B += w_exp_z * activation[spike_idx]

# # The value of the first derivative of the activation function in the
# # intersection point with the fire threshold is given by *A multiplied by a
# # never-negative value. Thus, if *A is negative the intersection will be in
# # a decreasing-potential area, and thus not a spike.
#         if A < 0:
#             continue
#         b_over_a = B/A
#         lambert_arg = -decay_params['rate'] * threshold / A * np.exp(decay_params['rate'] * b_over_a)
#         if lambert_arg >= kMinLambertArg and lambert_arg <= kMaxLambertArg:
#             val = lambertw(lambert_arg)
#             W = val
#             spike_time = b_over_a - W * decay_params['rate_inverse']

#             # For inhibitory weights, this might be a false alarm.
#             # This is not the same as spike_time < inputs[spike_ind]: it is also true for NaNs.
#             if not (spike_time >= activation[spike_idx]):
#                 spike_time = kNoSpike
#         # END_IF
#     # END_FOR
#     # If we get here, either there is no spike, in which case
#     # all presynaptic neurons are to blame, or there is eventually
#     # a spike caused by all presynaptic inputs.
#     if (spike_time == kNoSpike):
#         causal_set[:] = True
#     return spike_time, A, B, W, causal_set

def ActivateNeuronAlpha_itr(weight, activation, exp_activation, A_B_W, threshold):
    #spike_time = activation.new_full(len(weight), kNoSpike)
    w_exp_z = weight * exp_activation
    A_B_W[0] += w_exp_z
    A_B_W[1] += w_exp_z * activation

# The value of the first derivative of the activation function in the
# intersection point with the fire threshold is given by *A multiplied by a
# never-negative value. Thus, if *A is negative the intersection will be in
# a decreasing-potential area, and thus not a spike.
    # if A_B_W[0] < 0:
    #     continue
    b_over_a = A_B_W[1]/A_B_W[0]
    lambert_arg = -decay_params['rate'] * threshold / A_B_W[0] * np.exp(decay_params['rate'] * b_over_a)
    lambert_check = np.logical_and(np.greater_equal(lambert_arg, kMinLambertArg), np.less_equal(lambert_arg, kMaxLambertArg))
    lambert_tmp = np.where(lambert_check, lambert_arg, 0.0)
    w_temp = lambertw(lambert_tmp)
    spike_time = b_over_a - w_temp * decay_params['rate_inverse']
    abw_lambert_check = np.logical_and(np.greater_equal(A_B_W[0], 0), lambert_check)
    spike_time = np.where(np.logical_and(abw_lambert_check, np.greater_equal(spike_time, activation)), spike_time, kNoSpike)
    A_B_W[2] = np.where(abw_lambert_check, w_temp, A_B_W[2])
    return spike_time, A_B_W

def ActivateNeuronAlpha(weight, activations, exp_activation, sorted_indices, threshold):
    #causal_set, a, b, w, decay_params
    spike_time = np.full(len(weight), kNoSpike)
    A_B_W = np.zeros([3,len(weight)])
    causal_set = np.zeros_like(weight)
    done = np.ones_like(spike_time)

    for spike_idx in sorted_indices:
        done = spike_time <= activations[spike_idx]

        causal_set[:,spike_idx] = np.where(done, causal_set[:,spike_idx], 1)

        spike_time_tmp, A_B_W_tmp = ActivateNeuronAlpha_itr(weight[:,spike_idx], activations[spike_idx], exp_activation[spike_idx], A_B_W.copy(), threshold)
        spike_time = np.where(done, spike_time, spike_time_tmp)
        A_B_W = np.where(done, A_B_W, A_B_W_tmp)

    causal_set = np.where(spike_time[:, None]==kNoSpike, True, causal_set)
    return spike_time, A_B_W, causal_set
