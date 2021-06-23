import numpy as np
#from lambert2 import lambertw
import torch
from torch.utils.cpp_extension import load

lambertw = load(name="lambertw", sources=["lambert.cu"])

# kNoSpike = 1000.0
# decay_rate = 0.18176949150701854
# decay_params = {'rate': decay_rate, 'rate_inverse': 1/decay_rate}
layer_size = [784, 512, 10]
weights = [np.zeros([512, 784]), np.zeros([10, 512])]
fire_threshold = 1.0
penalty_no_spike = 48.3748
clip_derivative = 539.7

# Minimum argument for the main branch of the Lambert W function.
kMinLambertArg = torch.tensor(-1.0 / np.e, dtype=torch.double)
# Maximum argument for which gsl_sf_lambert_W0 produces a valid result.
kMaxLambertArg = torch.tensor(1.7976131e+308, dtype=torch.double)


def GetSortedIndices(activations):
    return torch.argsort(activations)

def ExponentiateSortedValidSpikes(activations, layer_param):
    return torch.where(activations==layer_param.kNoSpike, layer_param.kNoSpike, torch.exp(layer_param.decay_rate * activations))


def ActivateNeuronAlpha_itr(weight, activation, exp_activation, A_B_W, layer_param):
    #spike_time = activation.new_full(len(weight), kNoSpike)
    w_exp_z = torch.squeeze(weight) * exp_activation
    A_B_W[0] += w_exp_z
    A_B_W[1] += w_exp_z * activation

# The value of the first derivative of the activation function in the
# intersection point with the fire threshold is given by *A multiplied by a
# never-negative value. Thus, if *A is negative the intersection will be in
# a decreasing-potential area, and thus not a spike.
    # if A_B_W[0] < 0:
    #     continue
    b_over_a = A_B_W[1]/A_B_W[0]
    lambert_arg = -layer_param.decay_rate* layer_param.threshold / A_B_W[0] * torch.exp(layer_param.decay_rate * b_over_a)
    lambert_check = torch.logical_and(torch.greater_equal(lambert_arg, layer_param.kMinLambertArg), torch.less_equal(lambert_arg, layer_param.kMaxLambertArg))
    lambert_tmp = torch.where(lambert_check, lambert_arg, layer_param.zero)
    w_temp = lambertw(lambert_tmp)
    spike_time = b_over_a - w_temp * layer_param.decay_rate_inverse
    abw_lambert_check = torch.logical_and(torch.greater_equal(A_B_W[0], 0), lambert_check)
    spike_time = torch.where(torch.logical_and(abw_lambert_check, torch.greater_equal(spike_time, activation)), spike_time, layer_param.kNoSpike)
    A_B_W[2] = torch.where(abw_lambert_check, w_temp, A_B_W[2])
    return spike_time, A_B_W

def ActivateNeuronAlpha(weight, activations, exp_activation, sorted_indices, layer_param):
    batch_size = len(activations)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    #causal_set, a, b, w, decay_params
    spike_time = activations.new_full([batch_size, fan_out], layer_param.kNoSpike)
    A_B_W =  activations.new_full([3, batch_size, fan_out], 0)
    causal_set =  activations.new_full([batch_size, fan_out, fan_in], False, dtype=torch.bool)
    for i in range(fan_in):
        spike_idx = sorted_indices[:,i:i+1]
        this_activation = torch.gather(activations, 1, spike_idx)
        
        done = spike_time <= this_activation
        spike_time_tmp, A_B_W_tmp = ActivateNeuronAlpha_itr(torch.gather(weight[None,:,:].expand([batch_size,-1,-1]), 2, spike_idx[:,None,:].expand(-1,fan_out, -1)), this_activation, torch.gather(exp_activation, 1, spike_idx), A_B_W.clone(), layer_param)
        spike_time = torch.where(done, spike_time, spike_time_tmp)
        A_B_W = torch.where(done, A_B_W, A_B_W_tmp)

    causal_set = torch.logical_or(spike_time[:, :, None]==layer_param.kNoSpike, spike_time[:,:,None] > activations[:,None,:])
    return spike_time, A_B_W, causal_set

def ActivateNeuronAlpha2(weight, activation, exp_activation, sorted_indices, threshold):
    #causal_set, a, b, w, decay_params
    device = activation.device
    A_B_W = torch.tensor([0.0, 0.0, 0.0], device = device)
    spike_time = kNoSpike
    causal_set = torch.zeros_like(activation)
    # input spike one by one
    for spike_idx in sorted_indices:
        if spike_time <= activation[spike_idx]:
            return spike_time, A_B_W, causal_set # no need to integrate more
            # Otherwise, integrate this spike
            # Reset spike time, in case an inhibitory input cancels a potential spike.
        spike_time = kNoSpike
        causal_set[spike_idx] = 1

        w_exp_z = weight[spike_idx] * exp_activation[spike_idx]
        A_B_W[0] += w_exp_z
        A_B_W[1] += w_exp_z * activation[spike_idx]

# The value of the first derivative of the activation function in the
# intersection point with the fire threshold is given by *A multiplied by a
# never-negative value. Thus, if *A is negative the intersection will be in
# a decreasing-potential area, and thus not a spike.
        if A_B_W[0] < 0:
            continue
        b_over_a = A_B_W[1]/A_B_W[0]
        lambert_arg = -decay_params['rate'] * threshold / A_B_W[0] * torch.exp(decay_params['rate'] * b_over_a)
        if lambert_arg >= kMinLambertArg and lambert_arg <= kMaxLambertArg:
            val = lambertw(lambert_arg)
            A_B_W[2] = val
            spike_time = b_over_a - A_B_W[2] * decay_params['rate_inverse']

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
    return spike_time, A_B_W, causal_set

