from unittest.main import main
import torch

import lambertw
lambertw=lambertw.lambertw

import spiketime
# from test.lambert2 import lambertw

clip_derivative = 539.7

def GetSortedIndices(activations):
    return torch.argsort(activations)

def GetSortedSpikes(activations):
    return torch.sort(activations)


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
    for i in range(fan_in):
        spike_idx = sorted_indices[:,i:i+1]
        this_activation = torch.gather(activations, 1, spike_idx)
        
        done = spike_time <= this_activation
        spike_time_tmp, A_B_W_tmp = ActivateNeuronAlpha_itr(torch.gather(weight[None,:,:].expand([batch_size,-1,-1]), 2, spike_idx[:,None,:].expand(-1,fan_out, -1)), this_activation, torch.gather(exp_activation, 1, spike_idx), A_B_W.clone(), layer_param)
        spike_time = torch.where(done, spike_time, spike_time_tmp)
        A_B_W = torch.where(done, A_B_W, A_B_W_tmp)

    causal_set = torch.logical_or(spike_time[:, :, None]==layer_param.kNoSpike, spike_time[:,:,None] > activations[:,None,:])
    return spike_time, A_B_W, causal_set

def ActivateNeuronAlpha_fast(weight, activations, sorted_activations, sorted_exp_activation, sorted_indices, layer_param):
    batch_size = len(activations)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    sorted_activation_expand = sorted_activations[:,None,:].expand(-1, fan_out, -1).contiguous()
    valid_activation_map = sorted_activation_expand!=layer_param.kNoSpike
    weight_extend = weight[None,:,:].expand(batch_size, -1,-1)
    weight_extend = torch.gather(weight_extend, -1, sorted_indices[:,None,:].expand(-1, fan_out,-1))
    w_exp_z = weight_extend * sorted_exp_activation[:,None,:]
    A = torch.cumsum(w_exp_z, dim=-1)
    B = torch.cumsum(w_exp_z*sorted_activation_expand, dim=-1)

    b_over_a = B/A
    lambert_arg = -layer_param.decay_rate* layer_param.threshold / A * torch.exp(layer_param.decay_rate * b_over_a)
    lambert_check = torch.logical_and(valid_activation_map, torch.logical_and(torch.greater_equal(lambert_arg, layer_param.kMinLambertArg), torch.less_equal(lambert_arg, layer_param.kMaxLambertArg)))
    lambert_tmp = torch.where(lambert_check, lambert_arg, layer_param.zero)
    w_temp = lambertw(lambert_tmp)
    spike_time = b_over_a - w_temp * layer_param.decay_rate_inverse
    abw_lambert_check = torch.logical_and(torch.greater_equal(A, layer_param.zero), lambert_check)
    W = torch.where(abw_lambert_check, w_temp, layer_param.zero)

    spike_time = torch.where(abw_lambert_check, spike_time, layer_param.kNoSpike)
    valid_spike_time = torch.logical_and(spike_time > sorted_activation_expand, valid_activation_map)

    spike_time, spike_time_index= spiketime.spiketime(spike_time, sorted_activation_expand, layer_param.kNoSpike)
    causal_set = torch.logical_or(spike_time==layer_param.kNoSpike, spike_time > activations[:,None,:])

    A = torch.gather(A, -1, spike_time_index).squeeze()
    B = torch.gather(B, -1, spike_time_index).squeeze()

    W = torch.gather(W, -1, spike_time_index).squeeze()

    return spike_time.squeeze(), torch.stack([A,B,W]), causal_set
