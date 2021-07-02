import numpy as np
import tempcoder
from tempcoder import *
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math


def _generatePulse(n_pulse, input_range):
    return torch.linspace(input_range[0], input_range[1], n_pulse+2)[1:-1]




def _compute_weight_gradient_real(grad, decay_params, input_activation, A, B, W):
    e_K_tp = torch.exp(decay_params['rate']*input_activation)
    d = e_K_tp[:, None,:]*(input_activation[:,None,:] - (B / A - W * decay_params['rate_inverse'])[:,:, None]) / ((A * (1.0 + W))[:,:,None])
    d = torch.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
    return torch.clip(d * grad[:, :,None], -tempcoder.clip_derivative, tempcoder.clip_derivative)

def _compute_weight_gradient(grad, input_activation, causal_set, spike_time, A, B, W, layer_param):
    grad_w_real = _compute_weight_gradient_real(grad, layer_param.decay_params, input_activation, A, B, W)
    grad_w = torch.where(causal_set, grad_w_real, layer_param.zero)
    grad_w = torch.where(spike_time[:, :,None] == layer_param.kNoSpike, - layer_param.penalty_no_spike, grad_w)
    return grad_w

def _compute_input_gradient_real(grad, input_activation, weight, A, B, W, decay_params):
    e_K_tp = torch.exp(decay_params['rate']*input_activation)
    d = weight[None, :,:] * e_K_tp[:, None, :] * (decay_params['rate'] * (input_activation[:, None, :] - (B / A)[:,:, None]) + W[:, :, None] + 1) / (A * (1.0 + W))[:,:, None]
    d = torch.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
    return torch.clip(grad[:, :, None]*d, -tempcoder.clip_derivative, tempcoder.clip_derivative)

def _compute_input_gradient(grad, input_activation, causal_set, spike_time, weight, A, B, W, layer_param):
    check = torch.logical_and(torch.logical_and(torch.less(spike_time[:, :, None], layer_param.kNoSpike),torch.less(input_activation[:, None, :], layer_param.kNoSpike)),causal_set)
    grad_x_real = _compute_input_gradient_real(grad, input_activation, weight, A, B, W, layer_param.decay_params)
    return torch.sum(torch.where(check, grad_x_real, layer_param.zero), axis=1)


class Layer:
    def __init__(self, in_feature, out_feature, threashold=1.0, n_pulse=2, layer_param=None, input_range=(0,1)) -> None:
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.threshold = threashold
        self.n_pulse = n_pulse
        self.input_range = input_range
        self.layer_param = layer_param
        if self.n_pulse > 0:
            self.pulse = _generatePulse(n_pulse, self.input_range)

        self.weight = torch.rand((out_feature, in_feature+n_pulse), requires_grad=True)

    def forward(self, activation):
        if self.n_pulse>0:
            self.input_activation = torch.cat([activation, self.pulse[None,:].expand([len(activation), -1])], dim=1)
        else:
            self.input_activation = activation
        sorted_activations, sorted_indices = GetSortedSpikes(self.input_activation)
        sorted_exp_activation = ExponentiateSortedValidSpikes(sorted_activations, self.layer_param)

        self.spike_time, self.A_B_W, self.causal_set = ActivateNeuronAlpha_fast(self.weight, self.input_activation, sorted_activations, sorted_exp_activation, sorted_indices, self.layer_param)
        self.A = self.A_B_W[0]
        self.B = self.A_B_W[1]
        self.W = self.A_B_W[2]
        return self.spike_time


    def backward(self, grad):
        self.grad_w = _compute_weight_gradient(grad, self.input_activation, self.causal_set, self.spike_time, self.A, self.B, self.W, self.layer_param)
        self.grad_x = _compute_input_gradient(grad, self.input_activation, self.causal_set, self.spike_time, self.weight, self.A, self.B, self.W, self.layer_param)
        self.grad_pulse = self.grad_x[-self.n_pulse:]




kEps = 1e-8
def _torch_cross_entropy_loss_with_kEps(x, target):
    exp_out = torch.exp(x)
    exp_sum = torch.sum(exp_out, dim=-1, keepdim=True)
    return torch.sum(target * -torch.log(exp_out/exp_sum  + kEps), dim=-1)


def loss(x, target, penalty_output_spike_time=0):
    min_value, _ = torch.min(x, dim=-1,keepdim=True)
    loss = _torch_cross_entropy_loss_with_kEps(min_value-x, target)
    loss += torch.sum(x*x, dim=-1) * penalty_output_spike_time
    return loss, min_value

def loss_derivative(x, target, min_value, penalty_output_spike_time=0.0):
    exp_out = torch.exp(min_value-x)
    exp_sum = torch.sum(exp_out, dim=-1, keepdim=True)
    return -(exp_out/exp_sum - target + penalty_output_spike_time * exp_out/exp_sum)


# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, pulse, layer_param):
        input_activation = torch.cat([input, pulse[None,:].expand([len(input), -1])], dim=1)
        sorted_activations, sorted_indices = GetSortedSpikes(input_activation)
        exp_activation = ExponentiateSortedValidSpikes(input_activation, layer_param)
        sorted_exp_activation = ExponentiateSortedValidSpikes(sorted_activations, layer_param)
        # spike_time, A_B_W, causal_set = ActivateNeuronAlpha(weight, input_activation, exp_activation, sorted_indices, layer_param)

        spike_time, A_B_W, causal_set = ActivateNeuronAlpha_fast(weight, input_activation, sorted_activations, sorted_exp_activation, sorted_indices, layer_param)
        # np.testing.assert_allclose(spike_time.detach().cpu().numpy(), spike_time2.detach().cpu().numpy(), rtol=1e-3)

        # np.testing.assert_allclose(causal_set.detach().cpu().numpy(), causal_set2.detach().cpu().numpy(), rtol=1e-3)


        # np.testing.assert_allclose(A_B_W.detach().cpu().numpy(), A_B_W2.detach().cpu().numpy(), rtol=1e-3)
        ctx.save_for_backward(input_activation, weight, spike_time, A_B_W, causal_set)
        ctx.layer_param = layer_param
        ctx.n_pulse = len(pulse)

        return spike_time


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input_activation, weight, spike_time, A_B_W, causal_set = ctx.saved_tensors
        grad_input = grad_weight = grad_pulse = None

        grad_weight = _compute_weight_gradient(grad_output, input_activation, causal_set, spike_time, A_B_W[0], A_B_W[1], A_B_W[2], ctx.layer_param)
        grad_input = _compute_input_gradient(grad_output,input_activation, causal_set, spike_time, weight, A_B_W[0], A_B_W[1], A_B_W[2], ctx.layer_param)
        grad_pulse = grad_input[:,-ctx.n_pulse:]

        return grad_input[:, :-ctx.n_pulse], grad_weight, grad_pulse, None, None

class CrossEntropyLoss(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, target, penalty_output_spike_time=0):
        loss_v, min_value = loss(x, target, penalty_output_spike_time)
        ctx.save_for_backward(x, target, min_value)
        ctx.penalty_output_spike_time = penalty_output_spike_time
        return loss_v

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        x, target, min_value = ctx.saved_tensors
        return loss_derivative(x, target, min_value, ctx.penalty_output_spike_time) , None, None


def cross_entropy_loss(x, target, penalty_output_spike_time=0):
    return CrossEntropyLoss.apply(x, target, penalty_output_spike_time)

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_pulse: int, layer_param) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_pulse= n_pulse
        self.layer_param=layer_param
        self.weight = Parameter(torch.Tensor(out_features, in_features+n_pulse))
        if n_pulse > 0 :
            self.pulse = Parameter(_generatePulse(n_pulse, layer_param.input_range))
        else:
            self.register_parameter('pulse', None)
        self._init_weight()

    def _init_weight(self):
        std_dev = math.sqrt(2/(self.in_features+self.out_features))
        
        if self.n_pulse > 0:
            nn.init.normal_(self.weight,std_dev*self.layer_param.pulse_init_multiplier, std_dev)
        else:
            nn.init.normal_(self.weight,std_dev*self.layer_param.nopulse_init_multiplier, std_dev)


    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.pulse, self.layer_param)

class LayerParam(nn.Module):
    def __init__(self, kNoSpike, decay_rate, threshold, penalty_no_spike, pulse_init_multiplier, nopulse_init_multiplier, input_range, dtype = torch.float32, device='cpu') -> None:
        super(LayerParam, self).__init__()
        self.kNoSpike = torch.tensor(kNoSpike,dtype=dtype, device=device)
        self.decay_rate = torch.tensor(decay_rate,dtype=dtype, device=device)
        self.decay_rate_inverse = 1/self.decay_rate
        self.decay_params = {'rate': self.decay_rate, 'rate_inverse': self.decay_rate_inverse}
        self.threshold = torch.tensor(threshold,dtype=dtype, device=device)
        self.kMinLambertArg = torch.tensor(-1.0 / np.e, dtype=dtype, device=device)
        self.kMaxLambertArg = torch.tensor(1.7976131e+308, dtype=dtype, device=device)
        self.zero = torch.tensor(0.0, dtype=dtype, device=device)
        self.one = torch.tensor(1.0, dtype=dtype, device=device)
        self.penalty_no_spike = torch.tensor(penalty_no_spike, dtype=dtype, device=device)
        self.pulse_init_multiplier = pulse_init_multiplier
        self.nopulse_init_multiplier = nopulse_init_multiplier
        self.input_range = input_range