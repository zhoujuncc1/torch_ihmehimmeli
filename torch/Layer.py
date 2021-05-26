import numpy as np
import tempcoder
from tempcoder import GetSortedIndices, ExponentiateSortedValidSpikes, ActivateNeuronAlpha
import torch

def onehot(target, n_class):
    out = torch.zeros(n_class)
    out[target] = 1
    return out

def _generatePulse(n_pulse, input_range):
    return torch.linspace(input_range[0], input_range[1], n_pulse+2)[1:-1]

class Layer:
    def __init__(self, in_feature, out_feature, threashold=1.0, decay_param=None, n_pulse=0, input_range=(0,1)) -> None:
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.threshold = threashold
        self.n_pulse = n_pulse
        self.input_range = input_range
        if decay_param is None:
            self.decay_param = tempcoder.decay_params
        else:
            self.decay_param = decay_param
        if self.n_pulse > 0:
            self.pulse = _generatePulse(n_pulse, self.input_range)

        self.weight = torch.rand((out_feature, in_feature+n_pulse))

    def forward(self, activation):
        self.input_activation = torch.cat([activation, self.pulse])
        sorted_indices = GetSortedIndices(self.input_activation)
        exp_activation = ExponentiateSortedValidSpikes(
            self.input_activation, self.decay_param['rate'])

        self.spike_time, self.A_B_W, self.causal_set = ActivateNeuronAlpha(
                self.weight, self.input_activation, exp_activation, sorted_indices, self.threshold)
        self.A = self.A_B_W[0]
        self.B = self.A_B_W[1]
        self.W = self.A_B_W[2]
        return self.spike_time

    def _compute_weight_gradient_real(self, grad):
        e_K_tp = torch.exp(self.decay_param['rate']*self.input_activation)
        d = e_K_tp[None,:]*(self.input_activation[None,:] - (self.B / self.A - self.W * self.decay_param['rate_inverse'])[:, None]) / ((self.A * (1.0 + self.W))[:, None])
        d = torch.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
        return torch.clip(d * grad[:,None], -tempcoder.clip_derivative, tempcoder.clip_derivative)
    def _compute_weight_gradient(self, grad):
        grad_w_real = self._compute_weight_gradient_real(grad)
        grad_w = torch.where(self.causal_set==1, grad_w_real, 0.0)
        grad_w = torch.where(self.spike_time[:,None] == tempcoder.kNoSpike, - tempcoder.penalty_no_spike, grad_w)
        return grad_w

    def _compute_input_gradient_real(self, grad):
        e_K_tp = torch.exp(self.decay_param['rate']*self.input_activation)
        d = self.weight * e_K_tp[None, :] * (self.decay_param['rate'] * (self.input_activation[None, :] - (self.B / self.A)[:, None]) + self.W[:, None] + 1) / (self.A * (1.0 + self.W))[:, None]
        d = torch.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
        return torch.clip(grad[:, None]*d, -tempcoder.clip_derivative, tempcoder.clip_derivative)

    def _compute_input_gradient(self, grad):
        check = torch.logical_and(torch.logical_and(torch.less(self.spike_time[:, None], tempcoder.kNoSpike),torch.less(self.input_activation[None, :], tempcoder.kNoSpike)),self.causal_set)
        grad_x_real = self._compute_input_gradient_real(grad)
        return torch.sum(torch.where(check, grad_x_real, 0.0), axis=0)
    def backward(self, grad):
        self.grad_w = self._compute_weight_gradient(grad)
        self.grad_x = self._compute_input_gradient(grad)
        self.grad_pulse = self.grad_x[-self.n_pulse:]


def loss(x, target, penalty_output_spike_time_=0):
    target = onehot(target, len(x))
    min_value = torch.min(x)
    loss = _torch_cross_entropy_loss_with_kEps(min_value-x, target)
    loss += penalty_output_spike_time_ * torch.sum(x*x)
    return loss

kEps = 1e-8
def _torch_cross_entropy_loss_with_kEps(x, target):
    exp_out = torch.exp(x)
    exp_sum = torch.sum(exp_out)
    return torch.sum(target * -torch.log(exp_out/exp_sum  + kEps))

def loss_derivative(x, target, penalty_output_spike_time_=0.0):
    target = onehot(target, len(x))
    min_value = torch.min(x)
    exp_out = torch.exp(min_value-x)
    exp_sum = torch.sum(exp_out)
    return -(exp_out/exp_sum - target + penalty_output_spike_time_ * exp_out/exp_sum)