import numpy as np
from numpy.core.numeric import tensordot
import tempcoder
from tempcoder import GetSortedIndices, ExponentiateSortedValidSpikes, ActivateNeuronAlpha


def onehot(target, n_class):
    out = np.zeros(n_class)
    out[target] = 1
    return out

def _generatePulse(n_pulse, input_range):
    return np.linspace(input_range[0], input_range[1], n_pulse+2)[1:-1]

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

        self.weight = np.random.random((out_feature, in_feature+n_pulse))

    def forward(self, activation):
        self.input_activation = np.concatenate([activation, self.pulse])
        sorted_indices = GetSortedIndices(self.input_activation)
        exp_activation = ExponentiateSortedValidSpikes(
            self.input_activation, self.decay_param['rate'])
        self.spike_time = []
        self.causal_set = []
        spike_time, A_B_W, causal_set = ActivateNeuronAlpha(self.weight, self.input_activation, exp_activation, sorted_indices, self.threshold)

        self.causal_set = causal_set
        self.A = A_B_W[0]
        self.B = A_B_W[1]
        self.W = A_B_W[2]
        self.spike_time = spike_time
        return self.spike_time



    def _compute_weight_gradient_real(self, grad):
        e_K_tp = np.exp(self.decay_param['rate']*self.input_activation)
        d = e_K_tp[np.newaxis,...]*(self.input_activation[np.newaxis,...] - (self.B / self.A - self.W * self.decay_param['rate_inverse'])[..., np.newaxis]) / ((self.A * (1.0 + self.W))[..., np.newaxis])
        d = np.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
        return np.clip(d * grad[..., np.newaxis], -tempcoder.clip_derivative, tempcoder.clip_derivative)
    def _compute_weight_gradient(self, grad):
        grad_w_real = self._compute_weight_gradient_real(grad)
        grad_w = np.where(self.causal_set, grad_w_real, 0)
        grad_w = np.where(self.spike_time[..., np.newaxis] == tempcoder.kNoSpike, - tempcoder.penalty_no_spike, grad_w)
        return grad_w

    def _compute_input_gradient_real(self, grad):
        e_K_tp = np.exp(self.decay_param['rate']*self.input_activation)
        d = self.weight * e_K_tp[np.newaxis,...] * (self.decay_param['rate'] * (self.input_activation[np.newaxis,...] - (self.B / self.A)[..., np.newaxis]) + self.W[..., np.newaxis] + 1) / (self.A * (1.0 + self.W))[..., np.newaxis]
        d = np.clip(d, -tempcoder.clip_derivative, tempcoder.clip_derivative)
        return np.clip(grad[..., np.newaxis]*d, -tempcoder.clip_derivative, tempcoder.clip_derivative)

    def _compute_input_gradient(self, grad):
        check = np.logical_and(np.logical_and(np.less(self.spike_time[..., np.newaxis], tempcoder.kNoSpike),np.less(self.input_activation[np.newaxis, ...], tempcoder.kNoSpike)),self.causal_set)
        grad_x_real = self._compute_input_gradient_real(grad)
        return np.sum(np.where(check, grad_x_real, 0), axis=0)

    def backward(self, grad):
        self.grad_w = self._compute_weight_gradient(grad)
        self.grad_x = self._compute_input_gradient(grad)
        self.grad_pulse = self.grad_x[-self.n_pulse:]


def loss(x, target, penalty_output_spike_time_=0):
    target = onehot(target, len(x))
    min_value = np.min(x)
    loss = _torch_cross_entropy_loss_with_kEps(min_value-x, target)
    loss += penalty_output_spike_time_ * np.sum(x*x)
    return loss

kEps = 1e-8
def _torch_cross_entropy_loss_with_kEps(x, target):
    exp_out = np.exp(x)
    exp_sum = np.sum(exp_out)
    return np.sum(target * -np.log(exp_out/exp_sum  + kEps))

def loss_derivative(x, target, penalty_output_spike_time_=0.0):
    target = onehot(target, len(x))
    min_value = np.min(x)
    exp_out = np.exp(min_value-x)
    exp_sum = np.sum(exp_out)
    return -(exp_out/exp_sum - target + penalty_output_spike_time_ * exp_out/exp_sum)