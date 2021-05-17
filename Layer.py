import numpy as np
import tempcoder
from tempcoder import GetSortedIndices, ExponentiateSortedValidSpikes, ActivateNeuronAlpha


def onehot(target, n_class):
    out = np.zeros(n_class)
    out[target] = 1
    return out

def _generatePulse(n_pulse, input_range):
    pulse = np.zeros(n_pulse)
    pulse_spacing = (input_range[1] - input_range[0]) / (n_pulse+1)
    for i in range(n_pulse):
        pulse[i] = input_range[0] + pulse_spacing * (i + 1)
    return pulse

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
        activation_pulse = np.concatenate([activation, self.pulse])
        sorted_indices = GetSortedIndices(activation_pulse)
        exp_activation = ExponentiateSortedValidSpikes(
            activation_pulse, sorted_indices, self.decay_param['rate'])
        self.spike_time = []
        self.A = []
        self.B = []
        self.W = []
        self.causal_set = []
        for i in range(self.out_feature):
            t, a, b, w, c = ActivateNeuronAlpha(
                self.weight[i], activation_pulse, exp_activation, sorted_indices, self.threshold)
            self.spike_time.append(t)
            self.A.append(a)
            self.B.append(b)
            self.W.append(w)
            self.causal_set.append(c)
        self.causal_set = np.asarray(self.causal_set)
        return self.spike_time


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