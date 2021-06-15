import unittest


from Layer import Layer, loss,loss_derivative, LinearFunction, LayerParam
import subprocess
import os
import numpy as np
from array import array
import math
import torch


n_pulse = 10
kNoSpike=1000
threshold = 1.16732
decay_rate = 0.18176949150701854
penalty_no_spike = 48.3748
pulse_init_multiplier = 7.83912
nopulse_init_multiplier = -0.275419
input_range = (0,1)
layer_params = LayerParam(kNoSpike, decay_rate, threshold, penalty_no_spike, pulse_init_multiplier, nopulse_init_multiplier, input_range, dtype=torch.float)
class TestLayer(unittest.TestCase):
    def test_forward(self):
        m = 200
        n = 100
        activation = np.random.random((2,m))
        activation = torch.tensor(activation, dtype=torch.float)
        threshold=1.0
        n_pulse=2
        layer = Layer(m, n, threshold, n_pulse=n_pulse, layer_param = layer_params)
        layer_activation_next = layer.forward(activation)

        func_activation_next = LinearFunction.apply(activation, layer.weight, layer.pulse, layer.layer_param)

        np.testing.assert_almost_equal(func_activation_next.detach().numpy(), layer_activation_next.detach().numpy(), decimal=6)

    def test_backward(self):
        m = 200
        n = 100
        activation = np.random.random((2,m))
        activation = torch.tensor(activation, requires_grad=True, dtype=torch.float)
        

        threshold=1.0
        n_pulse=2
        layer = Layer(m, n, threshold, n_pulse=n_pulse, layer_param = layer_params)
        layer_activation_next = layer.forward(activation)
        func_activation_next = LinearFunction.apply(activation, layer.weight, layer.pulse, layer.layer_param)
        np.testing.assert_almost_equal(func_activation_next.detach().numpy(), layer_activation_next.detach().numpy(), decimal=6)

        loss = torch.tensor(np.random.random((2,n)), requires_grad=True, dtype=torch.float32)

        func_activation_next.backward(loss)

        layer.backward(loss)
        np.testing.assert_almost_equal(layer.grad_w.mean(dim=0).detach().numpy(), layer.weight.grad.detach().numpy())
        np.testing.assert_almost_equal(layer.grad_x.detach().numpy()[:,:-n_pulse], activation.grad.detach().numpy())


if __name__ == '__main__':
    unittest.main()
