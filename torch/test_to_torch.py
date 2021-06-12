import unittest


from Layer import Layer, loss,loss_derivative, LinearFunction
import subprocess
import os
import numpy as np
from array import array
import math
import torch

decay_rate = 0.18176949150701854
kNoSpike = 1000

class TestLayer(unittest.TestCase):
    def test_forward(self):
        m = 200
        n = 100
        activation = np.random.random(m)
        min_v = min(activation)
        max_v = max(activation)
        activation = torch.tensor(activation)
        threshold=1.0
        n_pulse=2
        layer = Layer(m, n, threshold, n_pulse=n_pulse, input_range = (min_v, max_v))
        layer_activation_next = layer.forward(activation)

        func_activation_next = LinearFunction.apply(activation, layer.weight, layer.pulse, layer.decay_param, layer.threshold)

        np.testing.assert_almost_equal(func_activation_next.detach().numpy(), layer_activation_next.detach().numpy(), decimal=6)

    def test_backward(self):
        os.chdir('/mnt/d/workspace/ihmehimmeli/python')
        m = 200
        n = 100
        activation = np.random.random(m)
        min_v = min(activation)
        max_v = max(activation)
        activation = torch.tensor(activation, requires_grad=True)
        

        threshold=1.0
        n_pulse=2
        layer = Layer(m, n, threshold, n_pulse=n_pulse, input_range = (min_v, max_v))
        layer_activation_next = layer.forward(activation)
        func_activation_next = LinearFunction.apply(activation, layer.weight, layer.pulse, layer.decay_param, layer.threshold)
        np.testing.assert_almost_equal(func_activation_next.detach().numpy(), layer_activation_next.detach().numpy(), decimal=6)

        loss = torch.tensor(np.random.random(len(func_activation_next)), requires_grad=True)

        func_activation_next.backward(loss)

        layer.backward(loss)
        np.testing.assert_almost_equal(layer.grad_w.detach().numpy(), layer.weight.grad.detach().numpy())
        np.testing.assert_almost_equal(layer.grad_x.detach().numpy()[:-n_pulse], activation.grad.detach().numpy())


if __name__ == '__main__':
    unittest.main()
