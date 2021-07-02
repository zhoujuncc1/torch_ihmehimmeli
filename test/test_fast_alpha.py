import unittest

import sys
sys.path.append('..')
from tempcoder import *
from Layer import Layer, loss,loss_derivative
import subprocess
import os
import numpy as np
from array import array
import math
import torch
from Layer import LayerParam

n_pulse = 10
kNoSpike=1000
threshold = 1.16732
decay_rate = 0.18176949150701854
penalty_no_spike = 48.3748
pulse_init_multiplier = 7.83912
nopulse_init_multiplier = -0.275419
input_range = (0,1)
layer_params = LayerParam(kNoSpike, decay_rate, threshold, penalty_no_spike, pulse_init_multiplier, nopulse_init_multiplier, input_range, dtype=torch.float)

class TestActivateNeuronAlpha2(unittest.TestCase):
    def test_activation(self):
        m = 200
        n = 100
        weight = np.random.random((n, m))
        activation = np.random.random((2,m))
        test_index = 0
        activation_tensor = torch.tensor(activation, dtype=torch.float32)
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        sorted_activations, sorted_indices = GetSortedSpikes(activation_tensor)
        exp_activation = ExponentiateSortedValidSpikes(
            activation_tensor, layer_params)
        sorted_exp_activation = ExponentiateSortedValidSpikes(
            sorted_activations, layer_params)
        # threshold = 1.0
        spike_time, A_B_W, causal_set = ActivateNeuronAlpha(
                weight_tensor, activation_tensor, exp_activation, sorted_indices, layer_params)
        spike_time2, A_B_W2, causal_set2 = ActivateNeuronAlpha_fast(
                weight_tensor, activation_tensor, sorted_activations, sorted_exp_activation, sorted_indices, layer_params)
        np.testing.assert_almost_equal(causal_set.numpy(), causal_set2.numpy())

        np.testing.assert_almost_equal(spike_time.numpy(), spike_time2.numpy())

        np.testing.assert_almost_equal(A_B_W.numpy(), A_B_W2.numpy(), decimal=5)

        self.assertTrue(False in causal_set.numpy(), "causal net not containt false")


if __name__ == '__main__':
    unittest.main()
