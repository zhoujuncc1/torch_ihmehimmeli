import unittest
import sys
sys.path.append('..')
import lambert2
import lambertw
import time
import torch
import numpy as np
from tempcoder import *


class TestLambertWTime(unittest.TestCase):
    def test_time(self):
        device = "cuda"
        vs = torch.rand(100).to(device)

        tic = time.perf_counter()
        out1 = lambert2.lambertw(vs)
        toc = time.perf_counter()
        lambert2_t = toc - tic
        print(f"lambert2_t: {lambert2_t:0.4f} seconds")

        tic = time.perf_counter()
        out2 = lambertw.lambertw(vs)
        toc = time.perf_counter()
        lambert_t = toc - tic

        print(f"lambert_t: {lambert_t:0.4f} seconds")
        self.assertLess(lambert_t, lambert2_t)
        np.testing.assert_almost_equal(out1.cpu().numpy(),out2.cpu().numpy())


from Layer import LayerParam
n_pulse = 10
kNoSpike=1000
threshold = 1.16732
decay_rate = 0.18176949150701854
penalty_no_spike = 48.3748
pulse_init_multiplier = 7.83912
nopulse_init_multiplier = -0.275419
input_range = (0,1)

class TestAlphaTime(unittest.TestCase):
    def test_time(self):
        device = "cuda"
        cuda_layer_param = LayerParam(kNoSpike, decay_rate, threshold, penalty_no_spike, pulse_init_multiplier, nopulse_init_multiplier, input_range, dtype=torch.float, device=device)

        m = 200
        n = 100
        weight_tensor = torch.rand((n, m)).to(device)
        activation_tensor = torch.rand((2,m)).to(device)
        sorted_activations, sorted_indices = GetSortedSpikes(activation_tensor)
        exp_activation = ExponentiateSortedValidSpikes(
            activation_tensor, cuda_layer_param)
        sorted_exp_activation = ExponentiateSortedValidSpikes(
            sorted_activations, cuda_layer_param)
        # threshold = 1.0
        tic = time.perf_counter()
        spike_time, A_B_W, causal_set = ActivateNeuronAlpha(
                weight_tensor, activation_tensor, exp_activation, sorted_indices, cuda_layer_param)
        toc = time.perf_counter()
        alpha_t = toc - tic
        print(f"alpha_t: {alpha_t:0.4f} seconds")

        tic = time.perf_counter()
        spike_time2, A_B_W2, causal_set2 = ActivateNeuronAlpha_fast(
                weight_tensor, activation_tensor, sorted_activations, sorted_exp_activation, sorted_indices, cuda_layer_param)
        toc = time.perf_counter()
        alpha2_t = toc - tic
        print(f"alpha2_t: {alpha2_t:0.4f} seconds")
        self.assertLess(alpha2_t, alpha_t)

if __name__ == '__main__':
    unittest.main()
