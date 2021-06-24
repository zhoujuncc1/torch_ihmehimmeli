import unittest

# from lambertw import LambertW0 as CLambertW0
# from lambert2 import lambertw
import sys
sys.path.append('..')
from tempcoder import GetSortedIndices, ExponentiateSortedValidSpikes, ActivateNeuronAlpha
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

# class TestLambertW0(unittest.TestCase):
#     def test_values(self):
#         vs = torch.rand(100)*100
#         lambert_c = [CLambertW0(v) for v in vs]
#         np.testing.assert_almost_equal(lambertw(vs), lambert_c, decimal=5)

class TestExponentiateSortedValidSpikes(unittest.TestCase):
    def test_exp(self):
        x = np.random.random(10)
        cmd = ["./cpp/testExponentiateSortedValidSpikes"]
        for v in x:
            cmd.append(str(v))
        decay_rate = 0.18176949150701854
        out_x = ExponentiateSortedValidSpikes(torch.tensor(x, dtype=torch.float32), layer_params)
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        expected_str = str(expected_call.stdout)[2:-2].split(' ')
        expected_arr = np.asarray([float(s) for s in expected_str])
        np.testing.assert_almost_equal(out_x,expected_arr)

class TestActivateNeuronAlpha(unittest.TestCase):
    def test_activation(self):
        m = 200
        n = 100
        weight = np.random.random((n, m))
        activation = np.random.random((2,m))
        test_index = 0
        activation_tensor = torch.tensor(activation, dtype=torch.float32)
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        sorted_indices = GetSortedIndices(activation_tensor)
        exp_activation = ExponentiateSortedValidSpikes(
            activation_tensor, layer_params)
        # threshold = 1.0
        spike_time, A_B_W, causal_set = ActivateNeuronAlpha(
                weight_tensor, activation_tensor, exp_activation, sorted_indices, layer_params)
        weight_f = "./cpp/weight_f.bin"
        activation_f = "./cpp/activation_f.bin"
        A_f = "./cpp/A_f.bin"
        B_f = "./cpp/B_f.bin"
        W_f = "./cpp/W_f.bin"
        causal_set_f = "./cpp/causal_set_f.bin"
        m = str(m)
        n = str(n)
        theta = str(threshold)

        output_file = open(activation_f, 'wb')
        activation_arr = array('d', activation[test_index])
        activation_arr.tofile(output_file)
        output_file.close()

        output_file = open(weight_f, 'wb')
        weight_arr = array('d', weight.flatten())
        weight_arr.tofile(output_file)
        output_file.close()

        cmd = ["./cpp/testActivateNeuronAlpha", weight_f, activation_f, A_f, B_f, W_f, causal_set_f, m,n,theta]
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        spike_time_expect = array('d')
        A_expect = array('d')
        B_expect = array('d')
        W_expect = array('d')
        causal_set_expect = array('B')

        f = open(activation_f, 'rb')
        spike_time_expect.frombytes(f.read())
        f.close()
        f = open(A_f, 'rb')
        A_expect.frombytes(f.read())
        f.close()
        f = open(B_f, 'rb')
        B_expect.frombytes(f.read())      
        f.close()
        f = open(W_f, 'rb')
        W_expect.frombytes(f.read())
        f.close()
        f = open(causal_set_f, 'rb')
        causal_set_expect.frombytes(f.read())
        causal_set_expect = [v==1 for v in causal_set_expect]
        f.close()

        np.testing.assert_almost_equal(spike_time[test_index].numpy().flatten(), spike_time_expect)

        np.testing.assert_almost_equal(A_B_W[0,test_index,:].numpy().flatten(), A_expect, decimal=5)

        np.testing.assert_almost_equal(A_B_W[1,test_index,:].numpy().flatten(), B_expect, decimal=5)
        np.testing.assert_almost_equal(A_B_W[2,test_index,:].numpy().flatten(), W_expect, decimal=5)
        np.testing.assert_almost_equal(causal_set[test_index].numpy().flatten()==1, causal_set_expect)

        self.assertTrue(False in causal_set.numpy(), "causal net not containt false")

class TestLayer(unittest.TestCase):

    def test_forward(self):
        m = 200
        n = 100
        n_pulse = 10

        activation = np.random.random((2,m))
        test_index = 0        
        activation_tensor = torch.tensor(activation, dtype=torch.float32)
        layer = Layer(m, n, threshold, n_pulse=n_pulse, layer_param = layer_params)
        activation_next = layer.forward(activation_tensor)

        weight_f = "./cpp/weight_f.bin"
        activation_f = "./cpp/activation_f.bin"
        A_f = "./cpp/A_f.bin"
        B_f = "./cpp/B_f.bin"
        W_f = "./cpp/W_f.bin"
        causal_set_f = "./cpp/causal_set_f.bin"
        m = str(m)
        n = str(n)
        theta = str(threshold)
        n_pulse = str(n_pulse)

        output_file = open(activation_f, 'wb')
        activation_arr = array('d', activation[test_index])
        activation_arr.tofile(output_file)
        output_file.close()

        output_file = open(weight_f, 'wb')
        weight_arr = array('d', layer.weight.flatten())
        weight_arr.tofile(output_file)
        output_file.close()

        cmd = ["./cpp/testLayer", weight_f, activation_f, A_f, B_f, W_f, causal_set_f, m,n,theta, n_pulse]
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        spike_time_expect = array('d')
        A_expect = array('d')
        B_expect = array('d')
        W_expect = array('d')
        causal_set_expect = array('B')

        f = open(activation_f, 'rb')
        spike_time_expect.frombytes(f.read())
        f.close()
        f = open(A_f, 'rb')
        A_expect.frombytes(f.read())
        f.close()
        f = open(B_f, 'rb')
        B_expect.frombytes(f.read())      
        f.close()
        f = open(W_f, 'rb')
        W_expect.frombytes(f.read())
        f.close()
        f = open(causal_set_f, 'rb')
        causal_set_expect.frombytes(f.read())
        #causal_set_expect = [v==1 for v in causal_set_expect]
        f.close()
        
        #activation_next = torch.cat([activation_next, layer.pulse[None,:].expand([len(activation_next), -1])], dim=1)
        np.testing.assert_almost_equal(activation_next[test_index].detach().numpy().flatten(), spike_time_expect[:-10], decimal=6)

        np.testing.assert_almost_equal(layer.A_B_W[0,test_index,:].detach().numpy().flatten(), A_expect, decimal=5)

        np.testing.assert_almost_equal(layer.A_B_W[1,test_index,:].detach().numpy().flatten(), B_expect, decimal=5)
        np.testing.assert_almost_equal(layer.A_B_W[2,test_index,:].detach().numpy().flatten(), W_expect, decimal=5)
        np.testing.assert_almost_equal(layer.causal_set[test_index].detach().numpy().flatten()==1, causal_set_expect)
        self.assertTrue(layer.causal_set.sum() != layer.causal_set.size, "causal net not contain false")

    def test_backward(self):
        m = 200
        n = 100
        n_pulse = 10

        activation = np.random.random((2,m))
        test_index = 0
        activation_tensor = torch.tensor(activation, dtype=torch.float32)

        layer = Layer(m, n, threshold, n_pulse=n_pulse, layer_param = layer_params)
        activation_next = layer.forward(activation_tensor)
        loss = np.random.random((2, n))
        activation_next = torch.cat([activation_next, layer.pulse[None,:].expand([len(activation_next), -1])], dim=1)

        layer.backward(torch.tensor(loss, dtype=torch.float32))

        weight_f = "./cpp/weight_f.bin"
        activation_f = "./cpp/activation_f.bin"
        A_f = "./cpp/A_f.bin"
        B_f = "./cpp/B_f.bin"
        W_f = "./cpp/W_f.bin"
        causal_set_f = "./cpp/causal_set_f.bin"
        loss_f = './cpp/loss_f.bin'
        d_w_f = './cpp/d_w_f.bin'
        d_x_f = './cpp/d_x_f.bin'
        m = str(m)
        n = str(n)
        theta = str(threshold)
        n_pulse = str(n_pulse)

        output_file = open(activation_f, 'wb')
        activation_arr = array('d', activation[test_index])
        activation_arr.tofile(output_file)
        output_file.close()

        output_file = open(weight_f, 'wb')
        weight_arr = array('d', layer.weight.flatten())
        weight_arr.tofile(output_file)
        output_file.close()

        
        output_file = open(loss_f, 'wb')
        loss_arr = array('d', loss[test_index].flatten())
        loss_arr.tofile(output_file)
        output_file.close()

        cmd = ["./cpp/testLayer", weight_f, activation_f, A_f, B_f, W_f, causal_set_f, m,n,theta, n_pulse, loss_f, d_w_f, d_x_f]
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        spike_time_expect = array('d')
        A_expect = array('d')
        B_expect = array('d')
        W_expect = array('d')
        causal_set_expect = array('B')
        grad_w_expect=array('d')
        grad_x_expect=array('d')

        f = open(activation_f, 'rb')
        spike_time_expect.frombytes(f.read())
        f.close()
        f = open(A_f, 'rb')
        A_expect.frombytes(f.read())
        f.close()
        f = open(B_f, 'rb')
        B_expect.frombytes(f.read())      
        f.close()
        f = open(W_f, 'rb')
        W_expect.frombytes(f.read())
        f.close()
        f = open(causal_set_f, 'rb')
        causal_set_expect.frombytes(f.read())
        #causal_set_expect = [v==1 for v in causal_set_expect]
        f.close()
        f = open(d_w_f, 'rb')
        grad_w_expect.frombytes(f.read())
        f.close()
        f = open(d_x_f, 'rb')
        grad_x_expect.frombytes(f.read())
        f.close()
        np.testing.assert_almost_equal(activation_next[test_index].detach().numpy(), spike_time_expect, decimal=6)

        np.testing.assert_almost_equal(layer.A_B_W[0,test_index,:].detach().numpy().flatten(), A_expect, decimal=5)

        np.testing.assert_almost_equal(layer.A_B_W[1,test_index,:].detach().numpy().flatten(), B_expect, decimal=5)
        np.testing.assert_almost_equal(layer.A_B_W[2,test_index,:].detach().numpy().flatten(), W_expect, decimal=5)
        np.testing.assert_almost_equal(layer.causal_set[test_index].detach().numpy().flatten()==1, causal_set_expect)
        np.testing.assert_almost_equal(layer.grad_w[test_index].detach().numpy().flatten(), grad_w_expect)

        np.testing.assert_almost_equal(layer.grad_x[test_index].detach().numpy(), grad_x_expect, decimal=6)

        self.assertTrue(layer.causal_set.sum() != layer.causal_set.size, "causal net not contain false")


class TestCrossEntropyLoss(unittest.TestCase):
    def test_exp(self):
        batch_size = 5
        x = np.random.random((batch_size,10))
        target = np.random.randint(0, 10, batch_size)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        target_tensor = torch.tensor(target)
        onehot_target = torch.nn.functional.one_hot(target_tensor, 10)
        loss_v, min_value = loss(x_tensor, onehot_target)
        derivative = loss_derivative(x_tensor, onehot_target, min_value)

        test_index = 0
        for test_index in range(batch_size):
            print("Test Index:", test_index)
            cmd = ["./cpp/testCrossEntropyLoss"]
            for v in x[test_index]:
                cmd.append(str(v))
            cmd.append(str(target[test_index]))

            expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
            expected_str = str(expected_call.stdout)[2:-2].split(' ')
            expected_arr = np.asarray([float(s) for s in expected_str])
            np.testing.assert_almost_equal(loss_v[test_index].numpy(),expected_arr[0], decimal=6)
            np.testing.assert_almost_equal(derivative[test_index].numpy(),expected_arr[1:])
if __name__ == '__main__':
    unittest.main()
