import unittest

from lambertw import LambertW0 as CLambertW0
from lambert2 import lambertw
from tempcoder import GetSortedIndices, ExponentiateSortedValidSpikes, ActivateNeuronAlpha
from Layer import Layer, loss,loss_derivative
import subprocess
import os
import numpy as np
from array import array
import math

decay_rate = 0.18176949150701854
kNoSpike = 1000

class TestLambertW0(unittest.TestCase):
    def test_values(self):
        vs = np.random.random(10000)*1000
        for v in vs:
            self.assertAlmostEqual(CLambertW0(v),lambertw(v))


# class TestArgSort(unittest.TestCase):
#     def test_argsort(self):
#         x = np.asarray([2.5, 100, 0.5], dtype=np.float64)
#         expect = [1, 2, 0]
#         self.assertEqual(list(GetSortedIndices(x)), expect)

#         x = np.asarray([2, 4, 3], dtype=np.float64)
#         expect = [0, 2, 1]
#         self.assertEqual(list(GetSortedIndices(x)), expect)

#         x = np.asarray([3, 1, 0, 4], dtype=np.float64)
#         expect = [2, 1, 0, 3]
#         self.assertEqual(list(GetSortedIndices(x)), expect)


class TestExponentiateSortedValidSpikes(unittest.TestCase):
    def test_exp(self):
        x = np.random.random(10)
        cmd = ["./cpp/testExponentiateSortedValidSpikes"]
        for v in x:
            cmd.append(str(v))
        sorted_idx = GetSortedIndices(x)
        decay_rate = 0.18176949150701854
        out_x = ExponentiateSortedValidSpikes(x, sorted_idx, decay_rate)
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        expected_str = str(expected_call.stdout)[2:-2].split(' ')
        expected_arr = np.asarray([float(s) for s in expected_str])
        np.testing.assert_almost_equal(out_x,expected_arr)


class TestActivateNeuronAlpha(unittest.TestCase):
    def test_activation(self):
        os.chdir('/mnt/d/workspace/ihmehimmeli/python')
        m = 200
        n = 100
        weight = np.random.random((n, m))
        activation = np.random.random(m)
        sorted_indices = GetSortedIndices(activation)
        exp_activation = ExponentiateSortedValidSpikes(
            activation, sorted_indices, decay_rate)
        spike_time = []
        A = []
        B = []
        W = []
        causal_set = []
        threshold = 1.0
        for i in range(n):
            t, a, b, w, c = ActivateNeuronAlpha(
                weight[i], activation, exp_activation, sorted_indices, threshold)
            spike_time.append(t)
            A.append(a)
            B.append(b)
            W.append(w)
            causal_set.extend(c)

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
        activation_arr = array('d', activation)
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

        np.testing.assert_almost_equal(spike_time, spike_time_expect, decimal=5)

        np.testing.assert_almost_equal(A, A_expect)

        np.testing.assert_almost_equal(B, B_expect)
        np.testing.assert_almost_equal(W, W_expect)
        np.testing.assert_almost_equal(causal_set, causal_set_expect)

        self.assertTrue(False in causal_set, "causal net not containt false")

class TestLayerForward(unittest.TestCase):
    def test_activation(self):
        os.chdir('/mnt/d/workspace/ihmehimmeli/python')
        m = 200
        n = 100
        activation = np.random.random(m)
        
        threshold=1.0
        n_pulse=2
        layer = Layer(m, n, threshold, n_pulse=n_pulse, input_range = (min(activation), max(activation)))
        activation_next = layer.forward(activation)

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
        activation_arr = array('d', activation)
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
        
        activation_next = np.concatenate([activation_next, layer.pulse])
        np.testing.assert_almost_equal(activation_next, spike_time_expect, decimal=6)

        np.testing.assert_almost_equal(layer.A, A_expect)

        np.testing.assert_almost_equal(layer.B, B_expect)
        np.testing.assert_almost_equal(layer.W, W_expect)
        np.testing.assert_almost_equal(layer.causal_set.flatten(), causal_set_expect)

        self.assertTrue(layer.causal_set.sum() != layer.causal_set.size, "causal net not contain false")


class TestCrossEntropyLoss(unittest.TestCase):
    def test_exp(self):
        x = np.random.random(10)
        target = np.random.randint(0, 10, 1)[0]
        cmd = ["./cpp/testCrossEntropyLoss"]
        for v in x:
            cmd.append(str(v))
        cmd.append(str(target))
        loss_v = loss(x, target)
        derivative = loss_derivative(x, target)
        expected_call = subprocess.run(cmd, stdout=subprocess.PIPE)
        expected_str = str(expected_call.stdout)[2:-2].split(' ')
        expected_arr = np.asarray([float(s) for s in expected_str])
        np.testing.assert_almost_equal(loss_v,expected_arr[0])
        np.testing.assert_almost_equal(derivative,expected_arr[1:])

        
if __name__ == '__main__':
    unittest.main()
