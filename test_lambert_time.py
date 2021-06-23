import unittest
import lambert2
import lambertw
import time
import torch
import numpy as np
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


if __name__ == '__main__':
    unittest.main()
