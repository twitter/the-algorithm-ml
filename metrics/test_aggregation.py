import torch
import unittest
from aggregation import StableMean


class TestStableMean(unittest.TestCase):

    def setUp(self):
        self.metric = StableMean()

    def test_compute_empty(self):
        result = self.metric.compute()
        self.assertEqual(result, torch.tensor(0.0))

    def test_compute_single_value(self):
        self.metric.update(torch.tensor(1.0))
        result = self.metric.compute()
        self.assertEqual(result, torch.tensor(1.0))

    def test_compute_weighted_single_value(self):
        self.metric.update(torch.tensor(1.0), weight=torch.tensor(2.0))
        result = self.metric.compute()
        self.assertEqual(result, torch.tensor(1.0))

    def test_compute_multiple_values(self):
        self.metric.update(torch.tensor(1.0))
        self.metric.update(torch.tensor(2.0))
        self.metric.update(torch.tensor(3.0))
        result = self.metric.compute()
        self.assertEqual(result, torch.tensor(2.0))

    def test_compute_weighted_multiple_values(self):
        self.metric.update(torch.tensor(1.0), weight=torch.tensor(1.0))
        self.metric.update(torch.tensor(2.0), weight=torch.tensor(2.0))
        self.metric.update(torch.tensor(3.0), weight=torch.tensor(3.0))
        result = self.metric.compute()
        print(f"get=  {result.item()} but expected= 2.1666666667")
        self.assertAlmostEqual(result.item(), 2.1666666667, places=0)


if '__name__' == '__main__':
    unittest.main()
