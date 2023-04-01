import unittest
import torch
from tml.optimizers.config import LearningRate, OptimizerConfig
from .optimizer import compute_lr, LRShim, get_optimizer_class, build_optimizer


class TestComputeLR(unittest.TestCase):
    def test_constant_lr(self):
        lr_config = LearningRate(constant=0.1)
        lr = compute_lr(lr_config, step=0)
        self.assertAlmostEqual(lr, 0.1)

    def test_piecewise_constant_lr(self):
        lr_config = LearningRate(piecewise_constant={"learning_rate_boundaries": [10, 20], "learning_rate_values": [0.1, 0.01, 0.001]})
        lr = compute_lr(lr_config, step=5)
        self.assertAlmostEqual(lr, 0.1)
        lr = compute_lr(lr_config, step=15)
        self.assertAlmostEqual(lr, 0.01)
        lr = compute_lr(lr_config, step=25)
        self.assertAlmostEqual(lr, 0.001)


class TestLRShim(unittest.TestCase):
    def setUp(self):
        self.optimizer = torch.optim.SGD([torch.randn(10, 10)], lr=0.1)
        self.lr_dict = {"ALL_PARAMS": LearningRate(constant=0.1)}

    def test_get_lr(self):
        lr_scheduler = LRShim(self.optimizer, self.lr_dict)
        lr = lr_scheduler.get_lr()
        self.assertAlmostEqual(lr, [0.1])


class TestBuildOptimizer(unittest.TestCase):
    def test_build_optimizer(self):
        model = torch.nn.Linear(10, 1)
        optimizer_config = OptimizerConfig(sgd={"lr": 0.1})
        optimizer, scheduler = build_optimizer(model, optimizer_config)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertIsInstance(scheduler, LRShim)


if __name__ == "__main__":
    unittest.main()
