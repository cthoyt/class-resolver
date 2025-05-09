"""Tests for the torch contribution module."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Can not test torch contrib without ``pip install torch``.",
)
class TestTorch(unittest.TestCase):
    """Test for the torch contribution module."""

    def test_activation(self) -> None:
        """Tests for the activation resolver."""
        from torch.nn import ReLU, Softplus

        from class_resolver.contrib.torch import activation_resolver

        self.assertEqual(Softplus, activation_resolver.lookup("softplus"))
        self.assertEqual(ReLU, activation_resolver.lookup("relu"))
        self.assertEqual(ReLU, activation_resolver.lookup(None))

    def test_margin_activation(self) -> None:
        """Tests for the margin activation resolver."""
        from torch.nn import ReLU, Softplus

        from class_resolver.contrib.torch import margin_activation_resolver

        self.assertEqual(Softplus, margin_activation_resolver.lookup("softplus"))
        self.assertEqual(Softplus, margin_activation_resolver.lookup("soft"))
        self.assertEqual(ReLU, margin_activation_resolver.lookup("relu"))
        self.assertEqual(ReLU, margin_activation_resolver.lookup("hard"))
        self.assertEqual(ReLU, margin_activation_resolver.lookup(None))

    def test_optimizer(self) -> None:
        """Tests for the optimizer resolver."""
        from torch.optim import Adagrad, Adam

        from class_resolver.contrib.torch import optimizer_resolver

        self.assertEqual(Adagrad, optimizer_resolver.lookup("adagrad"))
        self.assertEqual(Adam, optimizer_resolver.lookup("adam"))
        self.assertEqual(Adam, optimizer_resolver.lookup(None))

    def test_initializer(self) -> None:
        """Tests for the initializer function resolver."""
        from torch.nn.init import xavier_normal_

        from class_resolver.contrib.torch import initializer_resolver

        self.assertEqual(xavier_normal_, initializer_resolver.lookup("xavier_normal_"))
        self.assertEqual(xavier_normal_, initializer_resolver.lookup("xavier_normal"))
        self.assertEqual(xavier_normal_, initializer_resolver.lookup("xaviernormal"))

    def test_lr(self) -> None:
        """Tests for the learning rate scheduler."""
        from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, ReduceLROnPlateau

        from class_resolver.contrib.torch import lr_scheduler_resolver

        self.assertEqual(LambdaLR, lr_scheduler_resolver.lookup("lambda"))
        self.assertEqual(LambdaLR, lr_scheduler_resolver.lookup("lambdalr"))
        self.assertEqual(ReduceLROnPlateau, lr_scheduler_resolver.lookup("reducelronplateau"))
        self.assertEqual(ExponentialLR, lr_scheduler_resolver.lookup("exponential"))
        self.assertEqual(ExponentialLR, lr_scheduler_resolver.lookup("exponentiallr"))
        self.assertEqual(ExponentialLR, lr_scheduler_resolver.lookup(None))
