# -*- coding: utf-8 -*-

"""Tests for the torch contribution module."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch, "Can not test torch contrib without ``pip install torch``.")
class TestTorch(unittest.TestCase):
    """Test for the torch contribution module."""

    def test_activation(self):
        """Tests for the activation resolver."""
        from class_resolver.contrib.torch import activation_resolver
        from torch.nn import ReLU, Softplus
        self.assertEqual(Softplus, activation_resolver.lookup("softplus"))
        self.assertEqual(ReLU, activation_resolver.lookup("relu"))
        self.assertEqual(ReLU, activation_resolver.lookup())

    def test_optimizer(self):
        """Tests for the optimizer resolver."""
        from class_resolver.contrib.torch import optimizer_resolver
        from torch.optim import Adam, Adagrad
        self.assertEqual(Adagrad, optimizer_resolver.lookup("adagrad"))
        self.assertEqual(Adam, optimizer_resolver.lookup("adam"))
        self.assertEqual(Adam, optimizer_resolver.lookup())

    def test_initializer(self):
        """Tests for the initializer function resolver."""
        from class_resolver.contrib.torch import initializer_resolver
        from torch.nn.init import xavier_normal_
        self.assertEqual(xavier_normal_, initializer_resolver.lookup("xavier_normalize_"))
