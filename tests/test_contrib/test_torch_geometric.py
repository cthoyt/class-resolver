# -*- coding: utf-8 -*-

"""Tests for the torch-geometric contribution module."""

import unittest

try:
    import torch_geometric
except torch_geometric:  # pragma: no cover
    torch = None  # pragma: no cover


@unittest.skipUnless(
    torch_geometric, "Can not test torch_geometric contrib without ``pip install torch``."
)
class TestTorch(unittest.TestCase):
    """Test for the torch-geometric contribution module."""

    def test_message_passing(self):
        """Tests for the message passing resolver."""
        from torch_geometric.nn.conv import MessagePassing, SimpleConv

        from class_resolver.contrib.torch_geometric import message_passing_resolver

        self.assertEqual(SimpleConv, message_passing_resolver.lookup("simple"))
        self.assertEqual(SimpleConv, message_passing_resolver.lookup(None))
