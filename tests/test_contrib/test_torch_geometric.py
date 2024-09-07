"""Tests for the torch-geometric contribution module."""

import unittest

try:
    import torch_geometric
except ImportError:  # pragma: no cover
    torch_geometric = None  # pragma: no cover


@unittest.skipUnless(
    torch_geometric,
    "Can not test torch_geometric contrib without ``pip install torch torch-geometric``.",
)
class TestTorch(unittest.TestCase):
    """Test for the torch-geometric contribution module."""

    def test_message_passing(self) -> None:
        """Tests for the message passing resolver."""
        from torch_geometric.nn.conv import SimpleConv

        from class_resolver.contrib.torch_geometric import message_passing_resolver

        self.assertEqual(SimpleConv, message_passing_resolver.lookup("simple"))
        self.assertEqual(SimpleConv, message_passing_resolver.lookup(None))

    def test_aggregation(self) -> None:
        """Test the aggregation resolver."""
        import torch

        from class_resolver.contrib.torch_geometric import aggregation_resolver

        # Feature matrix holding 1000 elements with 64 features each:
        x = torch.randn(1000, 64)

        # Randomly assign elements to 100 sets:
        index = torch.randint(0, 100, (1000,))

        for cls in aggregation_resolver:
            aggr = cls()
            output = aggr(x, index)
            self.assertEqual((100, 64), tuple(output.shape))
