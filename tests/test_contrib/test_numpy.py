"""Tests for the torch contribution module."""

import unittest

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None  # pragma: no cover


@unittest.skipUnless(numpy, "Can not test numpy contrib without ``pip install numpy``.")
class TestNumpy(unittest.TestCase):
    """Test for the numpy contribution module."""

    def test_activation(self) -> None:
        """Tests for the aggregation resolver."""
        import numpy as np

        from class_resolver.contrib.numpy import aggregation_resolver

        self.assertEqual(np.min, aggregation_resolver.lookup("min"))
        self.assertEqual(np.min, aggregation_resolver.lookup(np.min))
        self.assertEqual(np.mean, aggregation_resolver.lookup(None))
