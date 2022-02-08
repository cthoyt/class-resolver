# -*- coding: utf-8 -*-

"""Tests for the optuna contribution module."""

import unittest

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # pragma: no cover


@unittest.skipUnless(optuna, "Can not test optuna contrib without ``pip install optuna``.")
class TestTorch(unittest.TestCase):
    """Test for the optuna contribution module."""

    def test_sampler(self):
        """Tests for the sampler resolver."""
        from optuna.samplers import RandomSampler, TPESampler

        from class_resolver.contrib.optuna import sampler_resolver

        self.assertEqual(RandomSampler, sampler_resolver.lookup("random"))
        self.assertEqual(TPESampler, sampler_resolver.lookup("tpe"))
        self.assertEqual(TPESampler, sampler_resolver.lookup(None))

    def test_pruner(self):
        """Tests for the pruner resolver."""
        from optuna.pruners import MedianPruner, PatientPruner

        from class_resolver.contrib.optuna import pruner_resolver

        self.assertEqual(PatientPruner, pruner_resolver.lookup("patient"))
        self.assertEqual(MedianPruner, pruner_resolver.lookup("median"))
        self.assertEqual(MedianPruner, pruner_resolver.lookup(None))
