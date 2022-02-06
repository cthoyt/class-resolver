# -*- coding: utf-8 -*-

"""Class resolvers for Optuna."""

from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler

from ..api import ClassResolver

__all__ = [
    "sampler_resolver",
    "pruner_resolver",
]

sampler_resolver = ClassResolver.from_subclasses(
    BaseSampler,
    default=TPESampler,
    suffix="Sampler",
    exclude_private=False,
)

pruner_resolver = ClassResolver.from_subclasses(
    BasePruner,
    default=MedianPruner,
    suffix="Pruner",
    exclude_private=False,
)
# TODO figure out why this isn't auto-registered
pruner_resolver.register(MedianPruner)
