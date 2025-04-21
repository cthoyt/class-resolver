"""Class resolvers for Optuna."""

from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler

from ..api import ClassResolver

__all__ = [
    "pruner_resolver",
    "sampler_resolver",
]

sampler_resolver = ClassResolver.from_subclasses(
    BaseSampler,  # type:ignore[type-abstract]
    default=TPESampler,
    suffix="Sampler",
    exclude_private=False,
    location="class_resolver.contrib.optuna.sampler_resolver",
)
"""A resolver for :class:`optuna.samplers.BaseSampler` subclasses.

Building on the simple example from the Optuna website's homepage,
you can parametrize :func:`optuna.create_study` with a sampler
instantiated with :mod:`class_resolver`.

.. code-block:: python

    import optuna
    from class_resolver import Hint
    from class_resolver.contrib.optuna import sampler_resolver
    from optuna.sampler import BaseSampler

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2

    def optimize_study(sampler: Hint[BaseSampler] = None):
        study = optuna.create_study(
            sampler=sampler_resolver.make(sampler),
        )
        study.optimize(objective, n_trials=100)
        return study

    study = optimize_study(sampler="TPE")
    study.best_params  # E.g. {'x': 2.002108042}
"""

pruner_resolver = ClassResolver.from_subclasses(
    BasePruner,  # type:ignore[type-abstract]
    default=MedianPruner,
    suffix="Pruner",
    exclude_private=False,
    location="class_resolver.contrib.optuna.pruner_resolver",
)
"""A resolver for :class:`optuna.pruners.BasePruner` subclasses.

Building on the simple example from the Optuna website's homepage,
you can parametrize :func:`optuna.create_study` with a pruner
instantiated with :mod:`class_resolver`.

.. code-block:: python

    import optuna
    from class_resolver import Hint
    from class_resolver.contrib.optuna import pruner_resolver
    from optuna.pruner import BasePruner

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2

    def optimize_study(pruner: Hint[BasePruner] = None):
        study = optuna.create_study(
            pruner=pruner_resolver.make(pruner),
        )
        study.optimize(objective, n_trials=100)
        return study

    study = optimize_study(pruner="median")
    study.best_params  # E.g. {'x': 2.002108042}
"""

# TODO figure out why this isn't auto-registered
pruner_resolver.register(MedianPruner)
