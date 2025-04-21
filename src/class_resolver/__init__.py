"""The :mod:`class_resolver` package helps you look up related classes and functions to parametrize your code.

Getting Started
===============

An example might be when you have several implementations of the same algorithm (e.g.,
fast, greedy one and a slow but correct one) and want to write a function
``run_algorithm`` that can be easily switched between them with a string corresponding
to the name of the implementation

.. code-block:: python

    from class_resolver import ClassResolver, Hint


    class Algorithm:
        def run(self, data): ...


    class GreedyAlgorithm(Algorithm): ...


    class CorrectAlgorithm(Algorithm): ...


    algorithm_resolver = ClassResolver.from_subclasses(Algorithm)


    def run_algorithm(data, *, algorithm: Hint[Algorithm] = "greedy"):
        algorithm = algorithm_resolver.make(algorithm)
        return algorithm.run(data)

Note that the string keys in the class resolver are ``"greedy"`` for ``GreedyAlgorithm``
and ``"correct"`` for ``CorrectAlgorithm``. That's because it knows the name of the base
class is ``Algorithm`` and it can infer what you mean.

Pass a string, class, or instance
=================================

The ``Hint[Algorithm]`` signifies that the ``algorithm_resolver.make(...)`` function is
very powerful. You can pass it one of the following:

1. A string, like ``"GreedyAlgorithm"`` or ``"Greedy"`` or ``"greedy"`` and it deals
   with casing and the suffix
2. A class like ``GreedyAlgorithm``, ``CorrectAlgorithm``, or any potential subclass of
   ``Algorithm``
3. An instance of an ``Algorithm``, in case you have one pre-defined.
4. ``None``, if you defined the ``default=...` when you called
   ``ClassResolver.from_subclasses`` like in ``ClassResolver.from_subclasses(Algorithm,
   default=GreedyAlgorithm``.
"""

from .api import ClassResolver, KeywordArgumentError, Resolver, UnexpectedKeywordError, get_cls
from .base import (
    BaseResolver,
    RegistrationError,
    RegistrationNameConflict,
    RegistrationSynonymConflict,
)
from .docs import ResolverKey, update_docstring_with_resolver_keys
from .func import FunctionResolver
from .utils import (
    Hint,
    HintOrType,
    HintType,
    InstOrType,
    Lookup,
    LookupOrType,
    LookupType,
    OneOrManyHintOrType,
    OneOrManyOptionalKwargs,
    OptionalKwargs,
    get_subclasses,
    normalize_string,
)
from .version import VERSION

__all__ = [
    "VERSION",
    "BaseResolver",
    "ClassResolver",
    "FunctionResolver",
    "Hint",
    "HintOrType",
    "HintType",
    "InstOrType",
    "KeywordArgumentError",
    "Lookup",
    "LookupOrType",
    "LookupType",
    "OneOrManyHintOrType",
    "OneOrManyOptionalKwargs",
    "OptionalKwargs",
    "RegistrationError",
    "RegistrationNameConflict",
    "RegistrationSynonymConflict",
    "Resolver",
    "ResolverKey",
    "UnexpectedKeywordError",
    "get_cls",
    "get_subclasses",
    "normalize_string",
    "update_docstring_with_resolver_keys",
]
