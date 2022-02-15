# -*- coding: utf-8 -*-

"""An argument checker."""

import inspect
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, TypeVar

from .api import ClassResolver
from .utils import Hint, OptionalKwargs

__all__ = [
    "is_hint",
    "Metaresolver",
]

X = TypeVar("X")


def is_hint(hint: Any, cls: Type[X]) -> bool:
    """Check if the hint is applicable to the given class.

    :param hint: The hint type
    :param cls: The class to check
    :returns: If the hint is appropriate for the class
    """
    if not isinstance(cls, type):
        raise TypeError
    return hint == Hint[cls]  # type: ignore


class Metaresolver:
    """A resolver of resolvers."""

    def __init__(self, resolvers: Iterable[ClassResolver]):
        """Instantiate a meta-resolver.

        :param resolvers: A set of resolvers to index for checking kwargs
        """
        self.resolvers: Mapping[Type, ClassResolver] = {
            resolver.base: resolver for resolver in resolvers
        }
        self.names = {resolver.suffix: resolver for cls, resolver in self.resolvers.items()}

    def check_kwargs(self, func: Callable, kwargs: OptionalKwargs = None) -> bool:
        """Check the appropriate of the kwargs with a given function.

        :param func: A function or class to check
        :param kwargs: The keyword arguments to pass to the function
        :returns: True if there are no issues, raises if there are.
        """
        if kwargs is None:
            kwargs = {}
        for key, parameter, related_key in _iter_params(func):
            annotation = parameter.annotation
            next_resolver = self.names.get(key)
            if next_resolver is not None:
                if not is_hint(annotation, next_resolver.base):
                    raise TypeError(
                        f"{key} has bad annotation {annotation} wrt resolver {next_resolver}"
                    )
                query = kwargs.get(key)
                if query is None:
                    raise KeyError
                self.check_kwargs(
                    next_resolver.lookup(query),
                    kwargs.get(related_key),
                )
            else:
                if key not in kwargs:
                    if parameter.default is parameter.empty:
                        raise ValueError(f"{key} without default not given")
                else:
                    try:
                        instance_flag = isinstance(kwargs[key], parameter.annotation)
                    except TypeError:
                        raise TypeError(f"{key} {kwargs[key]} {parameter.annotation}") from None
                    if not instance_flag:
                        raise ValueError
        return True


def _iter_params(
    func,
) -> Iterable[Tuple[str, inspect.Parameter, str, Optional[inspect.Parameter]]]:
    parameters = inspect.signature(func).parameters
    kwarg_map = {}
    for key in parameters:
        related_key = f"{key}_kwargs"
        if related_key in parameters:
            kwarg_map[related_key] = key
    for key in parameters:
        if key in kwarg_map:
            continue
        yield key, parameters[key], f"{key}_kwargs"
