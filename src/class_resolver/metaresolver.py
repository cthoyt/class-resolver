# -*- coding: utf-8 -*-

"""An argument checker."""

import inspect
from typing import Any, Callable, Iterable, Mapping, Tuple, Type, TypeVar, Union

from typing_extensions import get_args, get_origin

from .api import ClassResolver
from .utils import Hint, OptionalKwargs

__all__ = [
    "check_kwargs",
    "is_hint",
    "Metaresolver",
]

X = TypeVar("X")


def is_hint(hint: Any, cls: Type[X]) -> bool:
    """Check if the hint is applicable to the given class.

    :param hint: The hint type
    :param cls: The class to check
    :returns: If the hint is appropriate for the class
    :raises TypeError: If the ``cls`` is not a type
    """
    if not isinstance(cls, type):
        raise TypeError
    return hint == Hint[cls]  # type: ignore


SIMPLE_TYPES = {float, int, bool, str, type(None)}


def check_kwargs(
    func: Callable,
    kwargs: OptionalKwargs = None,
    *,
    resolvers: Iterable[ClassResolver],
) -> bool:
    """Check the appropriate of the kwargs with a given function.

    :param func: A function or class to check
    :param kwargs: The keyword arguments to pass to the function
    :param resolvers: A set of resolvers to index for checking kwargs
    :returns: True if there are no issues, raises if there are.
    """
    return Metaresolver(resolvers).check_kwargs(func, kwargs)


class ArgumentError(TypeError):
    """A custom argument error."""


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
        :raises ArgumentError: If there is an error in the kwargs
        """
        if kwargs is None:
            kwargs = {}
        for key, parameter, related_key in _iter_params(func):
            annotation = parameter.annotation
            next_resolver = self.names.get(key)
            value = kwargs.get(key)
            if next_resolver is not None:
                if not is_hint(annotation, next_resolver.base):
                    raise ArgumentError(
                        f"{key} has bad annotation {annotation} wrt resolver {next_resolver}"
                    )
                if value is None:
                    raise ArgumentError(f"{key} is missing from the arguments")
                self.check_kwargs(
                    next_resolver.lookup(value),
                    kwargs.get(related_key),
                )
            else:
                if value is None:
                    if parameter.default is parameter.empty:
                        raise ArgumentError(f"{key} without default not given")
                else:
                    origin = get_origin(annotation)
                    if origin is Union:
                        args = get_args(annotation)
                        if all(arg in SIMPLE_TYPES for arg in args):
                            if isinstance(value, args):
                                pass
                            else:
                                raise ArgumentError
                        else:
                            raise ArgumentError
                    elif origin is None:
                        if isinstance(value, annotation):
                            pass
                        else:
                            raise ArgumentError
                    else:
                        raise ArgumentError(f"origin: {origin}")
        return True


def _iter_params(
    func,
) -> Iterable[Tuple[str, inspect.Parameter, str]]:
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
