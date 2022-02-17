# -*- coding: utf-8 -*-

"""An argument checker."""

import inspect
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import get_args, get_origin

from class_resolver.api import ClassResolver
from class_resolver.base import BaseResolver
from class_resolver.func import FunctionResolver
from class_resolver.utils import Hint, HintOrType, HintType, OptionalKwargs

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
    if hint == Hint[cls]:  # type: ignore
        return True
    if hint == HintType[cls]:  # type: ignore
        return True
    if hint == HintOrType[cls]:  # type: ignore
        return True
    return False


SIMPLE_TYPES = {float, int, bool, str, type(None)}


def _is_simple_union(u) -> bool:
    return all(arg in SIMPLE_TYPES for arg in get_args(u))


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

    def __init__(self, func, key, text):
        self.func = func
        self.key = key
        self.text = text

    def __str__(self):
        return f"{self.func} {self.key}: {self.text}"


class Metaresolver:
    """A resolver of resolvers."""

    def __init__(
        self,
        resolvers: Iterable[ClassResolver],
        extras: Optional[Mapping[str, BaseResolver]] = None,
    ):
        """Instantiate a meta-resolver.

        :param resolvers: A set of resolvers to index for checking kwargs
        """
        self.names = {resolver.suffix: resolver for resolver in resolvers}
        if extras:
            self.names.update(extras)

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
            if key == "kwargs":
                continue
            annotation = parameter.annotation
            next_resolver = self.names.get(key)
            value = kwargs.get(key)
            if next_resolver is not None:
                if isinstance(next_resolver, FunctionResolver):
                    continue
                if not is_hint(annotation, next_resolver.base):
                    raise ArgumentError(
                        func,
                        key,
                        f"has bad annotation {annotation} wrt resolver {next_resolver}",
                    )
                if value is None:
                    raise ArgumentError(func, key, f"is missing from the arguments")
                self.check_kwargs(
                    next_resolver.lookup(value),
                    kwargs.get(related_key),
                )
            else:
                if value is None:
                    if parameter.default is parameter.empty:
                        raise ArgumentError(
                            func,
                            key,
                            f"no default, value not given. Signature: {inspect.signature(func)}",
                        )
                else:
                    origin = get_origin(annotation)
                    if origin is Union:
                        if not _is_simple_union(annotation):
                            raise ArgumentError(
                                func, key, f"has inappropriate annotation: {annotation}"
                            )
                        else:
                            args = get_args(annotation)
                            if isinstance(value, args):
                                pass
                            else:
                                raise ArgumentError(
                                    func,
                                    key,
                                    f"value {value} does not match annotation {annotation}",
                                )
                    elif origin is None:
                        try:
                            instance_check = isinstance(value, annotation)
                        except TypeError:
                            raise ArgumentError(
                                func, key, f"invalid annotation {annotation} ({type(annotation)})"
                            ) from None
                        if instance_check:
                            pass
                        elif annotation == float and isinstance(value, int):
                            pass  # log a warning?
                        else:
                            raise ArgumentError(
                                func, key, f"{value} mismatched annotation {annotation}"
                            )
                    else:
                        raise ArgumentError(func, key, f"unhandled origin {origin}")
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


def _main():
    import json

    from pykeen.datasets import dataset_resolver
    from pykeen.experiments.cli import HERE
    from pykeen.losses import loss_resolver
    from pykeen.models import model_resolver
    from pykeen.nn.emb import constrainer_resolver, normalizer_resolver
    from pykeen.nn.init import initializer_resolver
    from pykeen.pipeline import pipeline
    from pykeen.regularizers import regularizer_resolver

    print(HERE)
    r = Metaresolver(
        [
            model_resolver,
            regularizer_resolver,
            loss_resolver,
            dataset_resolver,
            constrainer_resolver,
            normalizer_resolver,
            initializer_resolver,
        ],
        extras={
            "entity_initializer": initializer_resolver,
            "entity_normalizer": normalizer_resolver,
            "entity_constrainer": constrainer_resolver,
            "relation_initializer": initializer_resolver,
            "relation_normalizer": normalizer_resolver,
            "relation_constrainer": constrainer_resolver,
        },
    )
    for path in HERE.glob("*/*.json"):
        data = json.loads(path.read_text())
        kwargs = data["pipeline"]
        r.check_kwargs(pipeline, kwargs)


if __name__ == "__main__":
    _main()
