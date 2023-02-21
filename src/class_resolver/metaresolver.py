# -*- coding: utf-8 -*-

"""An argument checker."""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
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
    """Check if the arguments to type U are all simple."""
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

    def __init__(self, func, parameter, text):
        self.func = func
        self.parameter = parameter
        self.text = text

    def __str__(self):
        return f"{self.func} {self.parameter.name}: {self.text}"


class Metaresolver:
    """A resolver of resolvers."""

    parameter_name_to_resolver: Dict[str, BaseResolver]

    def __init__(
        self,
        resolvers: Iterable[ClassResolver],
        extras: Optional[Mapping[str, BaseResolver]] = None,
    ):
        """Instantiate a meta-resolver.

        :param resolvers: A set of resolvers to index for checking kwargs
        :param extras: A dictionary of additional resolvers, e.g. class resolvers
            that don't have suffixes or function resolvers given explicitly with
            parameter names
        """
        self.parameter_name_to_resolver = {resolver.suffix: resolver for resolver in resolvers}
        if extras:
            self.parameter_name_to_resolver.update(extras)

    def check_kwarg(self, func: Callable, parameter: inspect.Parameter, related_key: str, kwargs):
        annotation = parameter.annotation
        value_not_given = parameter.name not in kwargs
        value = kwargs.get(parameter.name)

        resolver = self.parameter_name_to_resolver.get(parameter.name)
        if resolver is not None:
            if isinstance(resolver, FunctionResolver):
                # TODO more careful checks of functions' arguments
                return True

            # If the annotation for this parameter does not match the base
            # class type inside the resolver, raise an exception
            if not is_hint(annotation, resolver.base):
                raise ArgumentError(
                    func,
                    parameter,
                    f"has bad annotation {annotation} wrt resolver {resolver}",
                )

            # If there's no value given and the resolver does not have a default,
            # raise an exception
            if not value and not resolver.default:
                raise ArgumentError(func, parameter, "is missing from the arguments")

            # Look up the class. If value is None, we already checked the resolver
            # should have a default and this will be okay
            cls = resolver.lookup(value)

            # Recur on the __init__() function of the class looked up by the resolver,
            # optionally using the related keyword arguments, if available.
            return self.check_kwargs(cls, kwargs.get(related_key))

        # If there's no value given and there's no default, raise an exception
        if value_not_given:
            if parameter.default is not parameter.empty:
                # We're going to go ahead and assume that the default value
                # matches the type annotation and not do any further checking
                return True
            raise ArgumentError(
                func,
                parameter,
                f"no default, value not given. Signature: {inspect.signature(func)}",
            )

        # Now comes the nitty-gritty part of checking - if

        # get_origin() checks if the annotation is inside a Union, List, etc.
        origin = get_origin(annotation)

        if origin is Union:
            # If the things inside the union are not all simple types,
            # i.e., float, int, bool, str, or None, then raise an error.
            # This is because anything other than these types can't live
            # inside JSON.
            if not _is_simple_union(annotation):
                raise ArgumentError(func, parameter, f"has inappropriate annotation: {annotation}")

            # Get the list of arguments inside the Union
            args = get_args(annotation)

            # if the value is one of the args, then we're done.
            if isinstance(value, args):
                return True
            # otherwise, raise a type error
            raise ArgumentError(
                func,
                parameter,
                f"value {value} does not match annotation {annotation}",
            )

        # If the origin is None, this means that it's just a single type
        if origin is None:
            try:
                instance_check = isinstance(value, annotation)
            except TypeError:
                # This type error gets thrown if there's some reason you can't
                # type check on the given annotation (i.e., it's not a subclass of `type`)
                raise ArgumentError(
                    func, parameter, f"invalid annotation {annotation} ({type(annotation)})"
                ) from None

            if instance_check:
                return True
            elif annotation == float and isinstance(value, int):
                # you can coerce an int into a float, so just say this is fine
                return True
            else:
                raise ArgumentError(func, parameter, f"{value} mismatched annotation {annotation}")

        # Unknown origin type
        # TODO might need to extend to handle list/dict
        raise ArgumentError(func, parameter, f"unhandled origin {origin}")

    def check_kwargs(self, func: Callable, kwargs: OptionalKwargs = None) -> bool:
        """Check the appropriate of the kwargs with a given function.

        :param func: A function or class to check
        :param kwargs: The keyword arguments to pass to the function
        :returns: True if there are no issues, raises if there are.
        :raises ArgumentError: If there is an error in the kwargs
        """
        if kwargs is None:
            kwargs = {}
        for parameter_name, parameter, related_key in _iter_params(func):
            self.check_kwarg(
                func=func,
                parameter=parameter,
                related_key=related_key,
                kwargs=kwargs,
            )
        return True


def _iter_params(func) -> Iterable[Tuple[str, inspect.Parameter, str]]:
    parameter_names: Mapping[str, inspect.Parameter] = inspect.signature(func).parameters
    parameter_name_to_kwargs = {}
    for parameter_name in parameter_names:
        kwargs_key = f"{parameter_name}_kwargs"
        if kwargs_key in parameter_names:
            parameter_name_to_kwargs[kwargs_key] = parameter_name
    for parameter_name, parameter in parameter_names.items():
        if parameter_name in parameter_name_to_kwargs:
            continue
        if parameter_name == "kwargs":
            continue
        yield parameter_name, parameter, f"{parameter_name}_kwargs"


def _main():
    import json

    from pykeen.datasets import dataset_resolver
    from pykeen.experiments.cli import HERE
    from pykeen.losses import loss_resolver
    from pykeen.models import model_resolver
    from pykeen.nn.init import initializer_resolver
    from pykeen.nn.representation import constrainer_resolver, normalizer_resolver
    from pykeen.optimizers import optimizer_resolver
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
            # skip optimizer since its instantiation is dynamic
            # "optimizer": optimizer_resolver,
        },
    )
    for path in HERE.glob("*/*.json"):
        data = json.loads(path.read_text())
        kwargs = data["pipeline"]
        try:
            r.check_kwargs(pipeline, kwargs)
        except Exception as e:
            print(path)
            raise e


if __name__ == "__main__":
    _main()
